from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import transformers
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torch
from typing import Dict, Type
from tqdm import tqdm
import json
from json import JSONEncoder
import math
import copy
import numpy as np
import random
from apex import amp
from datetime import datetime
from pathlib import Path
import os

nli_label2index = {
    'e': 0,
    'n': 1,
    'c': 2,
    'h': -1,
}

id2label = {
    0: 'e',
    1: 'n',
    2: 'c',
    -1: '-',
}


def unserialize_JsonableObject(d):
    global registered_jsonabl_classes
    classname = d.pop('_jcls_', None)
    if classname:
        cls = registered_jsonabl_classes[classname]
        obj = cls.__new__(cls)              # Make instance without calling __init__
        for key, value in d.items():
            setattr(obj, key, value)
        return obj
    else:
        return d

def load_jsonl(filename, debug_num=None):
    data = []
    with open(filename, encoding='utf-8', mode='r') as in_f:
        print("Load Jsonl:", filename)
        for line in tqdm(in_f):
            item = json.loads(line.strip(), object_hook=unserialize_JsonableObject)
            data.append(item)
            if debug_num is not None and 0 < debug_num == len(data):
                break
    return data

class JsonableObj():
    pass

class JsonableObjectEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, JsonableObj):
            d = {'_jcls_': type(o).__name__}
            d.update(vars(o))
            return d
        else:
            return super().default(o)

def save_jsonl(d_list, filename):
    print("Save to Jsonl:", filename)
    with open(filename, encoding='utf-8', mode='w') as out_f:
        for item in d_list:
            out_f.write(json.dumps(item, cls=JsonableObjectEncoder) + '\n')

def save_json(obj, filename, **kwargs):
    with open(filename, encoding='utf-8', mode='w') as out_f:
        json.dump(obj, out_f, cls=JsonableObjectEncoder, **kwargs)
        out_f.close()
       
def has_tensor(obj) -> bool:
    """
    Given a possibly complex data structure,
    check if it has any torch.Tensors in it.
    """
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(has_tensor(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(has_tensor(item) for item in obj)
    else:
        return False

def move_to_device(obj, cuda_device: int):
    """
    Given a structure (possibly) containing Tensors on the CPU,
    move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).
    """

    if cuda_device < 0 or not has_tensor(obj):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.cuda(cuda_device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, cuda_device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, cuda_device) for item in obj]
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # This is the best way to detect a NamedTuple, it turns out.
        return obj.__class__(*(move_to_device(item, cuda_device) for item in obj))
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, cuda_device) for item in obj)
    else:
        return obj

def list_to_dict(d_list, key_fields):       #   '_id' or 'pid'
    d_dict = dict()
    for item in d_list:
        assert key_fields in item
        d_dict[item[key_fields]] = item
    return d_dict


def dict_to_list(d_dict):
    d_list = []
    for key, value in d_dict.items():
        d_list.append(value)
    return d_list


def append_item_from_dict_to_list(d_list, d_dict, key_fieldname, append_fieldnames):
    if not isinstance(append_fieldnames, list):
        append_fieldnames = [append_fieldnames]
    for item in d_list:
        key = item[key_fieldname]
        if key in d_dict:
            for append_fieldname in append_fieldnames:
                item[append_fieldname] = d_dict[key][append_fieldname]
        else:
            print(f"Potential Error: {key} not in scored_dict. Maybe bc all forward items are empty.")
            for append_fieldname in append_fieldnames:
                item[append_fieldname] = []
    return d_list


def append_item_from_dict_to_list_hotpot_style(d_list, d_dict, key_fieldname, append_fieldnames):
    if not isinstance(append_fieldnames, list):
        append_fieldnames = [append_fieldnames]
    for item in d_list:
        key = item[key_fieldname]
        for append_fieldname in append_fieldnames:
            if key in d_dict[append_fieldname]:
                item[append_fieldname] = d_dict[append_fieldname][key]
            else:
                print(f"Potential Error: {key} not in scored_dict. Maybe bc all forward items are empty.")
                # for append_fieldname in append_fieldnames:
                item[append_fieldname] = []
    return d_list


def append_subfield_from_list_to_dict(subf_list, d_dict, o_key_field_name, subfield_key_name,
                                      subfield_name='merged_field', check=False):
    # Often times, we will need to split the one data point to multiple items to be feeded into neural networks
    # and after we obtain the results we will need to map the results back to original data point with some keys.

    # This method is used for this purpose.
    # The method can be invoke multiple times, (in practice usually one batch per time.)
    """
    :param subf_list:               The forward list.
    :param d_dict:                  The dict that contain keys mapping to original data point.
    :param o_key_field_name:        The fieldname of original data point key. 'pid'
    :param subfield_key_name:       The fieldname of the sub item. 'fid'
    :param subfield_name:           The merge field name.       'merged_field'
    :param check:
    :return:
    """
    for key in d_dict.keys():
        d_dict[key][subfield_name] = dict()

    for item in subf_list:
        assert o_key_field_name in item
        assert subfield_key_name in item
        map_id = item[o_key_field_name]
        sub_filed_id = item[subfield_key_name]
        assert map_id in d_dict

        # if subfield_name not in d_dict[map_id]:
        #     d_dict[map_id][subfield_name] = dict()

        if sub_filed_id not in d_dict[map_id][subfield_name]:
            if check:
                assert item[o_key_field_name] == map_id
            d_dict[map_id][subfield_name][sub_filed_id] = item
        else:
            print("Duplicate forward item with key:", sub_filed_id)

    return d_dict

def count_acc(gt_list, pred_list):
    assert len(gt_list) == len(pred_list)
    gt_dict = list_to_dict(gt_list, 'uid')
    pred_list = list_to_dict(pred_list, 'uid')
    total_count = 0
    hit = 0
    for key, value in pred_list.items():
        if gt_dict[key]['label'] == value['predicted_label']:
            hit += 1
        total_count += 1
    return hit, total_count

def evaluation_dataset(eval_dataloader, eval_list, model, r_dict, eval_name):
    pred_output_list = eval_model(model, eval_dataloader, 0)
    predictions = pred_output_list
    hit, total = count_acc(eval_list, pred_output_list)

    r_dict[f'{eval_name}'] = {
        'acc': hit / total,
        'correct_count': hit,
        'total_count': total,
        'predictions': predictions,
    }

def eval_model(model, dev_dataloader, device_num):
    model.eval()

    uid_list = []
    y_list = []
    pred_list = []
    logits_list = []

    with torch.no_grad():
        for i, batch in enumerate(dev_dataloader, 0):
            batch = move_to_device(batch, device_num)

            outputs = model(batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            token_type_ids=batch['token_type_ids'],
                            labels=batch['y'])

            loss = outputs["loss"]
            logits = outputs["logits"]

            uid_list.extend(list(batch['uid']))
            y_list.extend(batch['y'].tolist())
            pred_list.extend(torch.max(logits, 1)[1].view(logits.size(0)).tolist())
            logits_list.extend(logits.tolist())

    assert len(pred_list) == len(logits_list)
    assert len(pred_list) == len(logits_list)

    result_items_list = []
    for i in range(len(uid_list)):
        r_item = dict()
        r_item['uid'] = uid_list[i]
        r_item['logits'] = logits_list[i]
        r_item['predicted_label'] = id2label[pred_list[i]]

        result_items_list.append(r_item)

    return result_items_list

class ScoreLogger():
    def __init__(self, init_tracking_dict) -> None:
        super().__init__()
        self.logging_item_list = []
        self.score_tracker = dict()
        self.score_tracker.update(init_tracking_dict)

    def incorporate_results(self, score_dict, save_key, item=None) -> bool:
        assert len(score_dict.keys()) == len(self.score_tracker.keys())
        for fieldname in score_dict.keys():
            assert fieldname in self.score_tracker

        valid_improvement = False
        for fieldname, value in score_dict.items():
            if score_dict[fieldname] >= self.score_tracker[fieldname]:
                self.score_tracker[fieldname] = score_dict[fieldname]
                valid_improvement = True

        self.logging_item_list.append({'k': save_key, 'v': item})

        return valid_improvement

    def logging_to_file(self, filename):
        if Path(filename).is_file():
            old_logging_list = load_json(filename)
            current_saved_key = set()

            for item in self.logging_item_list:
                current_saved_key.add(item['k'])

            for item in old_logging_list:
                if item['k'] not in current_saved_key:
                    raise ValueError("Previous logged item can not be found!")

        save_json(self.logging_item_list, filename, indent=2, sort_keys=True)


def gen_file_prefix(model_name, directory_name='saved_models', date=None):
    date_now = datetime.now().strftime("%m-%d-%H_%M_%S") if not date else date
    file_path = os.path.join("/cluster/home/enricoma/NutNLI/Nut_NLI", directory_name, '_'.join((date_now, model_name)))
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return file_path, date_now


def get_cur_time_str():
    date_now = datetime.now().strftime("%m-%d[%H:%M:%S]")
    return date_now

def sample_data_list(d_list, ratio):
    if ratio <= 0:
        raise ValueError("Invalid training weight ratio. Please change --train_weights.")
    upper_int = int(math.ceil(ratio))
    if upper_int == 1:
        return d_list # if ratio is 1 then we just return the data list
    else:
        sampled_d_list = []
        for _ in range(upper_int):
            sampled_d_list.extend(copy.deepcopy(d_list))
        if np.isclose(ratio, upper_int):
            return sampled_d_list
        else:
            sampled_length = int(ratio * len(d_list))
            random.shuffle(sampled_d_list)
            return sampled_d_list[:sampled_length]
        
class DataConverter():
    @classmethod
    def converting(cls, data_to_convert):
        raise NotImplemented()


class IdentityConverter(DataConverter):
    @classmethod
    def converting(cls, data_to_convert):
        return data_to_convert


class TorchTensorConverter(DataConverter):
    def converting(self, data_to_convert):
        return torch.tensor(data_to_convert)


class TorchPadderConverter(DataConverter):
    def __init__(self, pad_idx) -> None:
        super().__init__()
        self.pad_idx = pad_idx

    def converting(self, data_to_convert):
        if not torch.is_tensor(data_to_convert[0]):
            data_to_convert = [torch.tensor(v) for v in data_to_convert]

        size = max(v.size(0) for v in data_to_convert)
        res = data_to_convert[0].new(len(data_to_convert), size).fill_(self.pad_idx)

        for i, v in enumerate(data_to_convert):
            res[i][:len(v)].copy_(v)
        return res

class BaseConverter():
    def __init__(self, converting_schema: Dict[str, DataConverter]) -> None:
        super().__init__()
        self.converting_schema: Dict[str, DataConverter] = converting_schema

    def __call__(self, batch):
        field_names = batch[0].keys()
        converted_data = dict()

        for field_name in field_names:
            if field_name not in self.converting_schema:
                converted_data[field_name] = IdentityConverter.converting([item[field_name] for item in batch])
            else:
                converted_data[field_name] = self.converting_schema[field_name].converting([item[field_name] for item in batch])

        return converted_data

class NLIDataset(Dataset):
    def __init__(self, list_of_data, transform) -> None:
        super().__init__()
        self.list_of_data = list_of_data
        self.len = len(self.list_of_data)
        self.transform = transform

    def __getitem__(self, i: int):
        return self.transform(self.list_of_data[i])

    def __len__(self) -> int:
        return self.len

class DataTransformer():
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, sample):
        # Standardization of data from all the dataset
        if "uid" not in sample.keys():
            if "pairID" in sample.keys():
                sample["uid"] = sample["pairID"]
            elif "fid" in sample.keys():
                sample["uid"] = sample["fid"]
        if "label" not in sample.keys():
            if "annotator_labels" in sample.keys():
                translator = {"entailment": "e", "neutral": "n", "contradiction": "c", "-":"n"}
                sample["label"] = translator[sample["gold_label"]]                   
        if sample["label"]  not in ["e", "n", "c"]:
            translator = {"SUPPORTS": "e", "NOT ENOUGH INFO": "n", "REFUTES": "c"}
            sample["label"] = translator[sample["label"]]
                
        if "context" not in sample.keys():
            if "sentence1" in sample.keys():
                sample["context"] = sample["sentence1"]
        if "hypothesis" not in sample.keys():
            if "sentence2" in sample.keys():
                sample["hypothesis"] = sample["sentence2"]
            elif "query" in sample.keys():
                sample["hypothesis"] = sample["query"]
        # Finish Preprocessing
        processed_sample = dict()
        processed_sample['uid'] = sample['uid']
        processed_sample['gold_label'] = sample['label']
        processed_sample['y'] = nli_label2index[sample['label']]

        premise: str = sample['context'] if 'context' in sample else sample['premise']
        hypothesis: str = sample['hypothesis']

        if premise.strip() == '':
            premise = 'empty'

        if hypothesis.strip() == '':
            hypothesis = 'empty'

        tokenized_input_seq_pair = self.tokenizer.encode_plus(premise, hypothesis,
                                                              max_length=self.max_length,
                                                              return_token_type_ids=True,
                                                              truncation=True)

        processed_sample.update(tokenized_input_seq_pair)

        return processed_sample
