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
from Utils import load_jsonl, save_jsonl, save_json, move_to_device, evaluation_dataset, gen_file_prefix, sample_data_list, IdentityConverter, TorchTensorConverter, TorchPadderConverter, BaseConverter, NLIDataset, DataTransformer


train_path = {
    'snli_train': "/cluster/home/enricoma/data/data_nli/snli_1.0/snli_1.0_train.jsonl",
    'mnli_train': "/cluster/home/enricoma/data/data_nli/multinli_1.0/multinli_1.0_train.jsonl",
    'fever_train': "/cluster/home/enricoma/data/data_nli/nli_fever/train_fitems.jsonl",
    'anli_r1_train': "/cluster/home/enricoma/data/data_nli/anli_v1.0/R1/train.jsonl",
    'anli_r2_train': "/cluster/home/enricoma/data/data_nli/anli_v1.0/R2/train.jsonl",
    'anli_r3_train': "/cluster/home/enricoma/data/data_nli/anli_v1.0/R3/train.jsonl",
}

train_data = {
    'snli_train': load_jsonl(train_path['snli_train']),
    'mnli_train': load_jsonl(train_path['mnli_train']),
    'fever_train': load_jsonl(train_path['fever_train']),
    'anli_r1_train': load_jsonl(train_path['anli_r1_train']),
    'anli_r2_train': load_jsonl(train_path['anli_r2_train']),
    'anli_r3_train': load_jsonl(train_path['anli_r3_train']),
}

train_weight = { 
    'snli_train': 1,
    'mnli_train': 1,
    'fever_train': 1,
    'anli_r1_train': 10,
    'anli_r2_train': 20,
    'anli_r3_train': 10,    
}

dev_path = {
    'snli_dev': "/cluster/home/enricoma/data/data_nli/snli_1.0/snli_1.0_dev.jsonl",
    'mnli_m_dev': "/cluster/home/enricoma/data/data_nli/multinli_1.0/multinli_1.0_dev_matched.jsonl",
    'mnli_mm_dev': "/cluster/home/enricoma/data/data_nli/multinli_1.0/multinli_1.0_dev_mismatched.jsonl",
    # 'fever_dev': "/cluster/home/enricoma/data/data_nli/nli_fever/dev_fitems.jsonl",
    'anli_r1_dev': "/cluster/home/enricoma/data/data_nli/anli_v1.0/R1/dev.jsonl",
    'anli_r2_dev': "/cluster/home/enricoma/data/data_nli/anli_v1.0/R2/dev.jsonl",
    'anli_r3_dev': "/cluster/home/enricoma/data/data_nli/anli_v1.0/R3/dev.jsonl",
}

dev_data = {
    'snli_dev': load_jsonl(dev_path['snli_dev']),
    'mnli_m_dev': load_jsonl(dev_path['mnli_m_dev']),
    'mnli_mm_dev': load_jsonl(dev_path['mnli_mm_dev']),
    # 'fever_dev': load_jsonl(dev_path['fever_dev']),
    'anli_r1_dev': load_jsonl(dev_path['anli_r1_dev']),
    'anli_r2_dev': load_jsonl(dev_path['anli_r2_dev']),
    'anli_r3_dev': load_jsonl(dev_path['anli_r3_dev'])
}

test_path = {
    'snli_test': "/cluster/home/enricoma/data/data_nli/snli_1.0/snli_1.0_test.jsonl",
    'fever_test': "/cluster/home/enricoma/data/data_nli/nli_fever/test_fitems.jsonl",
    'anli_r1_test': "/cluster/home/enricoma/data/data_nli/anli_v1.0/R1/test.jsonl",
    'anli_r2_test': "/cluster/home/enricoma/data/data_nli/anli_v1.0/R2/test.jsonl",
    'anli_r3_test': "/cluster/home/enricoma/data/data_nli/anli_v1.0/R3/test.jsonl",
}

def train(train_batch_size=16, eval_batch_size=32, max_length=156, epochs=3, do_lower_case=False, model_name="xlm-roberta-large",
          weight_decay=0, learning_rate=1e-5, adam_epsilon=1e-8, eval_frequency=2000):
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    data_transformer = DataTransformer(tokenizer, max_length)
    
    converting_schema = {
        'uid': IdentityConverter(),
        'y': TorchTensorConverter(),
        'input_ids': TorchPadderConverter(pad_idx=1), # (1, False) Convert a list of 1d tensors into a padded 2d tensor.
        'token_type_ids': TorchPadderConverter(pad_idx=0), # (0, False)  Convert a list of 1d tensors into a padded 2d tensor.
        'attention_mask': TorchPadderConverter(pad_idx=0), #(0, False) Convert a list of 1d tensors into a padded 2d tensor.
    }
    
    eval_data_loaders = {}
    for key in dev_path:
        my_dataset = NLIDataset(dev_data[key], data_transformer)
        eval_data_loaders[key] = DataLoader(dataset=my_dataset,
                                            batch_size=32,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            sampler=SequentialSampler(my_dataset),
                                            collate_fn=BaseConverter(converting_schema)
                                           )
    
    total_steps = 3234450 * epochs // train_batch_size
    warmup_steps = int(0.1 * total_steps)
    
    torch.cuda.set_device(0)
    model.cuda(0)
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    global_step = 0
    
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    
    file_path_prefix = '.'
    
    print("Total Steps:", total_steps)
    print("Warmup Steps:", warmup_steps)
    print("Actual Training Batch Size:", train_batch_size)

    is_finished = False
    
    file_path_prefix, date = gen_file_prefix("Noce_Prova_Finale")
    script_name = os.path.basename(__file__)
    with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
        out_f.write(it.read())
        out_f.flush()

    checkpoints_path = Path(file_path_prefix) / "checkpoints"
    if not checkpoints_path.exists():
        checkpoints_path.mkdir()
    prediction_path = Path(file_path_prefix) / "predictions"
    if not prediction_path.exists():
        prediction_path.mkdir()
        
    for epoch in tqdm(range(epochs), desc="Epoch"):
        
        training_list = []
        print("Build Training Data ...")
        for key in train_data:
            train_d = train_data[key]
            train_w = train_weight[key]
            augmented_train_data = sample_data_list(train_d, train_w)
            print(f"Data Name:{key}; Weight: {train_w}; "
                  f"Original Size: {len(train_d)}; Sampled Size: {len(augmented_train_data)}")
            training_list.extend(augmented_train_data)

        random.shuffle(training_list)
        
        train_dataset = NLIDataset(training_list, data_transformer)
        
        train_sampler = SequentialSampler(train_dataset)
        
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=train_batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      pin_memory=True,
                                      sampler=train_sampler,
                                      collate_fn=BaseConverter(converting_schema))
        
        for forward_step, batch in enumerate(tqdm(train_dataloader, desc="Iteration"), 0):
            model.train()
            batch = move_to_device(batch, 0)
            outputs = model(batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            token_type_ids=batch['token_type_ids'],
                            labels=batch['y'])
            loss = outputs["loss"]
            logits = outputs["logits"]
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            
            if (forward_step + 1) % 1 == 0:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
                
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                
                global_step += 1
                
                if global_step % eval_frequency == 0:
                    r_dict = dict()
                    for key in dev_data:
                        cur_eval_data_name = key
                        cur_eval_data_list = dev_data[key]
                        cur_eval_dataloader = eval_data_loaders[key]
                        
                        evaluation_dataset(cur_eval_dataloader, cur_eval_data_list, model, r_dict, eval_name=cur_eval_data_name)
                        
                    current_checkpoint_filename = f"{epoch}_{global_step}_"
                    
                    for key in dev_data:
                        current_checkpoint_filename += f'_{key}_{round(r_dict[key]["acc"], 4)}_'
                    
                    model_output_dir = checkpoints_path / current_checkpoint_filename
                    if not model_output_dir.exists():
                        model_output_dir.mkdir()
                    torch.save(model.state_dict(), str(model_output_dir / "model.pt"))
                    # torch.save(optimizer.state_dict(), str(model_output_dir / "optimizer.pt"))
                    # torch.save(scheduler.state_dict(), str(model_output_dir / "scheduler.pt"))
                    
                    cur_results_path = prediction_path / current_checkpoint_filename
                    if not cur_results_path.exists():
                        cur_results_path.mkdir(parents=True)
                    for key, item in r_dict.items():
                        save_jsonl(item['predictions'], cur_results_path / f"{key}.jsonl")

                    for key, item in r_dict.items():
                        del r_dict[key]['predictions']
                    save_json(r_dict, cur_results_path / "results_dict.json", indent=2)
                    
                if global_step == total_steps:
                    is_finished = True
                    break
                
        if is_finished:
            break
    
def main():
    transformers.logging.set_verbosity_error() # Be aware, overflowing tokens are not returned for the setting you have chosen, i.e. sequence pairs with the 'longest_first' truncation strategy. So the returned list will always be empty even if some tokens have been removed.
    train(train_batch_size=16, eval_batch_size=32, max_length=156, epochs=3, do_lower_case=False, model_name="xlm-roberta-large",
          weight_decay=0, learning_rate=1e-5, adam_epsilon=1e-8, eval_frequency=2000)

if __name__ == '__main__':
    main()
