import time
import torch
from torch import nn
from tqdm import tqdm
from transformers import XLMRobertaModel, get_linear_schedule_with_warmup, RobertaForSequenceClassification, RobertaTokenizer
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.optim import AdamW
from prettytable import PrettyTable
import pandas as pd
import matplotlib.pyplot as plt


class NutNLIModel(nn.Module):
    def __init__(self, batch_size=16, weight_path=None):
        super().__init__()

        self.batch_size = batch_size

        # set up dataset lists
        self.training_pairs = []
        self.training_labels = []
        self.validation_pairs = []
        self.validation_labels = []
        self.test_ids = []
        self.test_pairs = []

        # set up training stuff
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = None
        self.history_loss = []
        self.scheduler = None

        self.roberta = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-large", num_labels=3)
        if weight_path is not None:
            self.roberta.load_state_dict(torch.load(weight_path))
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

        # get some information from the bert model
        self.bert_out_size = self.roberta.config.hidden_size  # 768
        self.num_bert_layer = self.roberta.config.num_hidden_layers  # 12

        # add my linear head for NLI classification
        self.linear_nli_1 = nn.Linear(self.bert_out_size, 64)
        self.linear_nli_2 = nn.Linear(64, 3)  # 3 Because we have three possible labels
        self.softmax = nn.Softmax(dim=1)  # Using softmax for getting probability as outcome

        # Freezing all the Bert layers
        print("@" * 120)
        for p_name, p in self.roberta.named_parameters():
            # print(f"'{p_name}' -> {p.shape}")
            splitted_name = p_name.split('.')
            if splitted_name[0] == 'classifier' or (len(splitted_name) >= 4 and splitted_name[3] == '23'):
                print(f"Unfreezing [{p_name} ({p.shape})]")
                p.data = torch.randn(p.shape) * 0.02  # Random weight initialization
                p.requires_grad = True  # Not Freeze
            else:
                p.requires_grad = False  # Freeze
        print("@" * 120)
        print(self.roberta)
        self.to('cuda')

    def forward(self, sentence_pairs):
        """
        :param sentence_pairs: A list of tuple containing the sentences as text
                ex: [("I'm a man", "I will not die"), ("I like red", "I love red curtains"), ("A", "We!")]
        :return: The probability for each pair to be an ENTAIL, a NEUTRAL or a CONTRADICTION
        """
        # indexed_sentences, sentences_separators, attention_masks = self.from_text_to_bert_input(sentence_pairs)
        input_ids, attention_mask = self.from_text_to_bert_input(sentence_pairs)
        roberta_output = self.roberta(input_ids, attention_mask).logits
        probability = self.softmax(roberta_output)
        return probability

    def evaluation(self, validation_set=True):
        """
        :return: list of text or index that define the correlation between the corresponding sentence pair.
                ex with text=True: ["contradiction", "entail", "neutral"]
                ex with text=False: [2, 0, 1]
        """
        if validation_set:
            pairs = self.validation_pairs
            labels = self.validation_labels
        else:
            pairs = self.training_pairs
            labels = self.training_labels

        evaluation_tqdm = tqdm(range(0, len(pairs), self.batch_size), unit=" Sentences Pairs",
                               desc="Evaluation")

        self.eval()

        all_predicted_labels = []

        for start_batch_index in evaluation_tqdm:
            predicted_probability = self.forward(pairs[start_batch_index:start_batch_index + self.batch_size])
            predicted_labels = torch.argmax(predicted_probability, 1)
            all_predicted_labels += predicted_labels.tolist()

        tensor_true = torch.tensor(labels)
        tensor_prediction = torch.tensor(all_predicted_labels)

        correct_classification = torch.sum(tensor_true == tensor_prediction)
        accuracy = correct_classification / len(labels)
        return float(accuracy)

    def create_test_labels(self):
        pairs = self.test_pairs

        evaluation_tqdm = tqdm(range(0, len(pairs), self.batch_size), unit=" Sentences Pairs", desc="Evaluation")

        self.eval()

        all_predicted_labels = []

        for start_batch_index in evaluation_tqdm:
            predicted_probability = self.forward(pairs[start_batch_index:start_batch_index + self.batch_size])
            predicted_labels = torch.argmax(predicted_probability, 1)
            all_predicted_labels += predicted_labels.tolist()

        tensor_prediction = torch.tensor(all_predicted_labels)
        submission = open('submission.csv', 'w')
        submission.write(f"id,prediction\n")
        for i in range(len(self.test_ids)):
            submission.write(f"{self.test_ids[i]},{int(tensor_prediction[i])}\n")
        submission.close()


    def from_text_to_bert_input(self, sentence_pairs):
        """
        :param sentence_pairs: A list of tuple containing the sentences as text
                ex: [("I'm a man", "I will not die"), ("I like red", "I love red curtains"), ("A", "We!")]
        :return:
        """
        # input_ids, attention_mask
        tokenized_text = self.tokenizer(sentence_pairs, truncation="only_first", return_tensors="pt", padding=True)
        input_ids = tokenized_text["input_ids"].to('cuda')
        attention_mask = tokenized_text["attention_mask"].to('cuda')
        return input_ids, attention_mask

    def load_training_dataset(self, path, validation_percentage=0.2):
        df = pd.read_csv(path)
        train = int(df.shape[0] * (1 - validation_percentage))
        premises = df["premise"].values
        hypothesis = df["hypothesis"].values
        labels = df["label"].values

        loading_database_tqdm = tqdm(range(len(premises)), unit=" Sentences Pairs", desc="Loading Dataset")
        labels_dict = {0:0, 1:1, 2:2, "entailment":0, "neutral":1, "contradiction":2}

        for i in loading_database_tqdm:
            if i < train:
                if labels[i] in labels_dict:
                    self.training_pairs.append((premises[i], hypothesis[i]))
                    self.training_labels.append(labels_dict[labels[i]])
            else:
                if labels[i] in labels_dict:
                    self.validation_pairs.append((premises[i], hypothesis[i]))
                    self.validation_labels.append(labels_dict[labels[i]])

        print(f"Loaded {len(self.training_pairs)} training data and {len(self.validation_pairs)} validation data.")

    def load_kaggle_test_dataset(self):
        df = pd.read_csv("./Data/test.csv")
        ids = df["id"].values
        premises = df["premise"].values
        hypothesis = df["hypothesis"].values

        loading_database_tqdm = tqdm(range(len(premises)), unit=" Sentences Pairs", desc="Loading Dataset")

        for i in loading_database_tqdm:
            self.test_pairs.append((premises[i], hypothesis[i]))
            self.test_ids.append(ids[i])

        print(f"Loaded {len(self.test_pairs)} test data")

    def train_step(self, sentence_pairs, true_probability):
        predicted_probability = self.forward(sentence_pairs)

        loss = self.loss_fn(predicted_probability, true_probability)
        self.history_loss.append(float(loss.detach().item()))

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Propagation
        self.optimizer.step()

        # Update training rate
        self.scheduler.step()

    def train_me(self, epochs):

        total_step = len(self.training_pairs) // self.batch_size

        self.optimizer = AdamW(self.parameters(), lr=1e-5, eps=1e-8)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=int(0.1*total_step),
                                                         num_training_steps=total_step)

        for epoch in range(epochs):
            print(f"EPOCH {epoch}")
            time.sleep(0.1)
            print(f"Evaluation: {self.evaluation(validation_set=True):.2f}")
            time.sleep(0.1)
            training_tqdm = tqdm(range(0, len(self.training_pairs), self.batch_size), unit=" Sentences Pairs",
                                 desc="Training")
            steps = 0

            self.train()
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()

            for start_batch_index in training_tqdm:
                sentence_pairs = []
                labels = []
                for i in range(start_batch_index, min(start_batch_index + self.batch_size, len(self.training_pairs))):
                    sentence_pairs.append(self.training_pairs[i])
                    labels.append(self.training_labels[i])
                true_probability = self.label_to_probability(labels)
                self.train_step(sentence_pairs, true_probability)
                steps += 1

            plt.plot(list(range(len(self.history_loss))), self.history_loss)
            plt.savefig('loss.png')

    def save_model(self, path):
        torch.save(self.roberta.state_dict(), path)

    @staticmethod
    def label_to_probability(labels):
        probabilities = torch.zeros(len(labels), 3, dtype=torch.float)
        for i, label in enumerate(labels):
            probabilities[i, label] = 1.
        return probabilities.to('cuda')

def from_list_to_string(jsonl_list):
    i = 0
    a_string = ""
    while i < len(jsonl_list):
        if jsonl_list[i] is not None:
            a_string += str(jsonl_list[i])
        else:
            break
        i += 1
    return a_string

def unify_datasets():
    my_df = pd.DataFrame(columns=["premise", "hypothesis", "label", 'dataset'])


    print("Loading ANLI R1")
    f = open("./Data/ANLI/R1/train.jsonl", "r")
    line = f.readline()
    n = 0
    ANLI_dict = {"e":"entailment", "c":"contradiction", "n":"neutral"}
    while line:
        ANLI_line_df = pd.read_json(line, orient='index')

        premise = ANLI_line_df.iloc[1].tolist()[0]
        hypothesis = ANLI_line_df.iloc[2].tolist()[0]
        label = ANLI_dict[ANLI_line_df.iloc[3].tolist()[0]]
        #print(f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel: {label}")
        line = f.readline()
        n += 1
        my_df.loc[len(my_df)] = {'premise': premise, 'hypothesis': hypothesis, 'label': label, 'dataset': "ANLI_R1"}
        if n % 1000 == 0:
            print(n)
    print(f"Finished ANLI R1 {n}")
    f.close()

    print("Loading ANLI R2")
    f = open("./Data/ANLI/R2/train.jsonl", "r")
    line = f.readline()
    n = 0
    ANLI_dict = {"e":"entailment", "c":"contradiction", "n":"neutral"}
    while line:
        ANLI_line_df = pd.read_json(line, orient='index')

        premise = ANLI_line_df.iloc[1].tolist()[0]
        hypothesis = ANLI_line_df.iloc[2].tolist()[0]
        label = ANLI_dict[ANLI_line_df.iloc[3].tolist()[0]]
        #print(f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel: {label}")
        line = f.readline()
        n += 1
        my_df.loc[len(my_df)] = {'premise': premise, 'hypothesis': hypothesis, 'label': label, 'dataset': "ANLI_R2"}
        if n % 1000 == 0:
            print(n)
    print(f"Finished ANLI R2 {n}")
    f.close()

    print("Loading ANLI R3")
    f = open("./Data/ANLI/R3/train.jsonl", "r")
    line = f.readline()
    n = 0
    ANLI_dict = {"e":"entailment", "c":"contradiction", "n":"neutral"}
    while line:
        ANLI_line_df = pd.read_json(line, orient='index')

        premise = ANLI_line_df.iloc[1].tolist()[0]
        hypothesis = ANLI_line_df.iloc[2].tolist()[0]
        label = ANLI_dict[ANLI_line_df.iloc[3].tolist()[0]]
        #print(f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel: {label}")
        line = f.readline()
        n += 1
        my_df.loc[len(my_df)] = {'premise': premise, 'hypothesis': hypothesis, 'label': label, 'dataset': "ANLI_R3"}
        if n % 1000 == 0:
            print(n)
    print(f"Finished ANLI R3 {n}")
    f.close()

    print("Loading MNLI")
    f = open("./Data/MultiNLI/multinli_train.jsonl", "r")
    line = f.readline()
    n = 0
    while line:
        MNLI_line_df = pd.read_json(line, orient='index')
        premise = from_list_to_string(MNLI_line_df.iloc[5].tolist())
        hypothesis = from_list_to_string(MNLI_line_df.iloc[8].tolist())
        label = from_list_to_string(MNLI_line_df.iloc[2].tolist())
        #print(f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel: {label}")
        line = f.readline()
        n += 1
        my_df.loc[len(my_df)] = {'premise': premise, 'hypothesis': hypothesis, 'label': label, 'dataset': "MNLI"}
        if n % 1000 == 0:
            print(n)
    print(f"Finished MNLI {n}")
    f.close()

    print("Loading FEVER")
    f = open("./Data/FEVER/fever_train.jsonl", "r")
    line = f.readline()
    n = 0
    FEVER_dict = {"SUPPORTS":"entailment", "REFUTES":"contradiction", "NOT ENOUGH INFO":"neutral"}
    while line:
        FEVER_line_df = pd.read_json(line, orient='index')
        premise = FEVER_line_df.iloc[2].tolist()[0]
        hypothesis = FEVER_line_df.iloc[3].tolist()[0]
        label = FEVER_dict[FEVER_line_df.iloc[4].tolist()[0]]
        #print(f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel: {label}")
        line = f.readline()
        n += 1
        my_df.loc[len(my_df)] = {'premise': premise, 'hypothesis': hypothesis, 'label': label, 'dataset': "FEVER"}
        if n % 1000 == 0:
            print(n)
    print(f"Finished FEVER {n}")
    f.close()

    print("Loading SNLI")
    f = open("./Data/SNLI/snli_train.jsonl", "r")
    line = f.readline()
    n = 0
    while line:
        SNLI_line_df = pd.read_json(line, orient='index')
        premise = from_list_to_string(SNLI_line_df.iloc[4].tolist())
        hypothesis = from_list_to_string(SNLI_line_df.iloc[7].tolist())
        label = from_list_to_string(SNLI_line_df.iloc[2].tolist())
        #print(f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel: {label}")
        line = f.readline()
        n += 1
        my_df.loc[len(my_df)] = {'premise': premise, 'hypothesis': hypothesis, 'label': label, 'dataset': "SNLI"}
        if n % 1000 == 0:
            print(n)
    print(f"Finished SNLI {n}")
    f.close()

    my_df.to_csv("./Data/unified_dataset.csv", index=False)

if __name__ == '__main__':
    # unify_datasets()
    """
    model = NutNLIModel(batch_size=16)
    model.load_training_dataset("./Data/unified_dataset.csv", validation_percentage=0.01)
    model.train_me(3)
    model.save_model("./ENG_model.pt")
    """

    model = NutNLIModel(batch_size=32, weight_path="./ENG_model.pt")
    model.load_training_dataset("./Data/train.csv")
    model.train_me(3)
    model.load_kaggle_test_dataset()
    model.create_test_labels()
    model.save_model("./KAGGLE_model.pt")
