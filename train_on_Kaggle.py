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
    def __init__(self, batch_size=8, num_of_layers_to_unfreeze=0):
        super().__init__()

        self.batch_size = batch_size
        self.num_of_layers_to_unfreeze = num_of_layers_to_unfreeze

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
        self.roberta.load_state_dict(torch.load("./MyModel/model.pt"))
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

    def load_training_dataset(self, validation_percentage=0.2):
        df = pd.read_csv("../Data/train.csv")
        train = int(df.shape[0] * (1 - validation_percentage))
        premises = df["premise"].values
        hypothesis = df["hypothesis"].values
        labels = df["label"].values

        loading_database_tqdm = tqdm(range(len(premises)), unit=" Sentences Pairs", desc="Loading Dataset")

        for i in loading_database_tqdm:
            if i < train:
                self.training_pairs.append((premises[i], hypothesis[i]))
                self.training_labels.append(labels[i])
            else:
                self.validation_pairs.append((premises[i], hypothesis[i]))
                self.validation_labels.append(labels[i])

        print(f"Loaded {len(self.training_pairs)} training data and {len(self.validation_pairs)} validation data.")
    
    def load_test_dataset(self):
        df = pd.read_csv("../Data/test.csv")
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

        self.optimizer = AdamW(self.parameters(), lr=2e-5, eps=1e-6)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=int(0.2*total_step),
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
    
    @staticmethod
    def label_to_probability(labels):
        probabilities = torch.zeros(len(labels), 3, dtype=torch.float)
        for i, label in enumerate(labels):
            probabilities[i, label] = 1.
        return probabilities.to('cuda')


if __name__ == '__main__':
    model = NutNLIModel(batch_size=32, num_of_layers_to_unfreeze=12)
    model.load_training_dataset()
    model.train_me(3)
    model.load_test_dataset()
    model.create_test_labels()
    
    #symanto/xlm-roberta-base-snli-mnli-anli-xnli
