import time
import torch
from torch import nn
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
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

        # set up training stuff
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = None
        self.history_loss = []

        # get the Bert Model
        self.bert = BertModel.from_pretrained("symanto/xlm-roberta-base-snli-mnli-anli-xnli")  #  "bert-base-uncased")  # "bert-base-multilingual-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("symanto/xlm-roberta-base-snli-mnli-anli-xnli")  # "bert-base-uncased")  # "bert-base-multilingual-uncased")

        # get some information from the bert model
        self.bert_out_size = self.bert.config.hidden_size  # 768
        self.num_bert_layer = self.bert.config.num_hidden_layers  # 12

        # add my linear head for NLI classification
        self.linear_nli = nn.Linear(self.bert_out_size, 3)  # 3 Because we have three possible labels
        self.softmax = nn.Softmax(dim=1)  # Using softmax for getting probability as outcome

        # Freezing all the Bert layers
        print("@" * 120)
        for p_name, p in self.bert.named_parameters():
            """
            using:  print(f"'{p_name}' -> {p.shape}")
            we obtain:
            
            embeddings.word_embeddings.weight
            embeddings.position_embeddings.weight
            embeddings.token_type_embeddings.weight
            embeddings.LayerNorm.weight
            embeddings.LayerNorm.bias
            for each BertLayer (12 of them) we have:
                'encoder.layer.__i__.attention.self.query.weight' -> torch.Size([768, 768])
                'encoder.layer.__i__.attention.self.query.bias' -> torch.Size([768])
                'encoder.layer.__i__.attention.self.key.weight' -> torch.Size([768, 768])
                'encoder.layer.__i__.attention.self.key.bias' -> torch.Size([768])
                'encoder.layer.__i__.attention.self.value.weight' -> torch.Size([768, 768])
                'encoder.layer.__i__.attention.self.value.bias' -> torch.Size([768])
                'encoder.layer.__i__.attention.output.dense.weight' -> torch.Size([768, 768])
                'encoder.layer.__i__.attention.output.dense.bias' -> torch.Size([768])
                'encoder.layer.__i__.attention.output.LayerNorm.weight' -> torch.Size([768])
                'encoder.layer.__i__.attention.output.LayerNorm.bias' -> torch.Size([768])
                'encoder.layer.__i__.intermediate.dense.weight' -> torch.Size([3072, 768])
                'encoder.layer.__i__.intermediate.dense.bias' -> torch.Size([3072])
                'encoder.layer.__i__.output.dense.weight' -> torch.Size([768, 3072])
                'encoder.layer.__i__.output.dense.bias' -> torch.Size([768])
                'encoder.layer.__i__.output.LayerNorm.weight' -> torch.Size([768])
                'encoder.layer.__i__.output.LayerNorm.bias' -> torch.Size([768])
            
            in the end:
            'pooler.dense.weight' -> torch.Size([768, 768])
            'pooler.dense.bias' -> torch.Size([768])
            """
            splitted_name = p_name.split('.')
            if splitted_name[0] == 'pooler' or splitted_name[0] == 'encoder' and \
                    int(splitted_name[2]) >= self.num_bert_layer - self.num_of_layers_to_unfreeze:
                print(f"Unfreezing [{p_name} ({p.shape})]")
                p.data = torch.randn(p.shape) * 0.02  # Random weight initialization
                p.requires_grad = True  # Not Freeze
            else:
                p.requires_grad = False  # Freeze
        print("@" * 120)
        self.to('cuda')

    def forward(self, sentence_pairs):
        """
        :param sentence_pairs: A list of tuple containing the sentences as text
                ex: [("I'm a man", "I will not die"), ("I like red", "I love red curtains"), ("A", "We!")]
        :return: The probability for each pair to be an ENTAIL, a NEUTRAL or a CONTRADICTION
        """
        indexed_sentences, sentences_separators, attention_masks = self.from_text_to_bert_input(sentence_pairs)
        bert_pooler_output = self.bert(input_ids=indexed_sentences, token_type_ids=sentences_separators,
                                       attention_mask=attention_masks)[1]  # we are just interested in the pooler output
        linear_output = self.linear_nli(bert_pooler_output)
        probability = self.softmax(linear_output)

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

    def from_text_to_bert_input(self, sentence_pairs):
        """
        Example of a single pair: ("I'm a man", "I will not die")
        tokenizer_input = [CLS] I'm a man [SEP] I will not die [SEP]
        +----------------+---------+---------+---------+---------+---------+---------+---------+---------+----------+----------+----------+----------+
        | tokenized_text |  [CLS]  |    i    |    '    |    m    |    a    |   man   |  [SEP]  |    i    |   will   |   not    |   die    |  [SEP]   |
        | indexed_tokens |   101   |   151   |   112   |   155   |   143   |  10564  |   102   |   151   |  11229   |  10497   |  10121   |   102    |
        | sentences_ids  |    0    |    0    |    0    |    0    |    0    |    0    |    0    |    1    |    1     |    1     |    1     |    1     |
        +----------------+---------+---------+---------+---------+---------+---------+---------+---------+----------+----------+----------+----------+
        :param sentence_pairs: A list of tuple containing the sentences as text
                ex: [("I'm a man", "I will not die"), ("I like red", "I love red curtains"), ("A", "We!")]
        :return:
        """
        indexed_sentences = []
        sentences_separators = []
        attention_masks = []

        longest_sentence = 0

        for sent_1, sent_2 in sentence_pairs:
            tokenizer_input = "[CLS] {} [SEP] {} [SEP]".format(sent_1, sent_2)
            tokenized_sentence = self.tokenizer.tokenize(tokenizer_input)
            # ex: ['[CLS]', 'i', "'", 'm', 'a', 'man', '[SEP]', 'i', 'will', 'not', 'die', '[SEP]']
            indexed_tokens = self.tokenizer.encode(tokenizer_input)[1:-1]  # [1:-1] Because I have put by hand [CLS] and
            #                                                                [SEP] at the start and at the end
            # ex: [101, 151, 112, 155, 143, 10564, 102, 151, 11229, 10497, 10121, 102]
            sentences_separator = []
            # ex: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
            reached_separator = False
            for i in range(len(indexed_tokens)):
                if tokenized_sentence[i] == '[SEP]' and not reached_separator:
                    sentences_separator.append(0)  # I put SEP as a token of the FIRST sentence
                    reached_separator = True
                elif reached_separator:
                    sentences_separator.append(1)  # 1 means a token from the SECOND sentence
                else:
                    sentences_separator.append(0)  # 0 means a token from the FIRST sentence

            if indexed_tokens is None or sentences_separator is None:
                print(f"Is not possible to tokenize the following pairs:\n ({sent_1},{sent_2})")
                print(f"indexed_tokens = {indexed_tokens}")
                print(f"sentences_separator = {sentences_separator}")
            else:
                indexed_sentences.append(indexed_tokens)
                sentences_separators.append(sentences_separator)
                attention_masks.append([1 for _ in range(len(indexed_tokens))])
                if len(indexed_tokens) > longest_sentence:
                    longest_sentence = len(indexed_tokens)

        # Now we pad all the three list to have all the length of the longer one
        for i in range(len(indexed_sentences)):
            n = len(indexed_sentences[i])
            indexed_sentences[i] = indexed_sentences[i] + [0 for _ in range(longest_sentence - n)]
            sentences_separators[i] = sentences_separators[i] + [1 for _ in range(longest_sentence - n)]
            attention_masks[i] = attention_masks[i] + [0 for _ in range(longest_sentence - n)]

        return torch.tensor(indexed_sentences).to('cuda'), \
            torch.tensor(sentences_separators).to('cuda'), \
            torch.tensor(attention_masks).to('cuda')

    def load_training_dataset(self, validation_percentage=0.2):
        df = pd.read_csv("../Data/train_en.csv")
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

    def train_step(self, sentence_pairs, true_probability):
        predicted_probability = self.forward(sentence_pairs)

        loss = self.loss_fn(predicted_probability, true_probability)
        self.history_loss.append(float(loss.detach().item()))

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Propagation
        self.optimizer.step()

    def train_me(self, epochs):

        self.optimizer = AdamW(self.parameters(), lr=2e-5, eps=1e-6)

        for epoch in range(epochs):
            print(f"EPOCH {epoch}")
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

            print(f"Evaluation: {self.evaluation(validation_set=True):.2f}")
            time.sleep(0.1)

    @staticmethod
    def label_to_probability(labels):
        probabilities = torch.zeros(len(labels), 3, dtype=torch.float)
        for i, label in enumerate(labels):
            probabilities[i, label] = 1.
        return probabilities.to('cuda')


if __name__ == '__main__':
    model = NutNLIModel(batch_size=16, num_of_layers_to_unfreeze=2)
    # print(model)
    # model.forward([("I'm a man", "I will not die"), ("I like red", "I love red curtains"), ("A", "We!")])
    model.load_training_dataset()
    model.train_me(10)
