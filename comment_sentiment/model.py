import torch.nn as nn
import torch
import os
from transformers import BertModel
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

bert = BertModel.from_pretrained('bert-base-uncased')

class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text = [batch size, sent len]

        with torch.no_grad():
            embedded = self.bert(text)[0]

        # embedded = [batch size, sent len, emb dim]

        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]

        output = self.out(hidden)

        # output = [batch size, out dim]

        return output


HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERTGRUSentiment(bert,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)

device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')
model.load_state_dict(torch.load(path, map_location=device))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

eos_token_idx = tokenizer.sep_token_id
init_token_idx = tokenizer.cls_token_id

max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']


def return_pos_neg_and_rating(pred):
    def cond(i):
        return i <= pred < i + 0.1
    rating = 0
    while not cond(rating) and rating <= 0.9:
        rating += 0.1
    rating *= 10
    is_pos = pred > 0.5
    return rating, is_pos


def predict_sentiment(sentence, model = model, tokenizer = tokenizer):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).cpu()
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return return_pos_neg_and_rating(prediction.item())