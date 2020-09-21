import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .BaseRNN import BaseRNN

class EncoderRNN(BaseRNN):    
    def __init__(
            self,
            input_dim, emb_dim, hidden_dim, max_len, ids,
            num_layers=4, dropout=0.5, rnn_dropout=0.5,
            kind='LSTM', bidirectional=False, batch_first=True,
            variable_length=False,
            embedding=None, embedding_requires_grad=True,device=None):

        super(EncoderRNN, self).__init__(
            emb_dim, hidden_dim, max_len,ids,
            num_layers, rnn_dropout, bidirectional,
            kind, batch_first,device)

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.variable_length = variable_length

        self.embedding = nn.Embedding(self.input_dim, self.emb_dim)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = embedding_requires_grad
        self.dropout = nn.Dropout(dropout, inplace=True)

        self.init_params()

    def _fix_bidirectional(self, hidden):

        if not self.bidirectional:
            return hidden
        if type(hidden)==tuple:
            return tuple([torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) for h in hidden])
        else:
            return torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)

    def init_params(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            if isinstance(m,nn.Embedding):
                if self.embedding is None:
                    fan_in=m.in_features
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            if isinstance(m,nn.RNN):
                fan_in=m.in_features
                nn.init.normal_(m.weight,0,math.sqrt(2./fan_in))

    def forward(self, input, input_lens=None):
        x = self.dropout(self.embedding(input))
        src_mask=input.ne(self.pad_id)
        if self.variable_length:
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lens, batch_first=True)
        out, hidden = self.rnn(x)
        if self.variable_length:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        return out, self._fix_bidirectional(hidden),src_mask
