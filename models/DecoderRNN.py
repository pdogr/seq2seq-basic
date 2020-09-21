import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from .BaseRNN import BaseRNN
import math


class DecoderRNN(BaseRNN):
    KEY_ATTN='attn'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(
            self,
            emb_dim,
            rnn_input_dim, hidden_dim, out_dim,
            max_len, ids,
            num_layers=4, dropout=0, rnn_dropout=0,
            kind='LSTM', batch_first=True,
            embedding=None, embedding_requires_grad=True,
            attention=None, device=None):
        super(DecoderRNN, self).__init__(
            rnn_input_dim, hidden_dim, max_len, ids,
            num_layers, rnn_dropout, False,
            kind, batch_first, device)

        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.attention = attention
        self.device = device
        self.rnn_input_dim = rnn_input_dim
        self.embedding = nn.Embedding(self.out_dim, self.emb_dim)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)

        self.embedding.weight.requires_grad = embedding_requires_grad
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.fc = nn.Linear(self.hidden_dim, self.out_dim)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            if isinstance(m, nn.Embedding):
                if self.embedding is None:
                    fan_in = m.in_features
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            if isinstance(m, nn.RNN):
                fan_in = m.in_features
                nn.init.normal_(m.weight, 0, math.sqrt(2./fan_in))

    def _initialize_params(self, target, encoder_hidden, teacher_forcing_ratio):
        batch_size = 1
        if target is None and encoder_hidden is None:
            batch_size = 1
        else:
            if target is not None:
                batch_size = target.size(0)
            else:
                if self.rnn is nn.GRU:
                    batch_size = encoder_hidden.size(1)
                elif self.rnn is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
        if target is None:
            assert teacher_forcing_ratio == 0, "Teacher forcing needs target"
            target = torch.LongTensor(
                [self.sos_id]*batch_size).view(batch_size, 1)
            target = target.to(self.device)
            max_len = self.max_len
        else:
            max_len = target.size(1)-1
        decoder_hidden = None
        if isinstance(encoder_hidden, tuple):
            decoder_hidden = tuple([h for h in encoder_hidden])
        else:
            decoder_hidden = encoder_hidden
        return decoder_hidden, target, batch_size, max_len

    def _get_hidden_last_layer(self, hidden):
        if type(hidden) == tuple:
            hidden = hidden[0]
        return hidden[-1].unsqueeze(1)

    def _forward(self, input, hidden, encoder_outputs, src_mask):
        batch_size = input.size(0)
        x = self.dropout(self.embedding(input))
        # x [ batch_size x 1 x decoder_emb_dim]
        attn=None
        if self.attention is not None:
            context, attn = self.attention(
                self._get_hidden_last_layer(hidden), encoder_outputs, src_mask)
            # context [batch_size x 1 x attn_hidden_dim]
            # attn [batch_size x seq_len]
            x = torch.cat([x, context], dim=2)

        # x [batch_size x 1 x attn_hidden_dim + emb_size]

        x, h = self.rnn(x, hidden)
        # x [ batch_size x 1 x decoder_hidden_size]

        x = self.fc(x.contiguous().view(-1, self.hidden_dim))
        x = torch.log_softmax(x, dim=1).view(batch_size, 1, -1)
        # x [ batch_size x 1 x out_dim]
        return x, h,attn

    def _decode_output(self, decoder_output, cur_len, lengths):
        symbols_output = decoder_output.topk(1, dim=1)[1]
        finished_batches = symbols_output.eq(self.eos_id)
        update_mask = None
        if finished_batches.dim() > 0:
            finished_batches = finished_batches.cpu().view(-1).numpy()
            update_mask = np.logical_and(
                (lengths > cur_len), (finished_batches))
        return symbols_output, update_mask

    def forward(self, target=None, encoder_hidden=None, encoder_outputs=None, src_mask=None,
                teacher_forcing_ratio=0.5):
        ret = {}
        decoder_hidden, target, batch_size, max_length = self._initialize_params(
            target, encoder_hidden, teacher_forcing_ratio)
        valid_lengths = np.array([max_length]*batch_size)
        output_sequence = []
        decoder_outputs = []
        attn_outputs=[]

        def _update(cur_decoder_output, cur_len,cur_attn):
            attn_outputs.append(cur_attn)
            decoder_outputs.append(cur_decoder_output)
            symbols_output, update_mask = self._decode_output(
                cur_decoder_output, cur_len, valid_lengths)
            output_sequence.append(symbols_output)
            if update_mask is not None:
                valid_lengths[update_mask] = len(output_sequence)
            return symbols_output
        if self.attention is not None:
            self.attention.project(encoder_outputs)
        if random.random() < teacher_forcing_ratio:
            decoder_input = target[:, 0].unsqueeze(1)
            for clen in range(target.size(1)-1):
                decoder_output, decoder_hidden,attn = self._forward(
                    decoder_input, decoder_hidden, encoder_outputs, src_mask)
                cur_decoder_output = decoder_output.squeeze(1)
                _update(cur_decoder_output, clen,attn)
                decoder_input = target[:, clen+1].unsqueeze(1)
        else:
            decoder_input = target[:, 0].unsqueeze(1)
            for clen in range(max_length):
                decoder_output, decoder_hidden,attn = self._forward(
                    decoder_input, decoder_hidden, encoder_outputs, src_mask)
                cur_decoder_output = decoder_output.squeeze(1)
                decoder_input = _update(cur_decoder_output, clen,attn)

        ret[DecoderRNN.KEY_SEQUENCE] = output_sequence
        ret[DecoderRNN.KEY_LENGTH] = valid_lengths.tolist()
        ret[DecoderRNN.KEY_ATTN]=attn_outputs

        return decoder_outputs, decoder_hidden, ret
