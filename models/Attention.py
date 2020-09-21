import torch
import torch.nn as nn
import math


class BahdanauAttention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        encoder_hidden_dim, decoder_hidden_dim,
        ):
        super(BahdanauAttention, self).__init__()

        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim

        self.hidden_dim = hidden_dim
        self.decoder_project = nn.Linear(
            decoder_hidden_dim, hidden_dim, bias=False)
        self.encoder_project = nn.Linear(
            encoder_hidden_dim, hidden_dim, bias=False)
        self.energy = nn.Linear(hidden_dim, 1, bias=False)
        self.encoder_projected = None
        self.init_params()

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
        
    def project(self,encoder_outputs):
        self.encoder_projected=self.encoder_project(encoder_outputs)

    def forward(self, decoder_hidden_last, encoder_outputs, mask):
        # assert mask is not None, "mask required"

        # batch_first is enabled
        # decoder_hidden_last [ 1 x batch_size x decoder_hidden_dim ] maybe a tuple
        # encoder_outputs
        # [ batch_size x src_seq_len x encoder_hidden_dim * num_directions]
        # assert encoder_hidden_dim * num_directions is equal to decoder_hidden_dim

        decoder_projected = self.decoder_project(decoder_hidden_last)

        # encoder_projected batch_size x seq_len x hidden_dim
        energy_scores = self.energy(torch.tanh(
            decoder_projected+self.encoder_projected))
        # energy_scores  batch_size x seq_len x 1
        energy_scores = energy_scores.squeeze(2)
        # energy_scores  batch_size x seq_len
        energy_scores.data.masked_fill_(mask == 0, -float('inf'))

        attn_weights = torch.softmax(energy_scores,dim=-1)
        # attn_weights batch_size x seq_len

        context=torch.bmm(attn_weights.unsqueeze(1),encoder_outputs)

        # context batch_size x 1 x hidden_dim
        return context,attn_weights
