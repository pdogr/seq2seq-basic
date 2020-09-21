import torch
import torch.nn as nn


class BaseRNN(nn.Module):
    def __init__(
            self,
            input_dim, hidden_dim, max_len, ids,
            num_layers, rnn_dropout, bidirectional,
            kind, batch_first,device):
        super(BaseRNN, self).__init__()
        self.max_len = max_len
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.kind = kind
        self.sos_id, self.eos_id, self.pad_id = (ids)
        self.device=device
        if device is None:
            self.device=torch.device('cpu')
        try:
            self.rnn = getattr(torch.nn, kind)(
                input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                batch_first=batch_first, dropout=rnn_dropout, bidirectional=bidirectional)
        except Exception as e:
            raise e
