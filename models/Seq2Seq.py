import torch.nn as nn

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, input_lens=None,
                target=None, teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden,src_mask = self.encoder.forward(
            input, input_lens)
        return self.decoder.forward(target=target,
                                    encoder_hidden=encoder_hidden,
                                    encoder_outputs=encoder_outputs,
                                    src_mask=src_mask,
                                    teacher_forcing_ratio=teacher_forcing_ratio)
