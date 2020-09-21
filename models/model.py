from functools import total_ordering
from .DecoderRNN import DecoderRNN
from .EncoderRNN import EncoderRNN
from .Seq2Seq import Seq2seq
from .Attention import BahdanauAttention
from base import BaseModel
import torch


class ReversingModel(BaseModel):
    def __init__(
            self,
            encoder_emb_dim, decoder_emb_dim, hidden_dim,
            num_layers, max_len,
            src_vocab, tgt_vocab,
            sos_tok, eos_tok, pad_tok,
            device,
            bidirectional=False):
        super(ReversingModel, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        encoder_hidden_dim = hidden_dim
        decoder_hidden_dim = 2*encoder_hidden_dim if bidirectional else encoder_hidden_dim

        self.encoder = EncoderRNN(
            input_dim=len(src_vocab), emb_dim=encoder_emb_dim, hidden_dim=encoder_hidden_dim,
            num_layers=num_layers, max_len=max_len,
            ids=tuple(
                [src_vocab.stoi[tok] for tok in list([sos_tok, eos_tok, pad_tok])]),
            kind='LSTM', bidirectional=bidirectional,
            batch_first=True, variable_length=True,
            device=device
        )

        self.attention = None
        self.attention = BahdanauAttention(
            hidden_dim=hidden_dim,
            encoder_hidden_dim=2 * encoder_hidden_dim if bidirectional else encoder_hidden_dim,
            decoder_hidden_dim=decoder_hidden_dim
        )

        decoder_rnn_input_dim = decoder_emb_dim
        if self.attention is not None:
            decoder_rnn_input_dim = decoder_hidden_dim+decoder_emb_dim

        self.decoder = DecoderRNN(
            emb_dim=decoder_emb_dim,
            rnn_input_dim=decoder_rnn_input_dim,
            hidden_dim=decoder_hidden_dim,
            out_dim=len(tgt_vocab),
            num_layers=num_layers, max_len=max_len,
            ids=tuple(
                [tgt_vocab.stoi[tok] for tok in list([sos_tok, eos_tok, pad_tok])]),
            kind='LSTM',
            batch_first=True,
            attention=self.attention,
            device=device)

        self.seq2seq = Seq2seq(
            self.encoder,
            self.decoder)

    def state_dict(self):
        return {
            "model":super().state_dict(),
            "src_vocab":self.src_vocab,
            "tgt_vocab":self.tgt_vocab
        }
    def load_state_dict(self,state_dict):
        super().load_state_dict(state_dict['model'])
        self.src_vocab=state_dict['src_vocab']
        self.tgt_vocab=state_dict['tgt_vocab']
    def forward(
            self, input, input_lens=None,
            target=None,
            teacher_forcing_ratio=0):
        return self.seq2seq.forward(
            input=input, input_lens=input_lens,
            target=target,
            teacher_forcing_ratio=teacher_forcing_ratio
        )

    def predict(self, src_seq):
        src_id_seq = torch.LongTensor(
            [self.src_vocab.stoi[tok] for tok in src_seq]).view(1, -1)
        input_lens = torch.LongTensor([len(src_seq)])
        src_id_seq = src_id_seq.to(self.encoder.device)
        input_lens = input_lens.to(self.encoder.device)
        with torch.no_grad():
            _, _, sequence_dict = self.forward(
                input=src_id_seq,
                input_lens=input_lens,
                target=None,
                teacher_forcing_ratio=0)
        length = sequence_dict['length'][0]
        seq = sequence_dict['sequence']
        tgt_id_seq = [seq[i][0].cpu().item() for i in range(length)]
        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]

        attn_outputs = sequence_dict['attn']
        attn_outputs_relevant = attn_outputs[:length]

        return tgt_seq, torch.cat([attn_output for attn_output in attn_outputs_relevant]).cpu().numpy()
