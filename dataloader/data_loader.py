import torch
import torchtext
from base import BaseDataLoader


class Dataset(object):

    SOS_TOK = '<sos>'
    EOS_TOK = '<eos>'
    PAD_TOK = '<pad>'
    UNK_TOK = '<unk>'

    def __init__(
            self,
            path,
            src_preprocessing=None, tgt_preprocessing=None):
        self.src = torchtext.data.Field(
            batch_first=True,
            sequential=True, use_vocab=True,
            include_lengths=True,
            preprocessing=
            lambda x: src_preprocessing(x) if src_preprocessing is not None else x
        )

        self.tgt = torchtext.data.Field(
            batch_first=True,
            sequential=True, use_vocab=True,
            init_token=self.SOS_TOK, eos_token=self.EOS_TOK,
            pad_token=self.PAD_TOK, unk_token=self.UNK_TOK,
            preprocessing=
            lambda x: tgt_preprocessing(x) if tgt_preprocessing is not None else x
        )
        self.path = path
        self.data = None
        self._get_data()

    def _get_data(self):
        self.data = torchtext.data.TabularDataset(
            path=self.path, format='tsv',
            fields=[
                ('source', self.src),
                ('target', self.tgt)
            ]
        )
        self.src.build_vocab(self.data)
        self.tgt.build_vocab(self.data)

        self.src_vocab = self.src.vocab
        self.tgt_vocab = self.tgt.vocab


class Dataloader(torchtext.data.BucketIterator):
    SOS_TOK = '<sos>'
    EOS_TOK = '<eos>'
    PAD_TOK = '<pad>'
    UNK_TOK = '<unk>'
    def __init__(
            self, path, device, batch_size,
            src_preprocessing=None, tgt_preprocessing=None,
            train=True):
        self.path = path
        self.device = device
        dataset = Dataset(
            self.path,
            src_preprocessing=src_preprocessing,tgt_preprocessing=tgt_preprocessing,
        )
        self.src=dataset.src
        self.tgt=dataset.tgt
        self._src_vocab = dataset.src_vocab
        self._tgt_vocab = dataset.tgt_vocab

        super().__init__(
            dataset=dataset.data,
            batch_size=batch_size,
            sort=False, sort_key=lambda x: len(x.source), sort_within_batch=True,
            device=self.device, train=False, shuffle=True, repeat=False
        )

    @property
    def src_vocab(self):
        return self._src_vocab

    @property
    def tgt_vocab(self):
        return self._tgt_vocab
