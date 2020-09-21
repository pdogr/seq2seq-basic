import argparse
import collections
import os
from pathlib import Path
from matplotlib.pyplot import xticks

import numpy as np
import torch

import models
from base import get_optimizer
from dataloader import Dataloader
from parse_config import ConfigParser
from trainer import Trainer
from utils.loss import AvgPerplexity
from utils.metric import Accuracy
import seaborn as sns

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(config, model_path):
    logger = config.get_logger('train')

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    tgt_preprocessing = None
    def src_preprocessing(x): return x[::-1]

    model_args = config['arch']['args']
    state_dict=torch.load(model_path)['state_dict']
    model_args.update({
        'src_vocab': state_dict['src_vocab'],
        'tgt_vocab': state_dict['tgt_vocab'],
        'sos_tok': '<sos>',
        'eos_tok': '<eos>',
        'pad_tok': '<pad>',
        'device': device
    })
    model = getattr(models, config['arch']['type'])(**model_args)
    model.load_state_dict(state_dict)
    del state_dict
    model.to(device)
    while True:
        seq_str = input(">")
        seq = list(seq_str.strip())
        print(seq)
        if src_preprocessing is not None:
            seq = src_preprocessing(seq)

        tgt_seq, attn_weights = model.predict(seq)
        print(tgt_seq)
        fig = sns.heatmap(attn_weights, xticklabels=seq, yticklabels=tgt_seq)
        fig.xaxis.set_label_position('top')
        fig.figure.show()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        '-c',
        '--config',
        default=None,
        type=str,
        help='config file path (default: None)')
    args.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='path to model.pth to test (default: None')
    args.add_argument(
        '-r',
        '--resume',
        default=None,
        type=str,
        help='path to latest checkpoint (default: None)')
    args.add_argument(
        '-d',
        '--device',
        default=None,
        type=str,
        help='indices of GPUs to enable (default: all)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target help')
    config,args = ConfigParser.from_args(args)
    main(config, args.model_path)
