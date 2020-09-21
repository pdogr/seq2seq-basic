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


def main(config, root_dir):
    logger = config.get_logger('train')

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    root_dir = Path(root_dir)
    train_path = Path(root_dir/'train/train_data.txt')
    valid_path = Path(root_dir/'valid/valid_data.txt')
    test_path = Path(root_dir/'test/test_data.txt')

    assert os.path.exists(
        train_path), "Train path does not exists {}".format(train_path)
    assert os.path.exists(
        valid_path), "Valid path does not exists {}".format(valid_path)
    batch_size = config['data_loader']['args']['batch_size']

    tgt_preprocessing = None
    def src_preprocessing(x): return x[::-1]
    
    train_loader = Dataloader(
        path=train_path, device=device,
        src_preprocessing=src_preprocessing, tgt_preprocessing=tgt_preprocessing,
        batch_size=batch_size)

    valid_loader = Dataloader(
        path=valid_path, device=device,
        src_preprocessing=src_preprocessing, tgt_preprocessing=tgt_preprocessing,
        batch_size=batch_size, train=False)

    test_loader=Dataloader(
        path=test_path,device=device,
        src_preprocessing=src_preprocessing,tgt_preprocessing=tgt_preprocessing,
        batch_size=batch_size,train=False)

    model_args = config['arch']['args']
    model_args.update({
        'src_vocab': train_loader.src_vocab,
        'tgt_vocab': train_loader.tgt_vocab,
        'sos_tok': '<sos>',
        'eos_tok': '<eos>',
        'pad_tok': '<pad>',
        'device': device
    })
    model = getattr(models, config['arch']['type'])(**model_args)

    weight = torch.ones(len(train_loader.tgt_vocab))
    pad_tok = Dataloader.PAD_TOK
    criterion = AvgPerplexity(
        ignore_idx=train_loader.tgt_vocab.stoi[pad_tok],
        weight=weight)
    criterion.to(device)

    optimizer = get_optimizer(
        optimizer_params=filter(
            lambda p: p.requires_grad, model.parameters()),
        args_dict=config['optimizer'])

    metrics_ftns = [Accuracy(
        train_loader.tgt_vocab.stoi[pad_tok])]

    trainer = Trainer(
        model=model,
        criterion=criterion,
        metric_ftns=metrics_ftns,
        optimizer=optimizer,
        config=config,
        data_loader=train_loader,
        valid_data_loader=valid_loader,
        log_step=1
    )
    trainer.train()
    trainer.test(test_loader)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        '-c',
        '--config',
        default=None,
        type=str,
        help='config file path (default: None)')
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
    options = [
        CustomArgs(
            ['--lr', '--learning-rate'],
            type=float,
            target='optimizer;optimizer;args;lr',
            help='Change optimzer learning_rate'),
        CustomArgs(
            ['--bs', '--batch-size'],
            type=int,
            target='data_loader;args;batch_size',
            help='Change batch_size of dataloader'),

        CustomArgs(
            ['--d', '--root-dir'],
            type=str,
            target='root_dir',
            help='location of data')
    ]
    config,args = ConfigParser.from_args(args, options)
    main(config, config['root_dir'])
