import torch
import torch.nn as nn
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from base import Optimizer
from torchvision.utils import make_grid


class Trainer(BaseTrainer):
    def __init__(
            self,
            model, criterion, metric_ftns, optimizer: Optimizer,
            config, data_loader, valid_data_loader, len_epoch=None,
            log_step=2):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = log_step
        if len_epoch is None:
            self.len_epoch = len(self.data_loader)
        else:
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.train_metrics = MetricTracker(
            'train_loss',
            *['train_'+m.__name__ for m in self.metric_ftns],
            writer=self.writer)
        self.valid_metrics = MetricTracker(
            'val_loss',
            *['val_'+m.__name__ for m in self.metric_ftns],
            writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.logger.info(epoch)
        for batch_idx, batch in enumerate(self.data_loader):
            input_variables, input_lengths = getattr(batch, 'source')
            target = getattr(batch, 'target')
            self.optimizer.zero_grad()
            output, _, sequence_info = self.model.forward(
                input=input_variables, input_lens=input_lengths,
                target=target,
                teacher_forcing_ratio=0.5)
            loss = self.criterion.__call__(
                output,
                target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            # set train metrics

            self.train_metrics.update('train_loss', loss.item())
            for metric in self.metric_ftns:
                self.train_metrics.update(
                    'train_' + metric.__name__,
                    metric(output, target, sequence_info))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.8f}'.format(
                    epoch, self._progress(batch_idx), loss.item()))

        history = self.train_metrics.result()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            history.update(**{k: v for k, v in val_log.items()})

        self.optimizer.step_lr(history['train_loss'], epoch)
        return history

    def _valid_epoch(self, epoch):
        """
                Validate after training an epoch

                :param epoch: Integer, current training epoch.
                :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader):

                input_variables, input_lengths = getattr(batch, 'source')
                target = getattr(batch, 'target')
                output, _, sequence_info = self.model.forward(
                    input=input_variables, input_lens=input_lengths,
                    target=target)
                loss = self.criterion.__call__(
                    output,
                    target)

                # set writer step
                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx,
                    'valid')

                # set val metrics
                self.valid_metrics.update('val_loss', loss.item())
                for metric in self.metric_ftns:
                    self.valid_metrics.update(
                        'val_'+metric.__name__,
                        metric(output, target, sequence_info))

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return self.valid_metrics.result()


    def test(self,test_loader):
        self.model.eval()
        test_metrics = MetricTracker(
            'test_loss',
            *['test_'+m.__name__ for m in self.metric_ftns],
            writer=self.writer)

        with torch.no_grad():
            for batch_idx,batch in enumerate(test_loader):
                input_variables, input_lengths = getattr(batch, 'source')
                target = getattr(batch, 'target')
                output, _, sequence_info = self.model.forward(
                    input=input_variables, input_lens=input_lengths,
                    target=target)
                loss = self.criterion.__call__(
                    output,
                    target)

                # set writer step
                self.writer.set_step(
                    (self.epochs - 1) * len(self.valid_data_loader) + batch_idx,
                    'test')

                # set val metrics
                test_metrics.update('test_loss', loss.item())
                for metric in self.metric_ftns:
                    test_metrics.update(
                        'test_'+metric.__name__,
                        metric(output, target, sequence_info))

        for name, p in self.model.named_parameters():
            self.writer.add_histogram('test_' + name, p, bins='auto')
        for key, value in test_metrics.result().items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))
        
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    