import torch


class nll_loss(torch.nn.modules.loss._Loss):
    def __init__(
            self,
            ignore_idx=-100, weight=None, mask=None, reduction='mean'):
        super().__init__()
        if mask is not None:
            assert weight is not None, "Weight needed"
            weight[mask] = 0
        self._loss = torch.nn.NLLLoss(
            weight=weight,
            ignore_index=ignore_idx,
            reduction=reduction)

    def __call__(self, logit, target):
        return self.loss(logit, target)

    @property
    def loss(self):
        return self._loss


class Perplexity(nll_loss):
    def __init__(
            self,
            ignore_idx=-100, weight=None, mask=None, reduction='mean'):
        super().__init__(
            ignore_idx=ignore_idx,
            weight=weight,
            mask=mask,
            reduction=reduction)

    def __call__(self, logit, target):
        nll = self.loss.__call__(logit, target)
        return torch.exp(nll)


class AvgPerplexity(Perplexity):
    """
    Avg perplexity of a batch of input and target of length seq_len

    """
    def __init__(
            self,
            ignore_idx=-100, weight=None, mask=None, reduction='mean'):
        super().__init__(
            ignore_idx=ignore_idx,
            weight=weight,
            mask=mask,
            reduction=reduction)

    def __call__(self, outputs, targets):
        """
        targets: (batch_size x seq_len)
        outputs: (batch_size x seq_len)
        """
        batch_size = targets.size(0)
        loss = 0
        for step, output in enumerate(outputs):
            nll = self.loss.__call__(
                output.contiguous().view(batch_size, -1),
                targets[:, step+1])
            loss += torch.exp(nll)
        loss /= targets.size(1)
        return loss

    @property
    def __name__(self):
        return self.__class__.__name__
