class Accuracy(object):
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, output, target, *kwargs):
        total = 0
        correct = 0
        sequence=kwargs[0].pop('sequence')
        for i, outi in enumerate(output):
            tgt = target[:, i+1]
            unpadded = tgt.ne(self.pad_id)
            s=sequence[i]
            correct += s.view(-1).eq(tgt).masked_select(unpadded).sum().item()
            total += unpadded.sum().item()

        return correct/total

    @property
    def __name__(self):
        return str(self.__class__.__name__)