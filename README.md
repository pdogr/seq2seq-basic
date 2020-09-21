# Pytorch seq2seq

This repo contains the implementation of sequence-to-sequence (seq2seq) models using [PyTorch](https://github.com/pytorch/pytorch) 1.6 and [TorchText](https://github.com/pytorch/text) 0.7 using Python 3.8.

**If you find any mistakes or disagree with any of the explanations, please do not hesitate to [submit an issue](https://github.com/plaxi0s/seq2seq-basic/issues/new). I welcome any feedback, positive or negative!**

##### The main ideas used are from the following papers.
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) - RNN Encoder-Decoder based architecture using a multi-layer LSTMs 
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine - ranslation](https://arxiv.org/abs/1406.1078) - RNN Encoder-Decoder based seq2seq model using GRUs 
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - Intorduces the concept of additivite attention which allievates the information compression problem by allowing the decoder to "look back" at the input sentence by creating context vectors that are weighted sums of the encoder hidden states

## Getting Started

#### Install torch and torchtext
To install PyTorch, see installation instructions on the [PyTorch website](pytorch.org).

To install TorchText:

``` bash
pip install torch torchtext
```

#### Prepare toy dataset
Run the following command to generate dataset
``` bash
# The generated data is stored in {dir}/reverse/{vocab-size}{max-len}/
python3 data_generator.py --dir data/ --max-len 20 --vocab-size medium
# This generates dataset to data/reverse/medium30/
```
#### Train 
The parameters for training can be configured via config.json. By default trainined models, logs, tensorboard logs are stored in saved/ directory.
```bash
# For training new model
python3 train.py -c config.json --root-dir /data/reverse/medium30/ 
# For resuming from checkpoint 
python3 train.py -c config.json --root-dir /data/reverse/medium30 -r {path_to_model_checkpoint.pth}
```
#### Testing the model
After training you can check the model for predictions via entering a sequence
``` bash
python3 test.py -c config.json --model-path {path_to_model_checkpoint.pth}
```

## References

Here are some things I looked at while making these tutorials. Some of it may be out of date.
- https://github.com/bentrevett/pytorch-seq2seq
- https://github.com/spro/practical-pytorch
- https://blog.floydhub.com/attention-mechanism/
- https://github.com/IBM/pytorch-seq2seq
