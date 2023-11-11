
## Introduction

Nanonets is a simple Artificial Neuronal Network framework built from scratch (well, strictly over `numpy`) in pure python for personal learning purposes.
It is inspired by Andrej Karpathys lectures, I.e.

[The spelled-out intro to neural networks and backpropagation: building micrograd
](https://www.youtube.com/watch?v=VMj-3S1tku0)

But extended to perform auto-differentiation with matrix operators (vs. scalar nodes), which makes debugging, visualization and execution speed much more manageable.

It implements eager execution (as popularized by Pytorch), common activation/loss functions and network operators required to train simple MLP's and CNNs (conv2d, dropout, fully connected layers for example) and as mentioned above autodifferentation. It has been successfuly tested to train MNIST dataset for example.

## Dependencies

Assuming a MacOS environment:

    $ brew install graphviz
    $ python3 -mpip install graphviz
    $ python3 -mpip install numpy

## Quick start

Download and train a simple CNN to for MNIST dataset

    $ ./download_mnist.sh
    $ ./tests/test_mnist.py

The default run goes over a full epoch, but convergence can be appreciated much earlier.
