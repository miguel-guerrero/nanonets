
## Introduction

Nanonets is a simple Artificial Neuronal Network framework built from scratch (strictly over numpy) for learning purposes.

It implements autodifferentation and the common activation functions, network operators required for simple MLP's
and CNNs. 


## Dependencies

    $ brew install graphviz
    $ python3 -mpip install graphviz
    $ python3 -mpip install numpy

## Quick start

Download and train a simple CNN to for MNIST dataset

    $ ./download_mnist.sh
    $ ./tests/test_mnist.py
