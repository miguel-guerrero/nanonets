#!/usr/bin/env python3

import numpy as np
import nanonets as nn
import utils


kz, kc, ks = 2, 2, 3
kernelShape = (kz, kc, ks, ks)
h = 1e-10
N = 4  # batches
imgSide=10


class Perceptron(nn.Module):
    def __init__(self, act="leakyrelu"):
        super().__init__()

        layer0 = nn.Conv2D(kernelShape, activation=act)
        layer1 = nn.MaxPool2D((kc, 2, 2), activation=act)
        layer2 = nn.Flatten()
        layer3 = nn.FC(kz*(imgSide-ks+1)**2//4, 1, activation=act)

        self.layers = [
            layer0,
            layer1,
            layer2,
            layer3,
        ]


def test1(act, verbose=True):
    def debug(*args):
        if verbose:
            print(*args, flush=True)

    perc = Perceptron(act=act)

    def network(x):
        return perc(x).sum()

    inputSize = kc * imgSide ** 2
    x0val = np.arange(N * inputSize).reshape(N, kc, imgSide, imgSide) / inputSize
    x0val = x0val - np.mean(x0val)  # have pos an neg vals
    # print("x0val", x0val)

    x0 = nn.Var(x0val, name="input")
    y0 = network(x0)

    y0.zero_grad()
    y0.backward()

    # nn.dot_show.draw_dot(y0, filename=f"test_numerical_deriv_fc_post_{act}.vg")

    # numerically compute derivative w.r.t w
    for w in perc.parameters():
        print("checking deriv for w")
        utils.checkNumericDerivs(w, network, x0, y0, h, verbose=False)

    # numerically compute derivative w.r.t x
    print("df/dx")
    utils.checkNumericDerivs(x0, network, x0, y0, h, verbose=False)

if __name__ == "__main__":
    for act in ["relu", "tanh", "sigmoid", "leakyrelu"]:
        print("== Testing act", act)
        seed = 76
        np.random.seed(seed)
        test1(act, verbose=False)
