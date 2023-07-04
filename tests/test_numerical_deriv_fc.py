#!/usr/bin/env python3

import numpy as np
import nanonets as nn
import utils


ninputs = 50
nhidden = 10
h = 1e-10
N = 4  # batches


class Perceptron(nn.Module):
    def __init__(self, act="leakyrelu"):
        super().__init__()

        layer0 = nn.FC(ninputs, nhidden, activation=act, prefix="L0")
        layer1 = nn.FC(nhidden, 1, activation=act, prefix="L1")

        self.layers = [
            layer0,
            layer1,
        ]


def test1(act, verbose=True):
    def debug(*args):
        if verbose:
            print(*args, flush=True)

    perc = Perceptron(act=act)

    def network(x):
        return perc(x).sum()

    x0val = np.arange(N * ninputs).reshape(N, ninputs) / (N*ninputs)
    x0val = x0val - np.mean(x0val)  # have pos an neg vals
    # print("x0val", x0val)

    x0 = nn.Var(x0val, name="input")
    y0 = network(x0)

    y0.zero_grad()
    y0.backward()

    # nn.dot_show.draw_dot(y0, filename=f"test_numerical_deriv_fc_post_{act}.vg")

    # numerically compute derivative w.r.t w
    for w in perc.parameters():
        print("checking deriv for w", w.name)
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
