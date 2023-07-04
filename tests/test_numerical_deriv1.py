#!/usr/bin/env python3

import numpy as np
import nanonets as nn
import utils


imgSide = 20
zin = 1
zout = 4
zout2 = 1
ks = 3
h = 1e-4


class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        act = "leakyrelu"
        # input zin x imgSize x imgSize
        dimKernel0 = (zout, zin, ks, ks)
        layer0 = nn.Conv2D(dimKernel0, prefix="L0", activation=act)
        layer0.w.data = np.arange(np.prod(dimKernel0), dtype=float).reshape(dimKernel0)
        layer0.b.data = np.arange(zout, dtype=float) + 1

        dimKernel1 = (zout2, zout, ks, ks)
        layer1 = nn.Conv2D(dimKernel1, prefix="L1", activation=act)
        layer1.w.data = 2 * np.arange(np.prod(dimKernel1), dtype=float).reshape(
            dimKernel1
        )
        layer1.b.data = np.arange(zout2, dtype=float) - 10

        layer2 = nn.MaxPool2D((zout2, 2, 2), prefix="L2")  # zout2 x 8 x 8

        layer3 = nn.Flatten(prefix="L3")

        layer4 = nn.FC(zout2 * 8 * 8, 10, activation=act, prefix="L4_FC")  # 128
        # layer4 = nn.FC(zout2 * 16 * 16, 10, activation=act, prefix="L4_FC")     # 128

        self.layers = [
            layer0,
            layer1,
            layer2,
            layer3,
            layer4,
        ]


def test1(verbose=True):
    def debug(*args):
        if verbose:
            print(*args, flush=True)

    perc = Perceptron()

    batches = 1

    def network(x):
        return perc(x).sum()

    x0 = nn.Var(
        -2
        * np.arange(zin * imgSide * imgSide, dtype=float).reshape(
            [batches, zin, imgSide, imgSide]
        )
    )
    y0 = network(x0)

    y0.zero_grad()
    y0.backward()

    if verbose:
        nn.dot_show.draw_dot(y0, filename=f"test_nn4.vg")

    # numerically compute derivative w.r.t w
    for w in perc.parameters():
        print("checking deriv for", w)
        utils.checkNumericDerivs(w, network, x0, y0, h, verbose=False)

    # numerically compute derivative w.r.t x
    print("df/dx")
    utils.checkNumericDerivs(x0, network, x0, y0, h, verbose=False)


if __name__ == "__main__":
    seed = 76
    np.random.seed(seed)
    test1(verbose=False)
