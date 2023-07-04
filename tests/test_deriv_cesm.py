#!/usr/bin/env python3

import numpy as np
import nanonets as nn


imgSide = 6
zin = 2
zout = 3
ks = 3
eps = 1e-5
h = 1e-9


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # input zin x imgSize x imgSize
        layer0 = nn.Conv2D((zout, zin, ks, ks), prefix="L0")
        flatten = nn.Flatten(prefix="flatten")
        out = nn.FC(
            zout * (imgSide - ks + 1) ** 2, 3, prefix="FC", activation="passthru"
        )
        self.layers = [
            layer0,
            flatten,
            out,
        ]


def test1(verbose=True):
    def debug(*args):
        if verbose:
            print(*args, flush=True)

    def func(x):
        exp = nn.Const([[0], [1], [0]])
        if False:
            sm = x.softmax()
            cesm = sm.crossentropy(exp)
            return cesm.sum()
        else:
            cel = x.celoss(exp)
            return cel

    x0 = nn.Var([[0.5], [0.3], [-0.2]])
    print("x0", x0.data)
    y0 = func(x0)
    print("y0", y0.data.item())

    y0.zero_grad()
    y0.backward()

    # numerically compute derivative w.r.t L0_w and compare to backwards pass
    j = 0
    for i in range(3):
        org = x0.data[i, j]
        x0.data[i, j] += h
        yh = func(x0)
        numDeriv = (yh.data.item() - y0.data.item()) / h
        x0.data[i, j] = org
        print("j:", j)
        diff = np.abs(numDeriv - x0.grad[i, j])
        if verbose or np.any(diff > eps):
            print("x0.grad numerical:", numDeriv)
            print("x0.grad auto-diff:", x0.grad[i, j])
            print("diff:", diff)


if __name__ == "__main__":
    seed = 76
    np.random.seed(seed)
    test1(verbose=True)
