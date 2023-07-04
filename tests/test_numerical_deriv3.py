#!/usr/bin/env python3

import numpy as np
import nanonets as nn
import utils


imgSide = 6
zin = 2
zout = 3
ks = 3
h = 1e-8  # 1e-9


class Perceptron(nn.Module):
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


def test1(lossOpt, verbose=True):
    def debug(*args):
        if verbose:
            print(*args, flush=True)

    print("== test lossOpt", lossOpt)
    perc = Perceptron()

    batches = 1

    def network(x):
        class_ = 1
        exp = nn.Const([[0, 0, 0]])
        exp.data[0, class_] = 1.0

        p = perc(x)

        if lossOpt == 1:  # ok
            sm = p.softmax()
            cesm = sm.crossentropy(exp)
            loss = cesm.sum()
        elif lossOpt == 2:  # ok
            loss = ((p - exp) ** 2).mean()
        elif lossOpt == 3:  # ok
            loss = (p - exp).abs().mean()
        elif lossOpt == 4:  # ok
            loss = p.celoss(exp)
        elif lossOpt == 5:  # ok
            lsm = p.logsoftmax()
            loss = lsm.nll(class_)
        else:
            raise RuntimeError(f"invalid lossOpt={lossOpt}")
        return loss

    x0 = nn.Var(
        np.arange(zin * imgSide * imgSide, dtype=float).reshape(
            [batches, zin, imgSide, imgSide]
        )
    )

    y0 = network(x0)
    # print("x0", x0.data)
    # print("y0", y0.data.item())

    y0.zero_grad()
    y0.backward()

    # nn.dot_show.draw_dot(y0, filename=f"test_numerical_deriv3.vg")

    # numerically compute derivative w.r.t w
    for w in perc.parameters():
        print("checking deriv for", w)
        utils.checkNumericDerivs(w, network, x0, y0, h, verbose=False)

    # numerically compute derivative w.r.t x and compare to backwards pass
    print("df/dx")
    utils.checkNumericDerivs(x0, network, x0, y0, h, verbose=False)


if __name__ == "__main__":
    seed = 76
    for lossOpt in range(1, 6):
        np.random.seed(seed)
        test1(lossOpt, verbose=False)
