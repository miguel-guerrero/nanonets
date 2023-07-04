#!/usr/bin/env python3

import nanonets as nn
import numpy as np


def test1():
    x = nn.Var(2)
    y = nn.Var(3)
    z = nn.Var(7)
    c = nn.Const(2)
    f = x * y + z * c
    f.zero_grad()
    f.backward()
    df_dx = x.get_pass_grad()
    df_dy = y.get_pass_grad()
    df_dz = z.get_pass_grad()
    df_dc = c.get_pass_grad()
    print("f_val", f.data)
    print("df_dx", df_dx)
    print("df_dy", df_dy)
    print("df_dz", df_dz)
    print("df_dc", df_dc)
    assert (f.data == 20).all()
    assert (df_dx == 3).all()
    assert (df_dy == 2).all()
    assert (df_dz == 2).all()
    assert (df_dc == 0).all()


def test2():

    # to train
    w = nn.Var(np.array([[1, 1, 0]]).T)

    # inputs
    x = nn.Const(
        [
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )
    exp = nn.Const([[0], [0], [0], [1]])

    lr = 0.1
    for it in range(100):
        print("== iter", it)
        # eval error
        logit = x @ w
        pred = logit.relu()
        errs = (pred - exp).abs()
        loss = nn.Const([[0.25, 0.25, 0.25, 0.25]]) @ errs  # reduce
        print("x:", x, "pred:", pred.data, "exp:", exp)

        loss.zero_grad()
        loss.backward()
        print("loss", loss.data, "deriv", loss.grad)
        print("w", w.data, "deriv", w.get_pass_grad())

        # update weights
        w.data = w.data - lr * w.grad
        print("updated w", w)

    print("Final weights")
    print(w)


if __name__ == "__main__":
    # test1()
    test2()
