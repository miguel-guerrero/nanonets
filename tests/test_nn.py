#!/usr/bin/env python3

import numpy as np
import nanonets as nn


def test1():
    nlp = nn.NLP(2, [1])

    nlp.layers[0].w = nn.Var(np.array([[1.0, 1.0]]).T, name="L0_w")
    nlp.layers[0].b = nn.Var(np.array([[0.0]]), name="L0_b")

    print("Initial weights")
    print(nlp.parameters())

    # inputs
    x = nn.Const(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    y = nn.Const([[0], [0], [0], [1]])

    niters = 100
    lr = 0.1
    for it in range(niters):
        print("== iter", it, "lr:", lr)
        pred = nlp(x)
        errs = (pred - y).abs()
        loss = nn.Const([[0.25, 0.25, 0.25, 0.25]]) @ errs  # reduce

        print("x:\n", x)
        print("y:\n", y)
        print("pred:\n", pred.data)

        nlp.zero_grad()
        loss.backward()

        print("loss:", loss.data)

        if it == 0 or it == niters - 1:
            nn.dot_show.draw_dot(loss, filename=f"test_nn_it{it}.vg")

        if False:
            for w in nlp.parameters():
                print("w:\n", w.data)
                print("w deriv:\n", w.grad)

        # update weights
        for w in nlp.parameters():
            w.data -= lr * w.grad

        if False:
            for w in nlp.parameters():
                print("updated w", w.data)

    print("Final weights")
    print(nlp.parameters())


if __name__ == "__main__":
    test1()
