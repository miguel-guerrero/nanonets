#!/usr/bin/env python3

import numpy as np
import nanonets as nn


def test1(verbose=True, niters=300):
    def debug(*args):
        if verbose:
            print(*args, flush=True)

    # nlp = nn.NLP(2, [5, 3], init="default", distribution="uniform", activation="relu")
    # nlp = nn.NLP(2, [5, 3], init="bias2", distribution="normal", activation="relu")
    nlp = nn.NLP(2, [5, 3], init="bias2", distribution="uniform", activation="relu")

    debug("Initial weights")
    debug(nlp.parameters())

    # inputs
    x = nn.Const(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    y = nn.Const(
        [
            [0, 0, 0],
            [1, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ]
    )

    lri = 0.1 / 4
    lrf = lri / niters
    for it in range(niters):
        lr = lri + (lrf - lri) * it / (niters - 1)
        debug("== iter", it, "lr:", lr)
        pred_raw = nlp(x)
        # pred = pred_raw.softmax()
        pred = pred_raw
        errs4x3 = (pred - y).abs()
        # errs4x3 = (pred - y) ** 2
        errs1x3 = nn.Const([[1, 1, 1, 1]]) @ errs4x3  # reduce
        loss = errs1x3 @ nn.Const([[1], [1], [1]])  # reduce

        debug("x:\n", x)
        debug("y:\n", y)
        debug("pred:\n", pred.data)

        # nlp.zero_grad()
        loss.zero_grad()
        loss.backward()

        debug("loss:", loss.data)

        if verbose and (it == 0 or it == niters - 1):
            nn.dot_show.draw_dot(loss, filename=f"test_xor_and_or_it{it}.vg")

        if False:
            for w in nlp.parameters():
                debug("w:\n", w.data)
                debug("w deriv:\n", w.grad)

        # update weights
        for w in nlp.parameters():
            w.data -= lr * w.grad

        if False:
            for w in nlp.parameters():
                debug("updated w:\n", w.data)

    debug("Final weights")
    debug(nlp.parameters())

    return loss.data[0, 0]


if __name__ == "__main__":
    if False:
        seed = 76
        np.random.seed(seed)
        loss = test1(verbose=True, niters=500)
    else:
        cntLessOne = 0
        for seed in range(100):
            np.random.seed(seed)
            loss = test1(verbose=False)
            if loss < 1:
                cntLessOne += 1
            print(loss, seed)
        print(cntLessOne)
