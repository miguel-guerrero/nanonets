#!/usr/bin/env python3

import numpy as np
import nanonets as nn
from collections import defaultdict
import utils


h = 1e-8


def fprint(*args):
    print(*args, flush=True)


class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        # input shape is 1x28x28
        N0 = 32 # 8
        N1 = 64 # 16
        a = "sigmoid"  # converges with mean sqr error, nll & celoss
        # a = "relu"
        self.layers = [
            nn.Conv2D((N0, 1, 3, 3), activation=a, prefix="L0"),  # 32x26x26
            nn.Conv2D((N1, N0, 3, 3), activation=a, prefix="L1"),  # 64x24x24
            nn.MaxPool2D((N1, 2, 2), prefix="L2"),  # 64x12x12
            nn.DropOut(0.25, prefix="L3"),
            nn.Flatten(prefix="L4"),  # flatten
            nn.FC(N1 * 12 * 12, 128, activation=a, prefix="L5_FC"),  # 128
            nn.DropOut(0.5, prefix="L6"),
            nn.FC(128, 10, activation=a, prefix="L7_FC"),  # 10
        ]


class Network:
    def __init__(self):
        self.perc = Perceptron()

    def __call__(self, x, y, verbose=True):
        yCoded = (np.arange(10) == y) * 1.0
        ycoded = nn.Const(yCoded)
        x0 = nn.Const(x)
        pred = self.perc(x0)

        opt = 5
        if opt == 1:
            # mean square error
            err = ((pred - ycoded) ** 2).mean()
        elif opt == 2:
            # sum abs error with softmax output
            softmax = pred.softmax()
            err = (softmax - ycoded).abs().sum()
        elif opt == 3:
            # cross entropy
            softmax = pred.softmax()
            cesoftmax = ycoded.crossentropy(softmax)
            err = cesoftmax.sum()
        elif opt == 4:
            # combined cross entropy loss (includes softmax)
            err = pred.celoss(ycoded)  # TODO includes sum
        elif opt == 5:
            # negative log loss (NLL) of logsoftmax
            logsoftmax = pred.logsoftmax()
            err = logsoftmax.nll(y).sum()
        else:
            raise RuntimeError(f"unexpceted opt={opt}")

        accur = np.sum(np.argmax(pred.data, axis=1).reshape(-1, 1) == y)

        if verbose:
            fprint("ycoded:", yCoded)
            fprint("pred:  ", pred.data[0])
            fprint("err:", err.data)
        return err, accur

    def removeDropOut(self):
        # dropout layers are not needed if doing only inference
        newLayers = []
        for layer in self.perc.layers:
            if not isinstance(layer, nn.DropOut):
                newLayers.append(layer)
        self.perc.layers = newLayers

    def parameters(self):
        return self.perc.parameters()

    def load(self, fileName):
        fprint("Restoring model")
        self.perc = nn.load(fileName)
        fprint(f"Restore model {fileName} done")

    def save(self, fileName):
        fprint("Saving model")
        nn.save(fileName, self.perc)
        fprint(f"Save model {fileName} done")


def testAccuracy(testBatches, network):
    totalLoss = 0
    totalAcc = 0
    n = 0
    for currBatch in range(len(testBatches)):
        if currBatch % 100 == 0:
            fprint("checking test sample", currBatch)
        x, y = testBatches[currBatch]
        err, acc = network(x, y, verbose=False)
        totalLoss += err.data.item()
        totalAcc += acc
        n += x.shape[0]
    return totalLoss / n, totalAcc / n


def train(verbose=True, niters=-1, nepochs=-1):
    def debug(*args):
        if verbose:
            print(*args, flush=True)

    batchSize = 64
    dir = "tests"
    # dir = "."

    if niters == -1:  # run full test
        cnt = None
    else:
        cnt = (2*niters+1) * batchSize  # run shorter version

    trainSamples = utils.readMnistImages(f"{dir}/mnist/mnist_train.csv", cnt, batchSize)
    testSamples = utils.readMnistImages(f"{dir}/mnist/mnist_test.csv", cnt, batchSize)

    network = Network()

    lri = 0.0001
    lrf = lri / 1000  # niters
    totalBatches = len(trainSamples)
    epoch = 0
    it = 0
    # nits = niters if niters > 0 else 1000
    nits = niters
    currBatchNum = 0
    while it < niters or niters == -1:
        lr = lri + (lrf - lri) * it / (nits - 1)

        x, y = trainSamples[currBatchNum]
        loss, acc = network(x, y, verbose=False)

        if it % 10 == 0:
            debug(
                f"== iter: {it:4}",
                f"lr: {lr:.9f}",
                "epoch:", epoch,
                f"iter: {it:4}",
                f"loss: {loss.data.item():9.4f}",
                f"acc: {100 * acc / batchSize: 7.2f} %"
            )

        loss.zero_grad()
        loss.backward()

        if batchSize == 1:
            for w in network.parameters():
                debug("checking derivs for", w)
                utils.checkNumericDerivs(w, network, x, y, h, verbose)

        # update weights
        for w in network.parameters():
            w.data -= lr * w.grad

        if verbose and (it <= 1 or it == niters - 1):
            nn.dot_show.draw_dot(loss, filename=f"test_mnist_it{it}.vg")

        currBatchNum += 1
        if currBatchNum >= totalBatches:
            currBatchNum = 0
            epoch += 1
            unitaryLoss, testAcc = testAccuracy(testSamples, network)
            debug(f"---> tstAcc:  {100 * acc / batchSize: 7.2f} %")
            debug(f"---> tstLoss: {unitaryLoss * batchSize: 9.4f}")
            # save the model periodically
            network.save(f"mnist_perc_{it}.pkl")
            network.load(f"mnist_perc_{it}.pkl")
            if nepochs > 0 and epoch >= nepochs:
                break

        it += 1

    return loss.data[0]


def test(checkPointName, testSamples):
    network = Network()
    network.load(checkPointName)
    network.removeDropOut()
    _unitaryLoss, testAcc = testAccuracy(testSamples, network)
    fprint("---> tstAcc:", 100 * testAcc, "%")


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": lambda x: "{0:+0.3f}".format(x)})

    runTest = False
    runTrain = True
    runRegress = False

    if runTest:
        testSamples = utils.readMnistImages("mnist/mnist_test.csv")
        for it in range(1500, 4501, 1000):
            test(f"train_mnist_sigmoid_nnl/mnist_perc_{it}.pkl", testSamples)

    if runTrain:
        seed = 1  # 76
        np.random.seed(seed)
        # loss = train(verbose=True)
        # loss = train(verbose=True, nepochs=1)
        loss = train(verbose=True, niters=20)

    if runRegress:
        cntLessOne = 0
        for seed in range(100):
            np.random.seed(seed)
            loss = train(verbose=False)
            if loss < 1:
                cntLessOne += 1
            fprint(loss, seed)
        fprint(cntLessOne)
