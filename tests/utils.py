import numpy as np
from collections import defaultdict


absErr = 1e-5
tpuErr = 0.05


def closeOld(x, y):
    if y != 0 and x != 0:
        return np.abs((x - y) / y) < tpuErr
    return np.abs(x - y) < absErr


def close(x, y):
    if np.abs(x - y) < absErr:
        return True
    if y != 0 and x != 0:
        return np.abs((x - y) / y) < tpuErr
    return False


def checkNumericDerivs(w, network, x0, y0, h, verbose):
    wr = w.data.reshape(-1)
    gr = w.grad.reshape(-1)
    for i in range(len(wr)):
        gi = gr[i]
        org = wr[i]
        wr[i] += h  # modified as reference
        yh = network(x0)
        numDeriv = (yh.data - y0.data) / h
        wr[i] = org
        if verbose or not close(numDeriv, gi):
            print("i", i)
            print("w.grad numerical:", numDeriv)
            print("w.grad auto-diff:", gi)
            print("diff:", np.abs(numDeriv - gi))
            print("tpu:", np.abs(numDeriv - gi / gi))


def computeNumericDerivs2(w, network, x0, target, loss0, h, verbose):
    wr = w.data.reshape(-1)
    gr = w.grad.reshape(-1)
    for i in range(len(wr)):
        gi = gr[i]
        org = wr[i]
        wr[i] += h  # modified as reference
        lossh, _ = network(x0, target)  # returns loss, accuracy
        numDeriv = (lossh.data - loss0.data) / h
        wr[i] = org
        if verbose or not close(numDeriv, gi):
            print("i", i)
            print("w.grad numerical:", numDeriv)
            print("w.grad auto-diff:", gi)
            print("diff:", np.abs(numDeriv - gi))


def readMnistImages(fileName, cnt=-1, batchSize=64, verbose=True):
    imgSide = 28
    allBatchesLst = []
    k = 0
    counter = defaultdict(int)
    with open(fileName) as f:
        contents = f.readlines()
        for line in contents:
            if k % batchSize == 0:
                batchLstX = []
                batchLstY = []
            fields = line.split(",")
            class_, imgLst = fields[0], fields[1:]
            counter[class_] += 1
            imgArr = np.array(imgLst, dtype=float).reshape(1, imgSide, imgSide)
            batchLstX.append(imgArr)
            batchLstY.append(float(class_))
            if verbose and (k % 500) == 0:
                print("class:", class_, " - ", k, "/", len(contents))
            k += 1
            if k % batchSize == 0 or cnt == k:
                allBatchesLst.append((np.stack(batchLstX), np.vstack(batchLstY)))
                if cnt == k:
                    break
        print(counter)
    return allBatchesLst
