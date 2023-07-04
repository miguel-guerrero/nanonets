import numpy as np
from .node import Var, Const, prepend
import pickle


# --------------------------------------------------------------------------
# save / restore a network with its weights
# --------------------------------------------------------------------------
def save(filename, obj):
    with open(filename, "wb") as out_file:
        pickle.dump(obj, out_file)


def load(filename):
    with open(filename, "rb") as in_file:
        obj = pickle.load(in_file)
        return obj


# --------------------------------------------------------------------------
# utilities
# --------------------------------------------------------------------------
def fn_relu(x):
    return x.relu()


def fn_tanh(x):
    return x.tanh()


def fn_sigmoid(x):
    return x.sigmoid()


def fn_leakyrelu(x):
    return x.leakyrelu()


def fn_passthru(x):
    return x


def getActivationFunc(activation):
    if activation == "relu":
        return fn_relu  # avoiding lambdas as they don't pickle well
    elif activation == "tanh":
        return fn_tanh
    elif activation == "sigmoid":
        return fn_sigmoid
    elif activation == "leakyrelu":
        return fn_leakyrelu
    elif activation == "passthru":
        return fn_passthru
    else:
        assert not isinstance(activation, str), "expecting a callable or a known str"
        return activation


def getDefaultInit(activation):
    if activation.endswith("relu"):
        return "he"
    if activation == "tanh" or activation == "sigmoid":
        return "xavier"  # aka gloriot
    return "lecun"


def getDefaultDistribution(activation):
    if activation.endswith("relu"):
        return "normal"
    if activation == "tanh" or activation == "sigmoid":
        return "uniform"
    return "normal"


def getInitStd(init, shapeIn, shapeOut):
    ninputs = np.prod(shapeIn)
    noutputs = np.prod(shapeOut)
    if init == "he":  # aka Kaiming, normal - relu
        var = 2 / ninputs
    elif init == "xavier":  # aka Gloriot, uniform - sigm/tanh
        var = 2 / (ninputs + noutputs)  # 6 on numerator in some places
    elif init == "lecun":
        var = 1 / (ninputs + noutputs)
    elif init == "bias2" or init == "bias3":
        var = 0.45**2
    else:
        var = 1.0
    return np.sqrt(var)


def getInitAvg(init, std):
    if init == "bias1":
        wbias, bbias = std, std
    elif init == "bias2":
        wbias, bbias = 0.55, 0.55
    elif init == "bias3":
        wbias, bbias = 0, 0.55
    else:
        wbias, bbias = 0, 0
    return wbias, bbias


# --------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------
class Module:
    def __init__(self):
        self.layers = []

    def zero_grad(self):
        for p in self.parameters():
            p.clear_grad()

    @property
    def outsize(self):
        return np.prod(self.outshape)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return sum([layer.parameters() for layer in self.layers], [])

    @property
    def outshape(self):
        return self.layers[-1].outshape


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Fully Connected
# --------------------------------------------------------------------------
class FC(Module):
    def __init__(
        self,
        ninputs,
        noutputs,
        *,
        init="default",
        distribution="normal",
        activation="relu",
        prefix="",
    ):
        super().__init__()
        # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/

        if init == "default":
            init = getDefaultInit(activation)
        std = getInitStd(init, ninputs, noutputs)
        wavg, bavg = getInitAvg(init, std)
        if distribution == "default":
            distribution = getDefaultDistribution(activation)

        if distribution == "normal":
            w = np.random.normal(wavg, std, [ninputs, noutputs])
            b = np.random.normal(bavg, std, [1, noutputs])
        else:
            w = np.random.uniform(wavg - std, wavg + std, [ninputs, noutputs])
            b = np.random.uniform(bavg - std, bavg + std, [1, noutputs])

        self.w = Var(w, name=prepend(prefix, "w"))
        self.b = Var(b, name=prepend(prefix, "b"))
        self.activation = getActivationFunc(activation)
        self.value = None
        self.name = prefix

    def __call__(self, x):
        xs = x.data.shape
        replicate_rows = Const(np.ones([xs[0], 1]))
        b = replicate_rows @ self.b
        # b1 = self.b.replicate(xs[0], axis=0)
        logit = x @ self.w + b
        self.value = self.activation(logit)
        return self.value

    @property
    def outshape(self):
        noutputs = self.b.shape[1]
        return (noutputs,)

    def parameters(self):
        return [self.w, self.b]

    def __repr__(self):
        return f"FC('{self.name}')"


# --------------------------------------------------------------------------
# 2D convolution
# --------------------------------------------------------------------------
class Conv2D(Module):
    def __init__(
        self,
        dimKernel,
        *,
        init="default",
        distribution="normal",
        activation="relu",
        prefix="",
    ):
        super().__init__()
        self.dimKernel = dimKernel
        assert len(dimKernel) == 4
        zout, w, h, zin = dimKernel
        shapeIn = (w, h, zin)

        if init == "default":
            init = getDefaultInit(activation)
        std = getInitStd(init, shapeIn, zout)
        wavg, bavg = getInitAvg(init, std)

        if distribution == "normal":
            w = np.random.normal(wavg, std, dimKernel)
            b = np.random.normal(bavg, std, [zout])
        else:
            w = np.random.uniform(wavg - std, wavg + std, dimKernel)
            b = np.random.uniform(bavg - std, bavg + std, [zout])

        self.w = Var(w, name=prepend(prefix, "w"))
        self.b = Var(b, name=prepend(prefix, "b"))
        self.activation = getActivationFunc(activation)
        self.value = None
        self.name = prefix

    def __call__(self, x):
        logit = x.conv2d(self.w, self.b, prefix=self.name)
        self.value = self.activation(logit)
        return self.value

    def outshape(self, dimIn):
        xs = np.array(dimIn)
        assert xs.ndim == 3
        ks = np.array(self.dimKernel)
        assert ks.ndim == 4
        zout, ks = ks[0], ks[1:]
        res = (xs - ks + 1).tolist()
        return [zout, *res]

    def parameters(self):
        return [self.w, self.b]

    def __repr__(self):
        return f"Conv2D('{self.name}')"


class MaxPool2D(Module):
    def __init__(self, dimKernel, *, activation="relu", prefix=""):
        super().__init__()
        self.dimKernel = dimKernel
        assert len(dimKernel) == 3
        self.activation = getActivationFunc(activation)
        self.value = None
        self.name = prefix

    def __call__(self, x):
        logit = x.maxpool2d(self.dimKernel, prefix=self.name)
        self.value = self.activation(logit)
        return self.value

    def outshape(self, dimIn):
        raise NotImplementedError

    def parameters(self):
        return []

    def __repr__(self):
        return f"MaxPool2D('{self.name}')"


# --------------------------------------------------------------------------
# Flatten and more general reshape
# --------------------------------------------------------------------------
def Flatten(*, prefix=""):
    return ShapedAs(dimOut=None, prefix=prefix)


class ShapedAs(Module):
    def __init__(self, dimOut, *, prefix=""):
        super().__init__()
        self.dimOut = dimOut
        self.name = prefix

    def __call__(self, x):
        dimOut = self.dimOut
        if dimOut is None:
            n = x.data.shape[0] if x.data.ndim >= 4 else 1
            dimOut = (n, -1)
        return x.shapedas(dimOut, prefix=self.name)

    def outshape(self, dimIn):
        return np.prod(dimIn)

    def parameters(self):
        return []

    def __repr__(self):
        return f"ShapedAs('{self.name}')"


# --------------------------------------------------------------------------
# Dropout
# --------------------------------------------------------------------------
class DropOut(Module):
    def __init__(self, probDrop=0.8, *, prefix=""):
        super().__init__()
        self.probDrop = probDrop
        self.value = None
        self.name = prefix

    def __call__(self, x):
        self.value = x.dropout(self.probDrop, prefix=self.name)
        return self.value

    def outshape(self, dimIn):
        raise NotImplementedError

    def parameters(self):
        return []

    def __repr__(self):
        return f"DropOut('{self.name}', prob={self.prob})"


# --------------------------------------------------------------------------
# networks
# --------------------------------------------------------------------------
class NLP(Module):
    def __init__(
        self,
        ninputs,
        layer_sizes,
        *,
        init="default",
        distribution="normal",
        activation="relu",
    ):
        super().__init__()
        ni = ninputs
        for k, sz in enumerate(layer_sizes):
            self.layers.append(
                FC(
                    ni,
                    sz,
                    prefix=f"L{k}",
                    init=init,
                    distribution=distribution,
                    activation=activation,
                )
            )
            ni = sz
