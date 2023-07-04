import numpy as np
import random
from .conv import convertCHWfor2Dconv, conv2D_CHW, convertNHWfor2Dconv
from .conv import conv2D_NCHW_dualout, convertNCHWfor2Dconv, maxpool2D_NCHW, maxpool2D_NCHW_ref
from .conv_ref import show_diffs
from .dot_show import draw_dot


eps = 1e-9  # used in some places to avoid div by 0


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def extendHigherDimsTo(x, n):
    assert isinstance(n, int)
    if x.ndim >= n:
        return x
    else:
        newShape = (1,) * (n-x.ndim) + x.shape
        return x.reshape(newShape)


def softMax(x0):  # TODO should not reshape
    # substracting a value to all x scales equaly num and den
    x = x0 - np.max(x0, axis=1).reshape(-1, 1)
    norm = np.sum(np.exp(x), axis=1).reshape(-1, 1)
    return np.exp(x) / norm


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
        

# tanh = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
def tanh(x):
    e2x = np.exp(-2*x)
    return (1 - e2x) / (1 + e2x)


def makeNodeIdNext():
    nodeId = 0

    def getnodeid():
        nonlocal nodeId
        nodeId += 1
        return nodeId

    return getnodeid


getNodeId = makeNodeIdNext()


def prepend(prefix, rest):
    return rest if prefix == "" else prefix + "_" + rest


def wrap(x):
    if isinstance(x, Node):
        return x
    return Const(castToArray(x))


def castToArray(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, list):
        return np.array(x)
    else:
        return np.array([[x]])  # 2D


# -----------------------------------------------------------------------------
# Parent of all node classes
# -----------------------------------------------------------------------------
class Node:
    def __init__(self, inputs=None, op=""):
        self.inputs = [] if inputs is None else inputs
        for input in self.inputs:
            assert isinstance(input, Node)
        self.id = getNodeId()
        self.data = None
        self._grad = None
        self._op = f"[{self.id}] {op}"
        self.name = ""
        self._pass_grad = None
        self._topo_sorted = None
        self._needs_grad = False

    def x(self):
        return self.inputs[0].data

    def y(self):
        return self.inputs[1].data

    def z(self):
        return self.inputs[2].data

    def needs_grad(self):
        return self._needs_grad

    @property
    def grad(self):
        return self._grad.reshape(self.data.shape)

    # return a list of nodes where current node shows always before
    # its dependents (inputs), so root is node at index 0
    def toposort(self):
        if self._topo_sorted is None:
            self._topo_sorted = [self]
            visited = {self}
            active_nodes = [self]
            while len(active_nodes) > 0:
                n = active_nodes.pop(0)
                for i in n.inputs:
                    if i not in visited:
                        self._topo_sorted.append(i)
                        active_nodes.append(i)
                        visited.add(i)
        return self._topo_sorted

    def _zero_pass_grads(self):
        for node in self.toposort():
            node._pass_grad = None

    def backward(self):
        # figure out which portions of the graph need grads
        # a node needs grads if any of its children do
        for node in reversed(self.toposort()):
            self._needs_grad = False
        for node in reversed(self.toposort()):
            if not node.needs_grad():
                node._needs_grad = any(inp.needs_grad() for inp in node.inputs)

        # reset grads for this pass
        self._zero_pass_grads()
        # set _pass_grad for root node
        self._pass_grad = np.ones([1, self.data.size])
        # propagate backward
        # see https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf
        for node in self.toposort():
            if node.needs_grad():
                # print(f"backwards for node {node}")
                for idx, in_ in enumerate(node.inputs):
                    if in_.needs_grad():
                        jvp = node.jac_vec_prod(idx, node._pass_grad)
                        # make sure jvp is a column vector
                        assert jvp.ndim == 2
                        assert jvp.shape[1] == 1
                        # and matches expected size
                        assert jvp.size == node.inputs[idx].data.size
                        in_._pass_grad_add(jvp)
            # draw_dot(self.toposort()[0])
        # accumulate pass grads into grad
        for node in self.toposort():
            node._grad_add()

    # must implement on derived classes
    def jac_vec_prod(self, idx, ups):
        raise NotImplementedError

    def _grad_add(self):
        if self._grad is None:
            self._grad = self._pass_grad
        else:
            self._grad += self._pass_grad

    def _pass_grad_add(self, x):
        if self._pass_grad is None:
            self._pass_grad = x
        else:
            self._pass_grad += x

    def clear_grad(self):
        self._grad = None

    def zero_grad(self):
        for node in self.toposort():
            node.clear_grad()

    # basic operators

    def __add__(self, y):
        return PlusNode(self, wrap(y))

    def __radd__(self, y):
        return PlusNode(wrap(y), self)

    def __sub__(self, y):
        return MinusNode(self, wrap(y))

    def __rsub__(self, y):
        return MinusNode(wrap(y), self)

    def __mul__(self, y):
        return MulNode(self, wrap(y))

    def __rmul__(self, y):
        return MulNode(wrap(y), self)

    def __matmul__(self, y):
        return MatMulNode(self, wrap(y))

    def __pow__(self, exp):
        assert isinstance(exp, (float, int))
        return PowNode(self, exp)

    def __truediv__(self, den):
        return self * (den**-1)

    def __rtruediv__(self, num):
        return num * (self**-1)

    # activation functions

    def relu(self, prefix=""):
        n = LeakyReluNode(self, 0)
        n._op = f"[{n.id}] " + prepend(prefix, "relu")
        return n

    def leakyrelu(self, negSlope=0.01, prefix=""):
        return LeakyReluNode(self, negSlope, prefix=prefix)

    def sigmoid(self, prefix=""):
        return SigmoidNode(self, prefix=prefix)

    def tanh(self, prefix=""):
        return TanhNode(self, prefix=prefix)

    # to compute errors

    def abs(self, prefix=""):
        return LeakyReluNode(self, -1.0, prefix=prefix)

    def exp(self, prefix=""):
        return ExpNode(self, prefix=prefix)

    def softmax(self, prefix=""):
        return SoftmaxNode(self, prefix=prefix)

    def logsoftmax(self, prefix=""):
        return LogSoftmaxNode(self, prefix=prefix)

    def nll(self, y, prefix=""):
        return NllNode(self, wrap(y), prefix=prefix)

    def crossentropy(self, y, prefix=""):
        return CrossEntropyNode(self, y, prefix=prefix)

    def celoss(self, y, prefix=""):
        return CeLossNode(self, y, prefix=prefix)

    # reductions

    def sum(self, prefix=""):
        return SumNode(self, prefix=prefix)

    def mean(self, prefix=""):
        return MeanNode(self, prefix=prefix)

    # accessories for specific types of layers

    def replicate(self, cnt, axis, prefix=""):
        return ReplicateNode(self, cnt, axis, prefix=prefix)

    def shapedas(self, dims, prefix=""):
        return ShapedAsNode(self, dims, prefix=prefix)

    def dropout(self, prob=0.8, prefix=""):
        return DropOutNode(self, prob, prefix=prefix)

    def conv2d(self, kernel, bias, prefix=""):
        return Conv2DNode(self, kernel, bias, prefix=prefix)

    def maxpool2d(self, kernelDims, prefix=""):
        return MaxPool2DNode(self, kernelDims, prefix=prefix)

    def __repr__(self):
        return self._op


# -----------------------------------------------------------------------------
# A tensor that needs gradient
# -----------------------------------------------------------------------------
class Var(Node):
    def __init__(self, init_val, name=""):
        super().__init__([], op=name)  # no inputs
        self.data = castToArray(init_val)
        self._pass_grad = None
        self.name = f"[{self.id}] {name}"

    def needs_grad(self):
        return True

    def set(self, x):
        self.data = castToArray(x)

    def get_pass_grad(self):
        return self._pass_grad.reshape(self.data.shape)

    def show(self, typeStr):
        s = f"{typeStr}("
        s += f"{self.data}" if self.data.size <= 10 else f"{self.data.shape}"
        s += f", {self.name})" if self.name else ")"
        return s

    def __repr__(self):
        return self.show("Var")


# -----------------------------------------------------------------------------
# A tensor that does not need gradient
# -----------------------------------------------------------------------------
class Const(Var):
    def __init__(self, init_val, name=""):
        super().__init__(init_val, name or "const")

    def needs_grad(self):
        return False

    def get_pass_grad(self):
        return np.zeros(self.data.shape)

    def __repr__(self):
        return self.show("Const")


# -----------------------------------------------------------------------------
# Operator element wise + node
# -----------------------------------------------------------------------------
class PlusNode(Node):
    def __init__(self, x, y):
        super().__init__([x, y], "+")
        self.data = x.data + y.data

    def jac_vec_prod(self, _idx, ups):
        return ups

    def __repr__(self):
        return f"[{self.id}] PlusNode()"


# -----------------------------------------------------------------------------
# Operator element wise - node
# -----------------------------------------------------------------------------
class MinusNode(Node):
    def __init__(self, x, y):
        super().__init__([x, y], "-")
        self.data = x.data - y.data

    def jac_vec_prod(self, idx, ups):
        return ups if idx == 0 else -ups

    def __repr__(self):
        return f"[{self.id}] MinusNode()"


# -----------------------------------------------------------------------------
# Operator element wise * node
# -----------------------------------------------------------------------------
class MulNode(Node):
    def __init__(self, x, y):
        super().__init__([x, y], "*")
        self.data = x.data * y.data

    def jac_vec_prod(self, idx, ups):
        if idx == 0:  # dout/dx
            return self.y().reshape(-1, 1) * ups
        else:  # dout/dy
            return self.x().reshape(-1, 1) * ups

    def __repr__(self):
        return f"[{self.id}] MulNode()"


# -----------------------------------------------------------------------------
# Matrix multiply node
# -----------------------------------------------------------------------------
class MatMulNode(Node):
    def __init__(self, x, y):
        super().__init__([x, y], "@")
        self.data = x.data @ y.data

    def jac_vec_prod(self, idx, ups):
        r, c = self.data.shape
        if idx == 0:  # dout/dx
            y = self.y()  # k x c
            res = ups.reshape(r, c) @ y.T
        else:  # dout/dy
            x = self.x()  # r x k
            res = x.T @ ups.reshape(r, c)
        return res.reshape(-1, 1)

    def __repr__(self):
        return f"[{self.id}] MatMulNode()"


# -----------------------------------------------------------------------------
# Element wise power to constant function
# -----------------------------------------------------------------------------
class PowNode(Node):
    def __init__(self, x, exp, prefix=""):
        super().__init__([x], prepend(prefix, f"pow {exp}"))
        self.exp = exp
        self.data = np.power(x.data, exp)

    def jac_vec_prod(self, _idx, ups):
        x = self.x().reshape(-1, 1)
        return self.exp * np.power(x, self.exp - 1) * ups


# -----------------------------------------------------------------------------
# LeakyRelu activation function node
# -----------------------------------------------------------------------------
class LeakyReluNode(Node):
    def __init__(self, x, negSlope, prefix=""):
        super().__init__([x], prepend(prefix, f"leakyrelu {negSlope}"))
        self.negSlope = negSlope
        x_pos = x.data > 0
        self.data = x.data * (self.negSlope * ~x_pos + 1.0 * x_pos)

    def jac_vec_prod(self, _idx, ups):
        x = self.x().reshape(-1, 1)
        x_pos = x > 0
        return (self.negSlope * ~x_pos + 1.0 * x_pos) * ups


# -----------------------------------------------------------------------------
# Sigmoid activation function node
# -----------------------------------------------------------------------------
class SigmoidNode(Node):
    def __init__(self, x, prefix=""):
        super().__init__([x], prepend(prefix, "sigmoid"))
        self.data = sigmoid(x.data)

    def jac_vec_prod(self, _idx, ups):
        out = self.data.reshape(-1, 1)
        return (1 - out) * out * ups


# -----------------------------------------------------------------------------
# Tanh activation function node
# -----------------------------------------------------------------------------
class TanhNode(Node):
    def __init__(self, x, prefix=""):
        super().__init__([x], prepend(prefix, "tanh"))
        self.data = tanh(x.data)

    def jac_vec_prod(self, _idx, ups):
        out = self.data.reshape(-1, 1)
        return (1 - out * out) * ups


# -----------------------------------------------------------------------------
# element wise exponential function node
# -----------------------------------------------------------------------------
class ExpNode(Node):
    def __init__(self, x, prefix=""):
        super().__init__([x], prepend(prefix, "exp"))
        self.data = np.exp(x.data)

    def jac_vec_prod(self, _idx, ups):
        out = self.data.reshape(-1, 1)
        return out * ups


# -----------------------------------------------------------------------------
# Softmax function node
# -----------------------------------------------------------------------------
class SoftmaxNode(Node):
    def __init__(self, x, prefix=""):
        super().__init__([x], prepend(prefix, "softmax"))
        self.data = softMax(x.data)

    def jac_vec_prod(self, _idx, ups):
        # TODO check
        # Jij = (yi) .* (I - yj) = Jij.T
        # make the output a row vector
        out = self.data.reshape(1, -1)
        n = out.size
        # replicate its rows
        rr = np.tile(out, n).reshape(n, -1)
        jac = rr * (np.eye(n) - rr.T)
        return jac @ ups


# -----------------------------------------------------------------------------
# LogSoftmax function node
# -----------------------------------------------------------------------------
def logSoftMax(x0):  # TODO should not reshape
    # substracting a value to all x scales equaly num and den
    x = x0 - np.max(x0, axis=1).reshape(-1, 1)
    norm = np.sum(np.exp(x), axis=1).reshape(-1, 1)
    return x - np.log(norm)


class LogSoftmaxNode(Node):
    def __init__(self, x, prefix=""):
        super().__init__([x], prepend(prefix, "log_softmax"))
        self.data = logSoftMax(x.data)

    def jac_vec_prod(self, _idx, ups):
        # jac = dfj/dxi = J.T = Delta_ij - softmax(xi) = I - si
        # compute softmax of input as a column vector
        x = self.x()
        n, m = x.shape
        # replicate its columns
        sm = softMax(x).reshape(x.shape)
        sm3 = np.atleast_3d(sm)
        rc = sm3 @ np.ones((1, m))
        jac = np.eye(m) - rc
        ur = ups.reshape(n, -1, 1)
        res = jac @ ur
        return res.reshape(-1, 1)


# -----------------------------------------------------------------------------
# Negative Log Loss function node
# -----------------------------------------------------------------------------
def nll_deriv(logsoftmax, y):
    res = np.zeros_like(logsoftmax)
    n = res.shape[0]
    res[np.arange(n), y] = -1.0
    return res


class NllNode(Node):
    def __init__(self, logsoftmax, y, prefix=""):
        super().__init__([logsoftmax], prepend(prefix, "nll"))
        self.logsoftmax = logsoftmax.data
        n = self.logsoftmax.shape[0]
        self.cl = y.data.astype(int).reshape(-1)
        self.data = -self.logsoftmax[np.arange(n), self.cl]

    def jac_vec_prod(self, _idx, ups):
        dnll_dy = nll_deriv(self.logsoftmax, self.cl)
        # replicate columns in ups
        ups_rc = np.repeat(ups, dnll_dy.shape[1], axis=1)
        res = dnll_dy * ups_rc
        return res.reshape(-1, 1)


# -----------------------------------------------------------------------------
# CrossEntropy function node. This version penalizes softmax outputs not
# matching expected category
# -----------------------------------------------------------------------------
def crossEntropy(softmax, y):
    assert softmax.shape == y.shape
    p = np.clip(softmax, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def crossEntropy_deriv(softmax, y):
    # dce / dsoftmax
    p = np.clip(softmax, eps, 1 - eps)
    return -y / p + (1 - y) / (1 - p)


class CrossEntropyNode(Node):
    def __init__(self, softmax, y, prefix=""):
        super().__init__([softmax, y], prepend(prefix, f"centropy {y}"))
        self.softmax = softmax.data
        self.data = crossEntropy(self.softmax, y.data)

    def jac_vec_prod(self, idx, ups):
        if idx == 0:
            dce_dsm = crossEntropy_deriv(self.softmax, self.y())
            return dce_dsm.reshape(-1, 1) * ups
        else:
            return np.zeros_like(ups)


# -----------------------------------------------------------------------------
# CrossEntropyLoss function node. This version does not penalize softmax
# outputs not matching expected category
# -----------------------------------------------------------------------------
def crossLoss(softmax, y):
    # note that the 1-y term is not present
    assert softmax.shape == y.shape
    res = -y * np.log(softmax)
    return np.array([[np.sum(res)]])


def crossLoss_deriv(softmax, y):
    # dce / dsoftmax
    return softmax - y


class CeLossNode(Node):
    def __init__(self, preds, y, prefix=""):
        super().__init__([preds, y], prepend(prefix, f"celoss {y}"))
        self.softmax = softMax(preds.data)
        self.data = crossLoss(self.softmax, y.data)

    def jac_vec_prod(self, idx, ups):
        if idx == 0:
            dce_dy = crossLoss_deriv(self.softmax, self.y())
            return dce_dy.reshape(-1, 1) * ups
        else:
            return np.zeros_like(ups)


# -----------------------------------------------------------------------------
# Add up all elements of input tensor, producing a single item tensor
# -----------------------------------------------------------------------------
class SumNode(Node):
    def __init__(self, x, prefix=""):
        super().__init__([x], prepend(prefix, "sum"))
        self.data = np.array([[np.sum(x.data)]])

    def jac_vec_prod(self, _idx, ups):
        res = np.ones_like(self.x()) * ups.item()
        return res.reshape(-1, 1)


# -----------------------------------------------------------------------------
# Average all elements of input tensor, producing a single item tensor
# -----------------------------------------------------------------------------
class MeanNode(Node):
    def __init__(self, x, prefix=""):
        super().__init__([x], prepend(prefix, "mean"))
        x = self.x()
        self.data = np.array([[np.sum(x)]]) / x.size

    def jac_vec_prod(self, _idx, ups):
        x = self.x()
        res = np.ones_like(x) * ups.item() / x.size
        return res.reshape(-1, 1)


# -----------------------------------------------------------------------------
# Given a tensor, replicate over a given dimension
# -----------------------------------------------------------------------------
class ReplicateNode(Node):
    def __init__(self, x, cnt, axis, prefix=""):
        super().__init__([x], prepend(prefix, f"replicate {cnt}, {axis}"))
        x = self.x()
        self.axis = axis
        self.data = np.repeat(x, cnt, axis=axis)

    def jac_vec_prod(self, _idx, ups):
        ur = ups.reshape(self.data.shape)
        return np.sum(ur, axis=self.axis).reshape(-1, 1)


# -----------------------------------------------------------------------------
# Node to reshape a tensor dimensions
# -----------------------------------------------------------------------------
class ShapedAsNode(Node):
    def __init__(self, x, dims, prefix=""):
        super().__init__([x], prepend(prefix, f"shapedas {dims}"))
        self.data = self.x().reshape(dims)

    def jac_vec_prod(self, _idx, ups):
        return ups


# -----------------------------------------------------------------------------
# DropOut node. Randomly (on every evaluation) drop a number of inputs making
# them zero, according to provided probability
# -----------------------------------------------------------------------------
# generate a random mask of size 'shape' for probability of 0 being probDrop
def getDropOutMask(shape, probDrop):
    scale = 1.0 / (1.0 - probDrop)
    size = np.prod(shape)
    numZeros = int(probDrop * size)
    numOnes = size - numZeros
    ll = [1] * numOnes + [0] * numZeros
    x = np.array(ll)
    np.random.shuffle(x)
    return x.reshape(shape) * scale


class DropOutNode(Node):
    def __init__(self, x, probDrop=0.2, prefix=""):
        assert isinstance(probDrop, float)
        super().__init__([x], prepend(prefix, f"dropout {probDrop}"))
        x = self.x()
        self.mask = getDropOutMask(x.shape, probDrop)
        # output is scaled up so that if we remove this layer inference
        # still works
        self.data = self.mask * x

    def jac_vec_prod(self, _idx, ups):
        return self.mask.reshape(-1, 1) * ups

    def __repr__(self):
        return f"[{self.id}] DropOutNode()"


# -----------------------------------------------------------------------------
# Conv2D operator
# -----------------------------------------------------------------------------
def dconv2_dx_ref(x, w, ups, yshape):  # no longer used
    xdepth, xrows, xcols = x.shape
    _resdepth, resrows, rescols = yshape
    upsReshape = ups.reshape([1, resrows, rescols])
    upsExp = np.pad(
        upsReshape,
        [
            (0, 0),
            (xrows - resrows, xrows - resrows),
            (xcols - rescols, xcols - rescols),
        ],
    )
    res = []
    for c in range(xdepth):
        we = np.expand_dims(w[c, ::-1, ::-1], [0, 1])
        res.append(conv2D_CHW(upsExp, we))
    return np.stack(res)


def dconv2_dw_ref(x, w, ups):  # no longer used
    xConv2D = convertCHWfor2Dconv(x, w.shape)
    return xConv2D.T @ ups


def conv_for_dconv2_dx(ups, xshape, resShape):
    xbatch, _xdepth, xrows, xcols = xshape
    resbatch, _resdepth, resrows, rescols = resShape
    assert xbatch == resbatch
    upsReshape = ups.reshape(resShape)
    upsExp = np.pad(
        upsReshape,
        [
            (0, 0),
            (0, 0),
            (xrows - resrows, xrows - resrows),
            (xcols - rescols, xcols - rescols),
        ],
    )
    return upsExp


def dconv2_dx_ref(w, upsExp):
    res = []
    ue = np.expand_dims(upsExp, [0])
    for c in range(w.shape[0]):
        we = np.expand_dims(w[c, ::-1, ::-1], [0, 1])
        res.append(conv2D_CHW(ue, we))
    res = np.stack(res)
    return res


def dconv2_dx(k0, x0):
    # n = x0.shape(0)
    kc, ky, kx = k0.shape
    imgForConv = convertNHWfor2Dconv(x0, (ky, kx))  # n, ., ky kx
    ks = k0[np.arange(kc), ::-1, ::-1]  # kc, ky, kx
    kplanes = ks.reshape(kc, -1).T  # ky kx, kc
    imgFilt = imgForConv @ kplanes  # n, ., kc
    return imgFilt


class Conv2DNode(Node):
    def __init__(self, x, w, b, prefix=""):
        super().__init__([x, w, b], prepend(prefix, "conv2d"))
        self.data, self.xForConv = conv2D_NCHW_dualout(x.data, w.data, b.data)

    def jac_vec_prod(self, idx, ups):
        w = self.y()
        zout, kc, ky, kx = w.shape
        x = self.x()
        n = x.shape[0]
        ups_by_planes = ups.reshape(n, zout, -1)
        if idx == 0:  # dout/dx
            upsExp = conv_for_dconv2_dx(ups_by_planes, x.shape, self.data.shape)

            if True:
                result = sum(
                    [dconv2_dx(w[z, :, :, :], upsExp[:, z, :]) for z in range(zout)]
                )
                result = np.transpose(result, [0, 2, 1]).reshape(x.shape)

            if False:
                result_ref = []
                for ni in range(n):
                    res_ref = sum([dconv2_dx_ref(w[z, :, :, :], upsExp[ni, z, :]) for z in range(zout)])
                    result_ref.append(res_ref)
                result_ref = np.stack(result_ref)
                result_ref = result_ref.reshape(x.shape)
                if not np.all(np.abs(result_ref - result) < 1e-12):
                    show_diffs(result_ref, result)

        elif idx == 1:  # dout/dw
            result = ups_by_planes @ self.xForConv
            result = np.sum(result, axis=0)  # reduce across batch dim

        else:  # dout / db
            result = np.sum(ups_by_planes, axis=(0, 2))  # reduce across batch and lowest dim

        return result.reshape(-1, 1)


# -----------------------------------------------------------------------------
# MaxPool2D operator
# -----------------------------------------------------------------------------
class MaxPool2DNode(Node):
    def __init__(self, x, kernelDims, prefix=""):
        super().__init__([x], prepend(prefix, "maxpool2d"))
        x = self.x()
        self.data, self.imgMask = maxpool2D_NCHW(x, kernelDims)
        if False:
            self.data2, self.imgMask2 = maxpool2D_NCHW_ref(x, kernelDims)
            if not np.all(np.abs(self.data-self.data2) < 1e-12):
                show_diffs(self.data, self.data2)
            if not np.all(np.abs(self.imgMask-self.imgMask2) < 1e-12):
                show_diffs(self.imgMask, self.imgMask2)
        self.inDims = x.shape
        self.kernelDims = kernelDims

    def jac_vec_prod(self, idx, ups):
        # dout/dx
        kc, kh, kw = self.kernelDims
        n, c, h, w = self.inDims
        assert kc == c, f"filter channel count {kc} must match input one {c}"
        sy, sx = kh, kw
        zout = self.data.shape[1]
        assert zout == c, f"channels count must match on input {c} / output {zout}"
        ups_by_planes = ups.reshape(n, zout, -1)

        res = np.zeros(self.inDims)
        k = 0
        nr = np.repeat(np.arange(n), zout)
        zr = np.tile(np.arange(zout), n)
        for i in range(0, h - kh + 1, sy):
            for j in range(0, w - kw + 1, sx):
                mask = self.imgMask[nr, zr, k, :].reshape(n * zout, kh, kw)
                scale = ups_by_planes[nr, zr, k] / np.sum(mask, axis=(1, 2))
                scale = scale.reshape(n * zout, 1, 1)
                res[nr, zr, i : i + kh, j : j + kw] += mask * scale
                k += 1

        if False:
            res_ref = np.zeros(self.inDims)
            for ni in range(n):
                for z in range(zout):
                    k = 0
                    for i in range(0, h - kh + 1, sy):
                        for j in range(0, w - kw + 1, sx):
                            mask = self.imgMask[ni, z, k, :].reshape(kh, kw)
                            res_ref[ni, z, i : i + kh, j : j + kw] += (
                                mask * ups_by_planes[ni, z, k] / np.sum(mask)
                            )
                            k += 1
            if not np.all(np.abs(res_ref - res) < 1e-12):
                show_diffs(res_ref, res)

        return res.reshape(-1, 1)
