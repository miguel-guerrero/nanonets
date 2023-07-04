import numpy as np
from .conv import convertCHWfor2Dconv


# this file contains utilities and unoptimized/reference functions of
# the ones in conv.py


# -----------------------------------------------------------------------------
# show differing entries on two arrays
# -----------------------------------------------------------------------------
def show_diffs(a, b):
    eps = 1e-12
    if np.all(np.abs(a - b) < eps):
        return
    print("showing diffs:")
    print("a", a)
    print("b", b)
    ar = a.reshape(-1)
    br = b.reshape(-1)
    assert ar.size == br.size
    for i in range(ar.size):
        if np.abs(ar[i] - br[i]) >= eps:
            print("ar[", i, "]", ar[i])
            print("br[", i, "]", br[i])


# -----------------------------------------------------------------------------
# im2col conversion unoptimized. Image in HW format
# -----------------------------------------------------------------------------
def convertHWfor2Dconv_ref(x, kernelShape, strides=None):
    kh, kw = kernelShape
    h, w = x.shape
    if strides is None:
        strides = (1, 1)
    rows = []
    for i in range(0, 1 + h - kh, strides[0]):
        for j in range(0, 1 + w - kw, strides[1]):
            sub = x[i : i + kh, j : j + kw]
            sub_res = sub.reshape(-1)
            rows.append(sub_res)
    return np.stack(rows)


# -----------------------------------------------------------------------------
# im2col conversion unoptimized. Image in CHW format
# -----------------------------------------------------------------------------
def convertCHWfor2Dconv_ref(x, kernelShape, strides=None):
    kc, kh, kw = kernelShape
    c, h, w = x.shape
    assert c == kc, f"input number of channels {c} must match kernel's {kc}"
    if strides is None:
        strides = (1, 1)
    rows = []
    for i in range(0, 1 + h - kh, strides[0]):
        for j in range(0, 1 + w - kw, strides[1]):
            sub = x[:, i : i + kh, j : j + kw]
            sub_res = sub.reshape(-1)
            rows.append(sub_res)
    return np.stack(rows)


# -----------------------------------------------------------------------------
# reference function to test conv.convertCHWfor2Dconv_idx
# -----------------------------------------------------------------------------
def convertCHWfor2Dconv_idx_ref(xShape, kernelShape, strides=None):
    kc, kh, kw = kernelShape
    kernelSize = np.prod(kernelShape)
    c, h, w = xShape
    assert c == kc, f"input number of channels {c} must match kernel's {kc}"
    if strides is None:
        strides = (1, 1)
    ranges = [[], [], [], [], []]
    for i in range(0, 1 + h - kh, strides[0]):
        for j in range(0, 1 + w - kw, strides[1]):
            for ci in range(kc):
                for hi in range(kh):
                    for wi in range(kw):
                        ranges[0].append(i)
                        ranges[1].append(j)
                        ranges[2].append(ci)
                        ranges[3].append(i + hi)
                        ranges[4].append(j + wi)

    npranges = []
    for r in ranges[2:]:
        newr = np.array(r).reshape(-1, kernelSize)
        npranges.append(newr)

    return npranges


# -----------------------------------------------------------------------------
# 2D convolution of an image in CHW format unoptimized
# -----------------------------------------------------------------------------
def conv2D_CHW_ref(x, k, b=None):
    c, h, w = x.shape
    kz, kc, ky, kx = k.shape
    if b is None:
        b = np.zeros((kz,))
    (bz,) = b.shape
    assert c == kc, f"depth of input {c} must match depth of filter {kc}"
    assert bz == kz, f"size of bias {bz} must match zout of filter {kz}"
    assert h >= ky, f"height of input must be at least the one of filter {ky}"
    assert w >= kx, f"width of input must be at least the one of filter {kx}"
    imgForConv = convertCHWfor2Dconv(x, (kc, ky, kx))
    img3D = []
    for z in range(kz):
        kplane = k[z, :, :, :]
        imgFilt = imgForConv @ kplane.reshape(-1) + b[z]
        imgFinal = imgFilt.reshape(h - ky + 1, w - kx + 1)
        img3D.append(imgFinal)
    return np.stack(img3D)


# -----------------------------------------------------------------------------
# Convert an image in HWC format into a 2D matrix as per im2col so that the
# matrix multiplied by a column of weights will be equivalent to a 2D
# convolution
# -----------------------------------------------------------------------------
def convertHWCfor2Dconv(x, kernelShape):  # unused
    kh, kw, _kc = kernelShape
    h, w, _c = x.shape
    rows = []
    for i in range(1 + h - kh):
        for j in range(1 + w - kw):
            sub = x[i : i + kh, j : j + kw, :]
            sub_res = sub.reshape(-1)
            rows.append(sub_res)
    return np.stack(rows)
