import numpy as np
import functools


# -----------------------------------------------------------------------------
# given a set of ranges representing nested loops, generate a set of indices
# with all the values of the loops fully expanded as seen on the innermost loop
# e.g ranges=[
#   [1,3,5,7],
#   [2,4,6],
# ]
# will expand to:
# [
#    [1,1,1,3,3,3,5,5,5,7,7,7],
#    [2,4,6,2,4,6,2,4,6,2,4,6],
# ]
# useful to vectorize nested loops
# -----------------------------------------------------------------------------
def iters2indices(ranges):
    loopsAbove = []
    lastSize = 1
    for rng in ranges:
        loopsAbove.append(lastSize)
        lastSize *= rng.size
    loopsBelow = []
    lastSize = 1
    for rng in reversed(ranges):
        loopsBelow.append(lastSize)
        lastSize *= rng.size
    loopsBelow = loopsBelow[::-1]
    expRanges = []
    for i, rng in enumerate(ranges):
        expRanges.append(np.tile(np.repeat(rng, loopsBelow[i]), loopsAbove[i]))
    return expRanges


# -----------------------------------------------------------------------------
# Generate the indices that will convert a conv2d into a matrix multiply as
# per im2col conversion. These indices depend only on shape of input x and
# kernel, so the conversion can be cached (shapes are kept in most networks)
# -----------------------------------------------------------------------------
@functools.cache
def convertCHWfor2Dconv_idx(xShape, kernelShape, strides=None):
    kc, kh, kw = kernelShape
    kernelSize = np.prod(kernelShape)
    c, h, w = xShape
    assert c == kc, f"input number of channels {c} must match kernel's {kc}"
    if strides is None:
        strides = (1, 1)
    rows = iters2indices(
        [
            np.arange(0, 1 + h - kh, strides[0]),
            np.arange(0, 1 + w - kw, strides[1]),
            np.arange(c),
            np.arange(kh),
            np.arange(kw),
        ]
    )
    rows[3] += rows[0]  # i + 0:kh
    rows[4] += rows[1]  # j + 0:kw

    out = [r.reshape(-1, kernelSize) for r in rows[2:]]
    return out


# -----------------------------------------------------------------------------
# Generate the indices that will convert a conv2d into a matrix multiply as
# per im2col conversion. These indices depend only on shape of input x and
# kernel, so the conversion can be cached (shapes are kept in most networks)
# -----------------------------------------------------------------------------
@functools.cache
def convertNCHWfor2Dconv_idx(xShape, kernelShape, strides=None):
    kc, kh, kw = kernelShape
    kernelSize = np.prod(kernelShape)
    n, c, h, w = xShape
    assert c == kc, f"input number of channels {c} must match kernel's {kc}"
    if strides is None:
        strides = (1, 1)
    rows = iters2indices(
        [
            np.arange(n),
            np.arange(0, 1 + h - kh, strides[0]),
            np.arange(0, 1 + w - kw, strides[1]),
            np.arange(c),
            np.arange(kh),
            np.arange(kw),
        ]
    )
    rows[4] += rows[1]  # i + 0:kh
    rows[5] += rows[2]  # j + 0:kw

    out = [r.reshape(n, -1, kernelSize) for r in (rows[0], *rows[3:])]
    return out


# -----------------------------------------------------------------------------
# Generate the indices that will convert a conv2d into a matrix multiply as
# per im2col conversion. These indices depend only on shape of input x and
# kernel, so the conversion can be cached (shapes are kept in most networks)
# -----------------------------------------------------------------------------
@functools.cache
def convertNHWfor2Dconv_idx(xShape, kernelShape, strides=None):
    kh, kw = kernelShape
    kernelSize = np.prod(kernelShape)
    n, h, w = xShape
    if strides is None:
        strides = (1, 1)
    rows = iters2indices(
        [
            np.arange(n),
            np.arange(0, 1 + h - kh, strides[0]),
            np.arange(0, 1 + w - kw, strides[1]),
            np.arange(kh),
            np.arange(kw),
        ]
    )
    rows[3] += rows[1]  # i + 0:kh
    rows[4] += rows[2]  # j + 0:kw

    out = [r.reshape(n, -1, kernelSize) for r in (rows[0], *rows[3:])]
    return out


# -----------------------------------------------------------------------------
# Convert an image in CHW format into a 2D matrix as per im2col so that the
# matrix multiplied by a column of kernel weights will be equivalent to a 2D
# convolution
# -----------------------------------------------------------------------------
def convertCHWfor2Dconv(x, kernelShape, strides=None):
    if strides is None:
        strides = (1, 1)
    ci, hi, wi = convertCHWfor2Dconv_idx(x.shape, kernelShape, strides)
    return x[ci, hi, wi]


# -----------------------------------------------------------------------------
# Convert an image in CHW format into a 2D matrix as per im2col so that the
# matrix multiplied by a column of kernel weights will be equivalent to a 2D
# convolution
# -----------------------------------------------------------------------------
def convertNCHWfor2Dconv(x, kernelShape, strides=None):
    # expanding batch as a loop was slower
    ni, ci, hi, wi = convertNCHWfor2Dconv_idx(x.shape, kernelShape, strides)
    return x[ni, ci, hi, wi]


# -----------------------------------------------------------------------------
# Convert an image in CHW format into a 2D matrix as per im2col so that the
# matrix multiplied by a column of kernel weights will be equivalent to a 2D
# convolution
# -----------------------------------------------------------------------------
def convertNHWfor2Dconv(x, kernelShape, strides=None):
    # expanding batch as a loop was slower
    ni, hi, wi = convertNHWfor2Dconv_idx(x.shape, kernelShape, strides)
    return x[ni, hi, wi]


# -----------------------------------------------------------------------------
# As the above but image has no channel dimension
# -----------------------------------------------------------------------------
def convertHWfor2Dconv(x, kernelShape, strides=None):
    h, w = kernelShape
    return convertCHWfor2Dconv(np.expand_dims(x, [0]), (1, h, w), strides)


# -----------------------------------------------------------------------------
# Perform a convolution over a HW image (no C dim) as a matrix multiply
# -----------------------------------------------------------------------------
def conv2D_HW(x, k):
    h, w = x.shape
    ky, kx = k.shape
    imgForConv = convertHWfor2Dconv(x, k.shape)
    imgFilt = imgForConv @ k.reshape(-1)
    imgFinal = imgFilt.reshape(h - ky + 1, w - kx + 1)
    return imgFinal


# -----------------------------------------------------------------------------
# 2D convolution of a CHW image using conversion for mattrix multiply
# output converted image as well
# -----------------------------------------------------------------------------
def conv2D_CHW_dualout(x, k, b=None):
    c, h, w = x.shape
    kz, kc, ky, kx = k.shape
    assert c == kc, f"depth of input {c} must match depth of filter {kc}"
    assert h >= ky, f"height of input must be at least the one of filter {ky}"
    assert w >= kx, f"width of input must be at least the one of filter {kx}"
    imgForConv = convertCHWfor2Dconv(x, (kc, ky, kx))
    kplanes = k.reshape(kz, -1).T
    if b is None:
        imgFilt = imgForConv @ kplanes
    else:
        (bz,) = b.shape
        assert bz == kz, f"size of bias {bz} must match zout of filter {kz}"
        imgFilt = imgForConv @ kplanes + b
    imgFinal = imgFilt.T.reshape(kz, h - ky + 1, w - kx + 1)
    return imgFinal, imgForConv


# -----------------------------------------------------------------------------
# Like above but don't output converted image
# -----------------------------------------------------------------------------
def conv2D_CHW(x, k, b=None):
    return conv2D_CHW_dualout(x, k, b)[0]


# -----------------------------------------------------------------------------
# 2D convolution of a NCHW image using conversion for mattrix multiply
# output converted image as well
# -----------------------------------------------------------------------------
def conv2D_NCHW_dualout(x, k, b=None):
    n, c, h, w = x.shape
    kz, kc, ky, kx = k.shape
    assert c == kc, f"depth of input {c} must match depth of filter {kc}"
    assert h >= ky, f"height of input must be at least the one of filter {ky}"
    assert w >= kx, f"width of input must be at least the one of filter {kx}"
    imgForConv = convertNCHWfor2Dconv(x, (kc, ky, kx))
    kplanes = k.reshape(kz, -1).T
    if b is None:
        imgFilt = imgForConv @ kplanes
    else:
        (bz,) = b.shape
        assert bz == kz, f"size of bias {bz} must match zout of filter {kz}"
        imgFilt = imgForConv @ kplanes + b
    imgFinal = np.transpose(imgFilt, [0, 2, 1]).reshape(n, kz, h - ky + 1, w - kx + 1)
    return imgFinal, imgForConv


# -----------------------------------------------------------------------------
# Like above but don't output converted image
# -----------------------------------------------------------------------------
def conv2D_NCHW(x, k, b=None):
    return conv2D_NCHW_dualout(x, k, b)[0]


# -----------------------------------------------------------------------------
# 2D max pool of a NCHW image using image conversion to matrix as per im2col
# -----------------------------------------------------------------------------
def maxpool2D_NCHW_ref(x, ks):
    n, c, h, w = x.shape
    kc, ky, kx = ks
    sy, sx = ky, kx  # fixed strides for now
    assert sx >= kx, f"Only allowed x stride >= filter width (non overlapping case)"
    assert sy >= ky, f"Only allowed y stride >= filter height (non overlapping case)"
    assert c == kc, f"depth of input {c} must match depth of filter {kc}"
    assert h >= ky, f"height of input must be at least the one of filter {ky}"
    assert w >= kx, f"width of input must be at least the one of filter {kx}"
    batchImgFilt3D = []
    batchImgMask3D = []
    for ni in range(n):
        imgFilt3D = []
        imgMask3D = []
        for ci in range(c):
            imgForConv = convertHWfor2Dconv(x[ni, ci, :, :], (ky, kx), (sy, sx))
            imgFilt = np.max(imgForConv, axis=1).reshape(-1, 1)
            imgMask = (imgForConv == imgFilt) * 1
            imgFilt3D.append(imgFilt)
            imgMask3D.append(imgMask)
        batchImgFilt3D.append(np.stack(imgFilt3D))
        batchImgMask3D.append(np.stack(imgMask3D))
    resImgFilt3D = np.stack(batchImgFilt3D)
    resImgMask3D = np.stack(batchImgMask3D)

    resImgFinal3D = resImgFilt3D.reshape(n, c, h // sy, w // sx)
    return resImgFinal3D, resImgMask3D
    

def maxpool2D_NCHW(x, ks):
    n, c, h, w = x.shape
    kc, ky, kx = ks
    sy, sx = ky, kx  # fixed strides for now
    assert sx >= kx, f"Only allowed x stride >= filter width (non overlapping case)"
    assert sy >= ky, f"Only allowed y stride >= filter height (non overlapping case)"
    assert c == kc, f"depth of input {c} must match depth of filter {kc}"
    assert h >= ky, f"height of input must be at least the one of filter {ky}"
    assert w >= kx, f"width of input must be at least the one of filter {kx}"
    imgForConvFull = convertNCHWfor2Dconv(x, (kc, ky, kx), (sy, sx))  # n, . , c ky kx
    imgForConvFull = imgForConvFull.reshape(n, -1, c, ky * kx)  # n, . , c, ky kx
    imgForConvFull = np.transpose(imgForConvFull, [0, 2, 1, 3])  # n, c, ., ky kx
    imgMax = np.max(imgForConvFull, axis=3)  # n, c, .
    resImgFinal = imgMax.reshape(n, c, h // sy, w // sx)  # n, c, . , .
    resImgMask = (imgForConvFull == imgMax.reshape(n, c, -1, 1)) * 1  # n, c, ., ky kx 
    return resImgFinal, resImgMask