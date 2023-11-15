import cupy as cp
from skimage.filters import threshold_multiotsu


def threshold_cmotsu(x, threshold=1, bins=1024):
    x = cp.array(x)
    return threshold_multiotsu(hist=tuple([x.get() for x in cp.histogram(x, bins=bins)]))[threshold]
