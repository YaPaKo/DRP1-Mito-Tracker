from math import ceil, atan2
from multiprocessing import Pool, cpu_count

import cupy as cp
from cupyx.scipy.ndimage import label as cp_label
from numpy import mean, array, uint, empty, concatenate, nonzero, argsort
from pandas import DataFrame
from scipy.linalg import norm
from skimage.morphology import skeletonize
from tqdm import tqdm


class RadiusExtractor:
    """
    Helper class for the usage of multiprocessing with several arguments
    """

    def __init__(self, mito, mito_threshold, mito_maxima_threshold):
        self.mito = mito
        self.mito_threshold = mito_threshold
        self.mito_maxima_threshold = mito_maxima_threshold

    def work(self, t):
        return image_skeleton_data(self.mito.np(t), self.mito_threshold[t], self.mito_maxima_threshold[t])


def video_skeleton_data(mito, mito_threshold, mito_maxima_threshold):
    """ Calls the image_skeleton_data function for every frame via multiprocessing

    :param mito: video of the mitochondria channel
    :param mito_threshold: lower threshold to use to keep mitochondria together
    :param mito_maxima_threshold: higher threshold to filter out low background signals
    :return: list of the mitochondrial skeleton data for every frame
    """
    extractor = RadiusExtractor(mito, mito_threshold, mito_maxima_threshold)
    frames = len(mito)
    chunks = ceil(frames / cpu_count())
    with Pool() as p:
        result = list(tqdm(p.imap(extractor.work, range(frames), chunksize=chunks), total=frames))
    return result


def skeleton_as_dataframe(mito_skel_arr):
    """ Converts the array returned by the video_skeleton_data function into a dataframe
    
    :param mito_skel_arr: array returned by the video_skeleton_data function
    :return: dataframe
    """
    data = array(
        [[t, x, y, u, rr, rl] for t, data in enumerate(mito_skel_arr) for x, y, u, rr, rl in data])
    mito_df = DataFrame(
        {"t": data[:, 0],
         "x": data[:, 1],
         "y": data[:, 2],
         "u": data[:, 3],
         "rr": data[:, 4],
         "rl": data[:, 5]})
    mito_df = mito_df.astype({'t': 'int'})
    return mito_df


def image_skeleton_data(mito, threshold, maxima_threshold, depth=3):
    """ From the mitochondria channel, the mitochondrial skeleton will be extracted. It will be thinned to a strict
    8-connected skeleton and forks on the skeleton will be removed. The skeleton will be then smoothed for further
    data extraction. The smoothed position, the direction v along the skeleton and from there the radius from the
    smoothed points to the threshold will be calculated and returned.

    :param mito: Image of the mitochondria channel
    :param threshold: lower threshold to use to keep mitochondria together
    :param maxima_threshold: higher threshold to filter out low background signals
    :param depth: Depth of over how much pixel should be averages.
    Uses neighboring pixel in depth for both sides plus itself. 2*depth+1
    :return: Smoothed skeleton points with x and y position, direction vector v and right/left radius (x,y,v,rr,rl)
    """
    # Clean the image and get the skeleton
    mito = cp.array(mito)
    im_mito = cp.where(mito > threshold, mito, 0)

    label_mito, _ = cp_label(im_mito > 0)
    keep_label = cp.unique(label_mito * (im_mito > maxima_threshold))

    im_mito[cp.invert(array_values_in_array_cp(label_mito, keep_label))] = 0
    im_mito = im_mito.get()

    d = depth
    w = 2 * d + 1

    skel = skeletonize(im_mito)
    remove_skel_corner_points(skel)
    crop_skel(skel, w)
    remove_skel_corner_points(skel)

    sorted_skel_arr = empty((0, 2), dtype=uint)
    temp = skel.copy()
    while True:
        xs, ys = cp.nonzero(cp.array(temp))[::-1]
        if len(xs) == 0:
            break
        arr = skel_arr(temp, (xs[0].get(), ys[0].get()))
        temp[arr[:, 1], arr[:, 0]] = 0
        sorted_skel_arr = concatenate((sorted_skel_arr, arr), axis=0)
    del temp

    # Get the position of all forks on the skeleton
    forks = [(x, y)
             for y, x in list(zip(*skel.nonzero()))
             if zero_transition(skel[*clockwise_neighbors_idx(skel, x, y)[::-1]]) >= 3
             ]

    rs = []
    for i, _ in enumerate(sorted_skel_arr):
        arr = sorted_skel_arr[i - d:i + d + 1]

        # Ignore too short arrays
        if len(arr) < w:
            continue
        if not continuous(arr):
            continue
        if len([True for a in arr if (a[0], a[1]) in forks]) > 0:
            continue

        xx, yy = mean(arr[:, 0]), mean(arr[:, 1])
        xsm, ysm = skel_mean(skel, arr[d - 1], d)
        xsp, ysp = skel_mean(skel, arr[d + 1], d)
        dx, dy = (xsp - xsm) / 2, (ysp - ysm) / 2
        n = norm([dx, dy])
        dx, dy = dx / n, dy / n
        rl, rr = 0.75, 0.75

        # Extend perpendicular to the skeleton until the threshold ends, to determine the radius
        # To the right
        try:
            while im_mito[int(yy - dx * rr + .5), int(xx + dy * rr + .5)]:
                rr += .25
        except IndexError:
            pass

        # To the left
        try:
            while im_mito[int(yy + dx * rl + .5), int(xx - dy * rl + .5)]:
                rl += .25
        except IndexError:
            pass

        rs.append((xx, yy, atan2(dy, dx), rr, rl))

    return array(rs)


def continuous(arr):
    """ Checks if the neighboring pixel in the array are also neighboring on the 2D plane in an 8-connectivity.

    :param arr: Array of pixel positions
    :return: True if the neighboring pixel in the array are also neighboring on the 2D plane in an 8-connectivity.
    """
    for p1, p2 in zip(arr[:-1], arr[1:]):
        if norm(p1 - p2) > 1.5:
            return False
    return True


def array_values_in_array_cp(number, keep):
    """ Gets an array of all numbers and an array of numbers to keep and returns a mask for the number array.

    :param number: Array of all number
    :param keep: Array of numbers to keep
    :return: Mask for the number array
    """
    b = cp.full(number.shape, False)
    for d in keep:
        b |= (number == d)
    return b


def remove_skel_corner_points(skel):
    """ Crops 4-Connected points from the skeleton, so that it is a strictly 8-connected skeleton.
    It tries to remove the points with the lowest intensity first. Removal will be inline.

    :param skel: The skeleton to remove the corner points from. Removal will be inline.
    """
    xs, ys = nonzero(skel)[::-1]
    # Sort it by intensity to remove lower intensity points first
    idx = argsort(skel[ys, xs])
    for i in range(len(idx)):
        cn = clockwise_neighbors(skel, xs[i], ys[i])
        dn = cn[::2]
        en = cn[1::2]
        for j in range(4):
            if dn[j - 1] and dn[j] and not en[(j + 1) % 4]:
                skel[ys[i], xs[i]] = 0


def crop_skel(skeleton, crop_length):
    """ Crops branches from the skeleton that are as long as the crop_length or smaller.
    The Skeleton will be edited inline.

    :param skeleton: The skeleton to crop the branches from. The Skeleton will be edited inline.
    :param crop_length: How long the branches that should be cropped can be.
    """
    spec = skeleton.nonzero()
    spec = list(zip(*spec))
    con = True
    while con:
        con = False
        for y, x in spec:
            if x == 0 or y == 0 or x == skeleton.shape[1] - 1 or y == skeleton.shape[0] - 1:
                continue
            cni = clockwise_neighbors_idx(skeleton, x, y)
            s = zero_transition(skeleton[cni[::-1]])
            if s >= 3:
                for i in range(len(cni[0])):
                    if skeleton[cni[1][i], cni[0][i]] == 0:
                        continue
                    for j in range(len(cni[0])):
                        if skeleton[cni[1][j], cni[0][j]] == 0:
                            continue
                    arr = [(x, y)]
                    skel_arr_recur(skeleton, arr, (cni[0][i], cni[1][i]), crop_length + 1, False)
                    arr.remove((x, y))
                    if 0 < len(arr) <= crop_length:
                        con = True
                        for p in arr:
                            skeleton[*p[::-1]] = 0


def skel_arr(skel, point: tuple, depth=10000):
    """ Extracts an array from a skeleton at a certain point. It will have the same order as the skeleton.

    :param skel: The skeleton to extract the array from
    :param point: The point to start at
    :param depth: How deep it should search
    :return: The skeleton as an array
    """
    # Return None if this point is invalid
    if skel[*point[::-1]] == 0:
        return None

    arr = [(point[0], point[1])]
    if depth <= 0:
        return array(arr)
    depth -= 1

    px_idx = clockwise_neighbors_idx(skel, *point)
    px = array(skel[*px_idx[::-1]])
    # count = min(2, len(px))

    # Check the horizontal/vertical neighbors first
    for i in nonzero(px[::2])[0]:
        flip = len(arr) > 1
        p = px_idx[0][::2][i], px_idx[1][::2][i]
        if skel[p[::-1]] and p not in arr:
            args = p, depth, flip
            while args is not None:
                args = skel_arr_recur(skel, arr, *args)

    # Check the diagonal neighbors last
    for i in nonzero(px[1::2])[0]:
        flip = len(arr) > 1
        p = px_idx[0][1::2][i], px_idx[1][1::2][i]
        if skel[p[::-1]] and p not in arr:
            args = p, depth, flip
            while args is not None:
                args = skel_arr_recur(skel, arr, *args)

    return array(arr)


def skel_arr_recur(skel, arr, point: tuple, depth, flip):
    if point[0] < 0 or point[0] >= skel.shape[1] or point[1] < 0 or point[1] >= skel.shape[0]:
        return

    arr[:] = add(arr, [point], flip)
    if depth <= 0:
        return

    depth -= 1

    px_idx = clockwise_neighbors_idx(skel, *point)
    px = array(skel[*px_idx[::-1]])

    for i in nonzero(px[::2])[0]:
        p = px_idx[0][::2][i], px_idx[1][::2][i]
        if skel[p[::-1]] and p not in arr:
            return p, depth, flip

    for i in nonzero(px[1::2])[0]:
        p = px_idx[0][1::2][i], px_idx[1][1::2][i]
        if skel[p[::-1]] and p not in arr:
            return p, depth, flip


def add(a, b, flip=False):
    """ Adds two arrays together, either flipped or not

    :param a: First array
    :param b: Second array
    :param flip: If true, second array will be at first
    :return: Added array a+b or flipped added array b+a
    """
    return a + b if not flip else b + a


def skel_mean(skel, point, depth):
    """ Gets the mean position of the skeleton points of a given point to a given depth

    :param skel: The skeleton to extract the array from
    :param point: To point to start the extraction from
    :param depth: The depth of how much should be extracted
    :return: The mean position (x,y) of the point and its neighbors
    """
    arr = skel_arr(skel, point, depth)
    return mean(arr[:, 0]), mean(arr[:, 1])


def zero_transition(sequence):
    """ Return how many transitions from zero to a nonzero number occurs in the sequence.
    In a sequence with length n it checks as follows -1,0,1,...,n, so it wraps around.

    :param sequence: Sequence to be checked for zero to nonzero transitions.
    :return: Number of transitions
    """
    s = 0
    for i in range(len(sequence)):
        if sequence[i - 1] == 0 and sequence[i] != 0:
            s += 1
    return s


def clockwise_neighbors_idx(image, x, y):
    """ Return the indexes of the neighboring pixels in order of P2, P3, ..., P8, P9
    P9 P2 P3
    P8 P1 P4
    P7 P6 P5

    It will mirror the points that are over the edge of the image.

    :param image: The image we want the neighbors from
    :param x: x (width) index of the pixel in the image
    :param y: y (height) index of the pixel in the image
    :return: An one dimensional numpy array with pixel indexes in order of P2, P3, ..., P8, P9
    """
    xp1 = x + 1 if x + 1 < image.shape[1] else x - 1
    xm1 = x - 1 if x - 1 >= 0 else x + 1
    yp1 = y + 1 if y + 1 < image.shape[0] else y - 1
    ym1 = y - 1 if y - 1 >= 0 else y + 1
    return [x, xp1, xp1, xp1, x, xm1, xm1, xm1], [ym1, ym1, y, yp1, yp1, yp1, y, ym1]


def clockwise_neighbors(image, x, y):
    """ Return the values of the neighboring pixels in order of P2, P3, ..., P8, P9
    P9 P2 P3
    P8 P1 P4
    P7 P6 P5

    It will mirror the points that are over the edge of the image.

    :param image: The image we want the neighbors from
    :param x: x (width) index of the pixel in the image
    :param y: y (height) index of the pixel in the image
    :return: An one dimensional numpy array with pixel values in order of P2, P3, ..., P8, P9
    """
    return array(image[*clockwise_neighbors_idx(image, x, y)[::-1]])
