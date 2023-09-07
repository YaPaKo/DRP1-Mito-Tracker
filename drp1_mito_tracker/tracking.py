"""
Methods are mostly adapted from 2019 IAFIG Bioimage Analysis Python Course: Object tracking
Video: https://www.youtube.com/watch?v=mgDUFhly9bc
GitHub: https://github.com/RMS-DAIM/Python-for-Bioimage-Analysis
"""

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import label as cp_label
from numpy import where, array, unique, square, reshape, sqrt, mean, zeros
from pandas import DataFrame
from scipy.optimize import linear_sum_assignment, curve_fit
from skimage.filters import threshold_otsu, threshold_li
from tqdm import tqdm

inf = 100000


def tracker(video,
            threshold_method=threshold_li,
            maxima_threshold_method=threshold_otsu,
            max_distance=15,
            min_coords=0):
    """ Extracts the coordinates from the signal and then puts it together with an adjusted tracking algorithm from
    the "2019 IAFIG Bioimage Analysis Python Course: Object tracking".
 
    :param video: Video channel of the signal
    :param threshold_method: Lower threshold skimage method to keep the signal together (default: Li)
    :param maxima_threshold_method: Higher threshold skimage method to filter out weak signals, mostly coming from
    the cytosol (default: Otsu)
    :param max_distance: Maximal distance the signal coordinates are allowed to jump to be still counted as the same point
    :param min_coords: minimum coordinates that an id needs to be kept (default = 0)
    :return: DataFrame of the tracking coordinates with associated id
    """
    coord = extract_coordinates(video,
                                threshold_method=threshold_method,
                                maxima_threshold_method=maxima_threshold_method
                                )
    coord_df = DataFrame({"t": coord[:, 0].astype(dtype=int), "id": 0, "x": coord[:, 2], "y": coord[:, 1]})
    track_coordinates(coord_df, max_distance=max_distance)
    filter_low_counts(coord_df, min_coords)
    return coord_df


def extract_coordinates_for_frame_cp(drp1: cp.ndarray, frame: int, drp1_threshold, drp1_maxima_threshold):
    """ Extracts all coordinates of drp1 for this frame

    :param drp1: Image of the drp1 channel
    :param frame: Which frame it currently is on
    :param drp1_threshold: Lower threshold to keep drp1 together
    :param drp1_maxima_threshold: Higher threshold filter out weak signals, mostly coming from the cytosol
    :return: Frame and position of the found coordinates
    """
    im_drp1 = cp.where(drp1 > drp1_threshold, drp1, 0)

    label_drp1, _ = cp_label(im_drp1 > 0)
    keep_label = cp.unique(label_drp1 * (im_drp1 > drp1_maxima_threshold))
    if 0 in keep_label:
        keep_label = keep_label[1:]

    shp = drp1.shape
    return [[frame, *cp.unravel_index((im_drp1 * (label_drp1 == i)).argmax(), shp)] for i in keep_label]


class CoordinateExtractor:
    """
    Helper class for the usage of multiprocessing with several arguments
    """

    def __init__(self, video, threshold_method, maxima_threshold_method):
        self.video = video

        # Old method that needed a lot of ram
        # self.threshold = threshold_method(video.get())
        # self.maxima_threshold = maxima_threshold_method(video.get())
        # self.mean_threshold = threshold_method(video.get())
        # self.mean_maxima_threshold = maxima_threshold_method(video.get())

        # Mean from all frames calculated seperately
        self.threshold, self.mean_threshold = calc_threshold_fit(video, threshold_method)
        self.maxima_threshold, self.mean_maxima_threshold = calc_threshold_fit(video, maxima_threshold_method)

    def work(self, t):
        # return extract_coordinates_for_frame_cp(self.video[t], t, self.threshold, self.maxima_threshold)
        return extract_coordinates_for_frame_cp(self.video[t], t, self.threshold[t], self.maxima_threshold[t])


class ThresholdError(Exception):
    """
    Exception for when the threshold seems to high and found too many objects.
    """
    pass


def extract_coordinates(video, threshold_method=threshold_li, maxima_threshold_method=threshold_otsu):
    """ Extracts signal coordinates for the whole video

    :param video: Video channel of the signal
    :param threshold_method: Lower threshold skimage method to keep drp1 together (default: Li)
    :param maxima_threshold_method: Higher threshold skimage method to filter out weak signals, mostly coming from
    :return: Extracted coordinates over all frames
    """
    extractor = CoordinateExtractor(video, threshold_method, maxima_threshold_method)
    if extractor.mean_maxima_threshold < 1500:
        _, cot = cp_label(video[0] > extractor.mean_maxima_threshold)
        if cot > 1000:
            raise ThresholdError(
                f"Unusually low maxima threshold of {extractor.mean_maxima_threshold} < 1500 "
                f"which found too many tracking instances {cot} > 1000.")

    frames = len(video)

    # Multiprocessing has some memory issues under windows
    # chunks = ceil(frames / cpu_count())
    # with Pool() as p:
    #     coordinates = list(tqdm(p.imap(extractor.work, range(frames), chunksize=chunks), total=frames))
    # # coordinates = list(tqdm(p.imap(extractor.work, range(frames)), total=frames))
    # return array([(i[0], i[1], i[2]) for c in coordinates for i in c])

    coordinates = [extractor.work(f) for f in range(frames)]
    return array([(i[0], i[1].get(), i[2].get()) for c in coordinates for i in c])


def get_idx_of_coordinates_for_frame(df: DataFrame, frame: int):
    """(Modified from 2019 IAFIG Bioimage Analysis Python Course: Object tracking)"""
    return df.index[df.t == frame]


def assign_new_ids(df, frame):
    """ Assigns new ids for the current frame to all points that do not have an id yet.
    (Modified from 2019 IAFIG Bioimage Analysis Python Course: Object tracking)

    :param df: DataFrame with all the positions
    :param frame: Frame for which the ids should be assigned to
    """
    idx = get_idx_of_coordinates_for_frame(df, frame)
    max_id = max(df.id)
    for i in idx:
        if df.id[i] == 0:
            max_id = max_id + 1
            df.at[i, 'id'] = max_id
    return None


def get_all_id_idx(df, start_frame, end_frame):
    """ Get all the indexes from start_frame to end_frame which have an id (id > 0)
    (Modified from 2019 IAFIG Bioimage Analysis Python Course: Object tracking)


    :param df: DataFrame with all the positions
    :param start_frame: Start of the timeframe (included in search)
    :param end_frame: End of the timeframe (included in search)
    :return: List of indexes
    """
    rows = df.index[(df.t >= start_frame) & (df.t <= end_frame)]
    id_idx = []
    unique_ids = unique(df.id[rows])
    for uid in unique_ids:
        if uid == 0:
            continue
        id_idx.append(df.index[df.id == uid][-1])
    return id_idx


def calc_cost_matrix(df, tracks, points, max_distance):
    """ Puts together a cost matrix from the tracks (job) and the tracking points (worker). The cost is calculated with
    the Euclidean distance between the last points of the track and the tracking points.
    (Modified from 2019 IAFIG Bioimage Analysis Python Course: Object tracking)

    :param df: DataFrame with all the positions
    :param tracks: Indexes of all the tracks
    :param points: Indexes of all the tracking points
    :param max_distance: The maximum distance the tracking points will be associated to the tracks
    :return: The cost matrix from the tracks (job) and the tracking points (worker)
    """
    x1 = df.x[points]
    x2 = df.x[tracks]

    y1 = df.y[points]
    y2 = df.y[tracks]

    cm = sqrt(square(reshape(x1, (len(x1), 1)) - reshape(x2, (len(x2), 1)).T)
              + square(reshape(y1, (len(y1), 1)) - reshape(y2, (len(y2), 1)).T)).T
    return where(cm > max_distance, inf, cm)


def assign_ids(df, tracks, points, cost_matrix):
    """ Uses the scipy linear_sum_assignment to assign the tracking points to the tracks with the values
    of the given cost matrix.
    (Modified from 2019 IAFIG Bioimage Analysis Python Course: Object tracking)

    :param df: DataFrame with all the positions
    :param tracks: Indexes of all the tracks
    :param points: Indexes of all the tracking points
    :param cost_matrix: The cost matrix from the tracks (job) and the tracking points (worker)
    """
    assignments = linear_sum_assignment(cost_matrix)
    for t, p in zip(assignments[0], assignments[1]):
        if cost_matrix[t, p] < inf:
            df.at[points[p], 'id'] = df.id[tracks[t]]
    return None


def track_coordinates(df, max_distance, look_back_n_frames=5):
    """ Goes through all tracking points to assign them a common id with the Hungarian algorithm.
    (Modified from 2019 IAFIG Bioimage Analysis Python Course: Object tracking)

    :param df: DataFrame with all the positions
    :param max_distance: Maximal distance the drp1 coordinates are allowed to jump to be still counted as the same point
    :param look_back_n_frames: How many end frames of the already found track can be looked back for the tracking
    """
    assign_new_ids(df, 0)
    for frame in tqdm(range(1, max(df.t) + 1)):
        tracks = get_all_id_idx(df, max(0, frame - 1 - look_back_n_frames), frame - 1)
        points = get_idx_of_coordinates_for_frame(df, frame)
        cost_matrix = calc_cost_matrix(df, tracks, points, max_distance)
        assign_ids(df, tracks, points, cost_matrix)
        assign_new_ids(df, frame)
    return None


def filter_low_counts(df, frames):
    """ How many frames a track should at least have to stay in the DataFrame

    :param df: DataFrame with all the positions
    :param frames: Number of frames a track should at least have
    """
    df.drop(df.groupby('id').filter(lambda x: len(x) <= frames).index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    for idx, i in enumerate(df.id.unique()):
        df.loc[df.id == i, 'id'] = idx + 1
    return None


def calc_threshold_fit(video, threshold_method):
    """ Fitting an exponantiol line to the thresholding curve over time to include photobleaching

    :param video: The video we need the threshold for
    :param threshold_method: The thresholding method like Li and Otsu
    :return: An array with the threshold data for every frame
    """
    # Mean from all frames calculated seperately
    threshold = zeros(len(video))
    for i, frame in enumerate(video.get()):
        threshold[i] = threshold_method(frame)

    t = range(0, len(video))
    mean_threshold = mean(threshold)

    def exp(x, a, b, c):
        return a * np.exp(b * x) + c

    # noinspection PyTupleAssignmentBalance
    thres_min = np.min(threshold)
    thres_max = np.max(threshold)
    thres_curve = curve_fit(exp, t, threshold, p0=(threshold[0], -.03, threshold[-1]), maxfev=5000)[0]
    threshold_fit = [max(thres_min, min(thres_max, exp(x, *thres_curve))) for x in t]

    return threshold_fit, mean_threshold
