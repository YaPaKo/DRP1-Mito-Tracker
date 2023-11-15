from math import asin, pi, atan2
from multiprocessing import Pool, cpu_count

import numpy as np
import cupy as cp
from numpy import abs as np_abs
from cupy import abs as cp_abs
from numpy import array, unique, rad2deg, cos, sqrt, square, arctan2, argsort
from numpy import max as np_max
from numpy.linalg import norm
from scipy.signal import savgol_filter
from tqdm import tqdm


class DFHolder:
    """
    Helper class for the usage of multiprocessing with several arguments
    """

    def __init__(self, df, mito_df):
        self.df = df
        self.mito_df = mito_df

    def work(self, i):
        return nearest_mito_radi_point(self.mito_df[self.mito_df.t == self.df.t[i]], self.df.x[i], self.df.y[i])


def add_mito_data(df, mito_df):
    """ Searches the nearest mitochondrial skeleton point for every tracking points and adds its data to the tracking
    point. The direction of the mitochondrial skeleton will be kept from flipping. The z and angle projection will also
    be made here.

    :param df: DataFrame with all tracking points in
    :param mito_df: DataFrame of the mitochondrial skeleton data
    :return: DataFrame with all tracking points in combined with the mitochondrial skeleton data nearst to the
    tracking points
    """

    print("Nearest points")
    # extractor = DFHolder(df, mito_df)
    #
    # indicies = len(df.index)
    #
    # chunks = int(np.ceil(indicies / cpu_count()))
    # with Pool() as p:
    #     results = list(tqdm(p.imap(extractor.work, range(indicies), chunksize=chunks), total=indicies))
    # results = array([r for r in results]
    results = []
    t = -1
    for i in tqdm(df.index):
        if t != df.t[i]:
            t = df.t[i]
            mito_t = mito_df[mito_df.t == t]
        results.append(nearest_mito_radi_point(mito_t, df.x[i], df.y[i]))
    del mito_df
    results = array(results)

    # results = array([nearest_mito_radi_point(mito_df[mito_df.t == df.t[i]], df.x[i], df.y[i]) for i in tqdm(df.index)])

    # Add additional colums to the DataFrame
    df = df.assign(r=.0, rl=.0, rr=.0, d=.0, perp_angle=.0, angle=.0, mito_dir=.0, mx=.0, my=.0, z=.0)

    print("Assign mito radi values to tracking point")
    for i in tqdm(df.index):
        assign_mito_radi_values_to_tracking_point(i, df, *results[i])
    del results

    # Checks for angles that are beyond the found radius and adjust the upper angle range accordingly
    print("Angle check")
    for i in tqdm(unique(df.id)):
        base_cond = (df.id == i) & ((abs(abs(df.perp_angle) - 90) < 15) | (df.d < 3))
        cond_r = base_cond & (df.perp_angle < 0) & (df.d - df.rr <= 5)
        cond_l = base_cond & (df.perp_angle > 0) & (df.d - df.rl <= 5)

        if len(df[df.id == i]) < 45:
            continue

        rmax = 0
        lmax = 0

        if len(df[cond_r]) > 11:
            d1 = df.d[cond_r]
            r1 = df.rr[cond_r]
            rmax = np_max(savgol_filter((d1 - r1).values, 11, 1))

        if len(df[cond_l]) > 11:
            d2 = df.d[cond_l]
            r2 = df.rl[cond_l]
            lmax = np_max(savgol_filter((d2 - r2).values, 11, 1))

        for j in df.index[df.id == i]:
            mito_r = (df.rl[j] + lmax if df.perp_angle[j] > 0 else df.rr[j] + rmax)
            rad_angle = asin(min(df.d[j], mito_r) / mito_r)
            rad_angle = rad_angle if df.perp_angle[j] > 0 else -rad_angle
            df.at[j, 'angle'] = rad2deg(rad_angle)

    return df


def assign_mito_radi_values_to_tracking_point(i, df, t, x, y, u, rr, rl):
    """ Adds the several mitochondrial skeleton data to the tracking points.
    Checks for flipping mitochondrial skeleton direction and flips it back.
    Calculates the primarily projected angle and the position on the z axis.

    :param i: Index of the current tracking point
    :param df: DataFrame with all tracking points in
    :param t: Current Frame
    :param x: x position of the mitochondrial skeleton point
    :param y: y position of the mitochondrial skeleton point
    :param u: direction of the mitochondrial skeleton point
    :param rr: right radius of the mitochondrial skeleton point
    :param rl: left radius of the mitochondrial skeleton point
    """
    # Get id of current object
    oid = df.id[i]
    df_cond = df[(df.id == oid) & (df.t < t)]

    # Check if mito direction flipped and fix it
    if not df_cond.empty and abs(angle_difference(u, df_cond.mito_dir.values[-1])) > (pi / 2):
        u += pi if u < 0 else -pi
        rr, rl = rl, rr

    # Save data from nearest mito skeleton point in data frame
    df.at[i, 'rr'] = rr
    df.at[i, 'rl'] = rl
    df.at[i, 'mx'] = x
    df.at[i, 'my'] = y
    df.at[i, 'mito_dir'] = u

    xx = df.x[i]
    yy = df.y[i]

    # Get angle and distance to mito skeleton
    b = atan2(yy - y, xx - x)
    perp_angle = (rad2deg(u - b) - 180) % 360 - 180
    df.at[i, 'perp_angle'] = perp_angle
    d = norm([yy - y, xx - x])
    df.at[i, 'd'] = d

    cr = (rl if perp_angle > 0 else rr)
    df.at[i, 'r'] = cr

    rad_angle = asin(min(d, cr) / cr)
    rad_angle = rad_angle if perp_angle > 0 else -rad_angle

    df.at[i, 'angle'] = rad2deg(rad_angle)

    # Get position on cylinder z-axis
    if not df_cond.empty:
        curr_mito_x = x
        curr_mito_y = y
        prev_mito_x = df_cond.mx.values[-1]
        prev_mito_y = df_cond.my.values[-1]
        mito_move_distance = norm([curr_mito_x - prev_mito_x, curr_mito_y - prev_mito_y])
        mito_move_angle = atan2(curr_mito_y - prev_mito_y, curr_mito_x - prev_mito_x)
        prev_mito_dir = df_cond.mito_dir.values[-1]
        part = cos(angle_difference(mito_move_angle, prev_mito_dir))
        df.at[i, 'z'] = df_cond.z.values[-1] + part * mito_move_distance


def nearest_mito_radi_point(df_mito_radi, x, y):
    """Get the nearest point on mito skeleton from the (x,y) position.

    :param df_mito_radi: DataFrame of mito radi
    :param x: x position of point
    :param y: y position of point
    :return: DataFrame entry of the nearest skeleton point
    """
    # b0 = x - df_mito_radi.x
    # b1 = y - df_mito_radi.y
    #
    # distances = sqrt(square([b0, b1]).sum(axis=0))
    # angles = df_mito_radi.u - arctan2(b1, b0)
    # weighted = distances + np_abs(cos(angles)) * distances
    #
    # return df_mito_radi.values[argsort(weighted.values)[0]]

    b0 = x - cp.array(df_mito_radi.x)
    b1 = y - cp.array(df_mito_radi.y)

    distances = cp.sqrt(cp.square(cp.array([b0, b1])).sum(axis=0))
    angles = cp.array(df_mito_radi.u) - cp.arctan2(b1, b0)
    weighted = distances + cp_abs(cp.cos(angles)) * distances

    return df_mito_radi.values[cp.argsort(weighted)[0].get()]


def angle_difference(a, b):
    """ Calculates the difference between two angles and keeps them in the range of (-pi,pi).
    Values have to be in radians.

    :param a: First angle in radians
    :param b: Second angle in radians
    :return: Angle difference in radians
    """
    return (((b - a) - pi) % (2 * pi)) - pi
