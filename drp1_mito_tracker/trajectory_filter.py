import os

import numpy as np
import pandas as pd
from numpy import sqrt, square
from scipy.signal import savgol_filter


def filter_tracks(load_dir: str, save_dir: str):
    """ This will load all the tracking files with ending "_tracking.csv" from the load_dir. It will then go through
    all trajectories and filter. Filter conditions are
    1. At least 134 frames of uninterrupted trajectory data
    2. None of the perpendicular angles are more than 15 degrees away from 90 degrees.
    3. The position of the mitochondrial skeleton point doesn't move more than 5 pixel more than the drp1 tracking point

    :param load_dir: Directory where the tracking files with ending "_tracking.csv" are located.
    :param save_dir: Directory where to save the resulting Tracks.csv file.
    """
    # Load Tracking files
    files = []
    for (dir_path, dir_names, filenames) in os.walk(load_dir):
        for s in filenames:
            if s.endswith("_tracking.csv"):
                tracking_path = os.path.join(dir_path, s)
                ldf = pd.read_csv(tracking_path)
                ldf['File'] = s[:-13]
                files.append(ldf)
        break

    df = pd.DataFrame()
    cid = 1
    for file_df in files:
        filtered_ids = extract_filtered_ids(file_df)
        for i in filtered_ids:
            cf, a, cuts = continues_frames(file_df, i)
            if cf == 0:
                continue
            for p, ai in enumerate(a):
                if ai >= 134:
                    offset = int((ai - 134) / 2)
                    temp_df = file_df[file_df.id == i][cuts[p] + offset:cuts[p] + 134 + offset]
                    temp_df.loc[temp_df['id'] == i, 't'] -= min(temp_df[temp_df.id == i].t)
                    temp_df.loc[temp_df['id'] == i, 'z'] -= temp_df[temp_df.id == i].z.values[0]
                    # Mirror the z-axis so, that the most distance part is on the positive axis
                    if abs(temp_df[temp_df.id == i].z.min()) > abs(temp_df[temp_df.id == i].z.max()):
                        temp_df.loc[temp_df['id'] == i, 'z'] *= -1
                    # Mirror the angle so that the mean is on the positive side
                    if temp_df[temp_df.id == i].angle.mean() < 0:
                        temp_df.loc[temp_df['id'] == i, 'angle'] *= -1
                    temp_df['oid'] = i
                    temp_df.id = cid
                    cid += 1
                    df = pd.concat([df, temp_df])
                    continue

    df.to_csv(os.path.join(save_dir, 'Tracks.csv'), index=False)


def extract_filtered_ids(df):
    """ Get trajectory ids for trajectories that pass the filter criteria

    :param df: DataFrame with all tracking points with their skeleton data in
    :return: Ids of the trajectories that fit the conditions
    """
    filtered_ids = []

    # Conditions for perpendicular angle and not too far away from mito border
    cond = ((abs(abs(df.perp_angle) - 90) < 15) | (df.d < 3)) & (df.d - df.r <= 5)
    for i in df.id.unique():
        dfi = df[df.id == i]

        # Has enough data that fits perpendicular and distance condition
        cond1 = len(dfi.id[cond]) >= 134
        if not cond1:
            continue

        # If the mito position change is much bigger than the drp1 position change,
        # we probably have a jump from one mito to another, which gives us invalid data
        dm = sqrt(square(dfi.mx[:-1].values - dfi.mx[1:].values) + square(dfi.my[:-1].values - dfi.my[1:].values))
        dd = sqrt(square(dfi.x[:-1].values - dfi.x[1:].values) + square(dfi.y[:-1].values - dfi.y[1:].values))
        mito_drp1_jump_difference = abs(dm - dd)
        a = np.argmax(mito_drp1_jump_difference)
        if np.max(mito_drp1_jump_difference) > 5 and dm[a] > dd[a]:
            continue

        filtered_ids.append(i)
    return filtered_ids


def continues_frames(df, i):
    """ Gathers information of the different continues frames areas in a trajectory

    :param df: DataFrame with all tracking points with their skeleton data in
    :param i: ID of the trajectory
    :return: Count of different frame areas, length of the different frame areas, frame areas
    """
    cond = ((abs(abs(df.perp_angle) - 90) < 15) | (df.d < 3)) & (df.d - df.r <= 5)
    dfi = df[(df.id == i) & cond]
    cuts = np.argwhere(dfi.t.values[1:] - dfi.t.values[:-1] > 1).flatten()
    bcuts = np.append(np.insert(cuts, obj=[0], values=[-1]), len(dfi) - 1)
    dif = np.diff(bcuts)
    return np.sum(np.floor(dif / 134)), dif, bcuts + 1


def extract_features(df, spacing, frame_interval):
    """ Extracts several features from the trajectories.

    :param df: DataFrame with all tracking points with their skeleton data in
    :param spacing: spacing between pixels
    :param frame_interval: spacing between frames
    :return: DataFrame with several features extracted from the trajectories in the DataFrame
    """
    time = df.t.max() * frame_interval
    ndf = pd.DataFrame(index=df.id.unique())
    for i in df.id.unique():
        dfi = df[(df.id == i)]
        z = dfi.z * spacing
        d = dfi.d * spacing
        r = dfi.r * spacing
        zs = (z.values[1:] - z.values[:-1])
        sds = d * np.sign(dfi.angle)
        ds = sds.values[1:] - sds.values[:-1]
        dist = np.sqrt(np.square(zs) + np.square(ds))

        travel_distance = np.sum(dist)
        ndf.loc[ndf.index == i, 'Travel Distance'] = travel_distance

        travel_z_distance = np.sum(abs(zs))
        ndf.loc[ndf.index == i, 'Travel z-Distance'] = travel_z_distance

        travel_d_distance = np.sum(abs(ds))
        ndf.loc[ndf.index == i, 'Travel d-Distance'] = travel_d_distance

        ndf.loc[ndf.index == i, 'Avg. z-Velocity'] = savgol_filter(z, 11, 1, 1, frame_interval).mean()
        ndf.loc[ndf.index == i, 'Avg. d-Velocity'] = (sds.values[-1] - sds.values[0]) / time
        ndf.loc[ndf.index == i, 'Avg. Angle-Velocity'] = savgol_filter(dfi.angle, 11, 1, 1, frame_interval).mean()

        ndf.loc[ndf.index == i, 'Avg. unsigned d/r'] = (d / r).mean()
        ndf.loc[ndf.index == i, 'Std. unsigned d/r'] = (d / r).std()

        ndf.loc[ndf.index == i, 'Avg. signed d/r'] = (sds / r).mean()
        ndf.loc[ndf.index == i, 'Std. signed d/r'] = (sds / r).std()

        ndf.loc[ndf.index == i, 'Avg. Angle'] = dfi.angle.mean()
        ndf.loc[ndf.index == i, 'Std. Angle'] = dfi.angle.std()

        ndf.loc[ndf.index == i, 'Avg. Radius'] = r.mean()
        ndf.loc[ndf.index == i, 'Std. Radius'] = r.std()

        ndf.loc[ndf.index == i, 'Avg. z'] = z.mean()
        ndf.loc[ndf.index == i, 'Std. z'] = z.std()

        ndf.loc[ndf.index == i, 'Range'] = abs(z.max() - z.min())

        ndf.loc[ndf.index == i, 'Range Angle'] = abs(dfi.angle.max() - dfi.angle.min())

        ndf.loc[ndf.index == i, 'Median z'] = z.median()

        displacement = (z.values[-1] ** 2 + (sds.values[-1] - sds.values[0]) ** 2) ** (1 / 2)
        ndf.loc[ndf.index == i, 'Displacement'] = displacement
        ndf.loc[ndf.index == i, 'Straightness'] = displacement / travel_distance

    return ndf
