from math import ceil

import ipywidgets
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from numpy import max as np_max, array, concatenate, full
from numpy.ma import masked_where
from pandas import DataFrame
from scipy.signal import savgol_filter

# Colors for mito/drp1 representation
colors = [(1, 0, 0, 0), (1, 0, 0, 1)]  # first color is black, last is red
cm_red = LinearSegmentedColormap.from_list("Custom red", colors, N=100)
colors = [(0, 1, 0, 0), (0, 1, 0, 1)]  # first color is black, last is green
cm_green = LinearSegmentedColormap.from_list("Custom green", colors, N=100)


def trajectory_dashboard(mito, drp1, drp1_df, mito_df, frame_interval, spacing, unit, cond=None, immediate=False):
    """ Shows the mitochondrial and drp1 channel together with the trajectory data and mitochondrial skeleton

    :param mito: Video of the mitochondrial channel
    :param drp1: Video of the drp1 channel
    :param drp1_df: DRP1 tracking DataFrame with mitochondrial data
    :param mito_df: Mitochondrial skeleton
    :param frame_interval: Spacing between frames
    :param spacing: Spacing between pixels
    :param unit: Unit of pixel spacing
    :param cond: DataFrame condition to hide some lines
    :param immediate: Refresh the videos immediately
    """
    # Build ipywidgets sliders, one per non-planar dimension.
    play = ipywidgets.widgets.Play(
        value=0,
        min=0,
        max=max(drp1_df.t),
        step=1,
        interval=frame_interval * 1000,
        description="Press play",
        disabled=False
    )
    slider = ipywidgets.IntSlider(description="Frame", max=max(drp1_df.t), continuous_update=immediate)
    ipywidgets.widgets.jslink((play, 'value'), (slider, 'value'))
    widgets = {
        "Play": play,
        "Frame": slider,
        "TrackID": ipywidgets.IntSlider(description="TrackID", min=min(drp1_df.id), max=max(drp1_df.id),
                                        continuous_update=immediate),
        "Smoothing": ipywidgets.Checkbox()
    }

    cmap = 'viridis'

    if cond is None:
        cond = drp1_df.id >= 0
    current_id = [0]
    current_t = [-1]
    current_smooth = [False]

    fig = plt.figure(frameon=True, figsize=(16, 8), layout="constrained")
    fig.tight_layout(pad=0)

    gs = GridSpec(4, 2, figure=fig)
    im = fig.add_subplot(gs[0:3, 0])
    ang = fig.add_subplot(gs[0, 1])
    top = ang.twiny()
    avl = fig.add_subplot(gs[1, 1], sharex=ang)
    zax = fig.add_subplot(gs[2, 1], sharex=ang)
    zvl = fig.add_subplot(gs[3, 1], sharex=ang)
    za = fig.add_subplot(gs[3, 0])
    ax = [im, ang, zax, avl, zvl, top, za]

    im.set_xlim([0, mito[0].shape[1] - 1])
    im.set_ylim([mito[0].shape[0] - 1, 0])
    im.set_xlabel(f"x [px]")
    im.set_ylabel(f"y [px]")

    time = range(0, ceil(len(mito) * frame_interval), 2)
    ticks = [t / frame_interval for t in time]

    ang.set_xlim([0, len(mito)])
    ang.set_xticks(ticks, time)
    ang.set_ylim([-100, 100])
    ang.set_yticks(range(-90, 91, 30))
    ang.set_ylabel("Angle ϕ [deg]")
    ang.hlines(0, 0, len(mito), color="black", linestyles="dashed")

    zax.set_ylabel(f"z [{unit}]")

    avl.set_ylabel("Angle vel [deg/s]")
    avl.hlines(0, 0, len(mito), color="black", linestyles="dashed")

    zvl.set_xlabel("Time [s]")
    zvl.set_ylabel(f"z vel [{unit}/s]")
    zvl.hlines(0, 0, len(mito), color="black", linestyles="dashed")

    top.set_xlabel("Time [Frame]")
    top.set_xlim([0, len(mito)])
    top.set_xticks(range(0, len(mito), 20))

    za.set_ylim([-100, 100])
    za.set_yticks(range(-90, 91, 30))
    za.set_xlabel(f"z [{unit}]")
    za.set_ylabel("Angle ϕ [deg]")

    # Mito / Drp1 Image
    ax[0].set_facecolor('black')
    mim = ax[0].imshow(mito[0], cmap=cm_red, zorder=1, vmin=0, vmax=np_max(mito) * 3 / 5)
    dim = ax[0].imshow(drp1[0], cmap=cm_green, zorder=2, vmin=0, vmax=np_max(drp1) * 2 / 5)

    # Scatter
    # Show what smoothing the radius did
    sc1 = ax[0].scatter([], [], s=1.5, color="aqua", alpha=.5, zorder=3)

    # Show tracked point
    # Tracked Drp1
    sc2 = ax[0].scatter([], [], s=50, c="white", marker="x", zorder=6)
    # Tracked point on Mito
    sc3 = ax[0].scatter([], [], s=50, c="aqua", marker="x", zorder=6)

    # Tracked point on Mito
    sc4 = za.scatter([], [], s=50, c="red", marker="x", zorder=6)

    sc = [sc1, sc2, sc3, sc4]

    col = [None] * 6

    # Dashed line of current frame on plot
    t1_line = ax[1].vlines(0, 100, -100, color="black", linestyles="dashed")
    t2_line = ax[2].vlines(0, min(drp1_df.z) - 1, max(drp1_df.z) + 1, color="black", linestyles="dashed")
    t3_line = ax[3].vlines(0, -1, +1, color="black", linestyles="dashed")
    t4_line = ax[4].vlines(0, -1, +1, color="black", linestyles="dashed")

    # Create image plot with interactive sliders.
    def recalc(**kwargs):
        frame = kwargs["Frame"]
        track_id = kwargs["TrackID"]
        smoothing = kwargs["Smoothing"]

        track_n_angle_cond = (drp1_df.id == track_id) & cond
        has_data = not drp1_df[track_n_angle_cond].empty

        fig.suptitle(f"Track-ID: {track_id}, Time: {(frame * frame_interval):.2f}, Frame: {frame}")

        window = 11

        an = drp1_df.angle[track_n_angle_cond]
        zs = drp1_df.z[track_n_angle_cond] * spacing

        try:
            dadt = savgol_filter(an, window, 1, 1, frame_interval)
        except ValueError:
            dadt = None
        try:
            dzdt = savgol_filter(zs, window, 1, 1, frame_interval)
        except ValueError:
            dzdt = None

        if frame != current_t[0]:
            mim.set_data(mito[frame])
            dim.set_data(drp1[frame])

            sc[0].remove()
            sc[0] = ax[0].scatter(mito_df.x[mito_df.t == frame], mito_df.y[mito_df.t == frame],
                                  s=1.5, color="aqua", alpha=.5, zorder=3)

            t1_line.set_segments([array([[frame, -100], [frame, 100]])])

        if has_data:
            t2_line.set_segments([array([[frame, min(zs) - 1], [frame, max(zs) + 1]])])
            if dadt is not None:
                t3_line.set_segments([array([[frame, min(dadt) - 1], [frame, max(dadt) + 1]])])
            if dzdt is not None:
                t4_line.set_segments([array([[frame, min(dzdt) - 1], [frame, max(dzdt) + 1]])])

        # Show tacking line
        if track_id != current_id[0] or current_smooth[0] != smoothing:
            ts = drp1_df.t[track_n_angle_cond]
            norm = plt.Normalize(0, len(mito))

            for i in range(len(col)):
                if col[i] is not None:
                    # noinspection PyUnresolvedReferences
                    col[i].remove()
                    col[i] = None

            if has_data:
                points = array([drp1_df.x[track_n_angle_cond], drp1_df.y[track_n_angle_cond]]).T.reshape(-1, 1, 2)
                segments = concatenate([points[:-1], points[1:]], axis=1)

                bol = full(segments.shape, False)
                for i, t_i in enumerate(ts):
                    if i == 0:
                        continue
                    if ts.values[i] - ts.values[i - 1] != 1:
                        bol[i - 1, :, 0] = [True, True]

                lc = LineCollection(segments, cmap=cmap, norm=norm, zorder=5)
                lc.set_array(ts)
                col[0] = ax[0].add_collection(lc)

                points = array([ts, an]).T.reshape(-1, 1, 2)
                segments = concatenate([points[:-1], points[1:]], axis=1)
                segments = masked_where(bol, segments)
                lc = LineCollection(segments, cmap=cmap, norm=norm)
                lc.set_array(ts)
                col[1] = ax[1].add_collection(lc)

                points = array([ts, zs]).T.reshape(-1, 1, 2)
                segments = concatenate([points[:-1], points[1:]], axis=1)
                segments = masked_where(bol, segments)
                lc = LineCollection(segments, cmap=cmap, norm=norm, zorder=5)
                lc.set_array(ts)
                col[2] = ax[2].add_collection(lc)
                offset = abs(max(zs) - min(zs)) * .05
                ax[2].set_ylim([min(zs) - offset, max(zs) + offset])

                points = array([zs, an]).T.reshape(-1, 1, 2)
                segments = concatenate([points[:-1], points[1:]], axis=1)
                segments = masked_where(bol, segments)
                lc = LineCollection(segments, cmap=cmap, norm=norm, zorder=5)
                lc.set_array(ts)
                col[5] = za.add_collection(lc)
                offset = abs(max(zs) - min(zs)) * .05
                za.set_xlim([min(zs) - offset, max(zs) + offset])

                if dadt is not None:
                    points = array([ts, dadt]).T.reshape(-1, 1, 2)
                    segments = concatenate([points[:-1], points[1:]], axis=1)
                    segments = masked_where(bol, segments)
                    lc = LineCollection(segments, cmap=cmap, norm=norm, zorder=5)
                    lc.set_array(ts)
                    col[3] = ax[3].add_collection(lc)
                    offset = abs(max(dadt) - min(dadt)) * .05
                    ax[3].set_ylim([min(dadt) - offset, max(dadt) + offset])

                if dzdt is not None:
                    points = array([ts, dzdt]).T.reshape(-1, 1, 2)
                    segments = concatenate([points[:-1], points[1:]], axis=1)
                    segments = masked_where(bol, segments)
                    lc = LineCollection(segments, cmap=cmap, norm=norm, zorder=5)
                    lc.set_array(ts)
                    col[4] = ax[4].add_collection(lc)
                    offset = abs(max(dzdt) - min(dzdt)) * .05
                    ax[4].set_ylim([min(dzdt) - offset, max(dzdt) + offset])

        # Show tracked point
        point_cond = (drp1_df.id == track_id) & (drp1_df.t == frame)
        # Tracked point on Mito
        sc[2].remove()
        sc[2] = ax[0].scatter(drp1_df.mx[point_cond], drp1_df.my[point_cond], s=50, c="aqua", marker="x", zorder=6)

        # Tracked Drp1
        sc[1].remove()
        sc[1] = ax[0].scatter(drp1_df.x[point_cond], drp1_df.y[point_cond], s=50, c="white", marker="x", zorder=7)

        # Tracked point on z-angle plot
        sc[3].remove()
        sc[3] = za.scatter([drp1_df.z[point_cond] * spacing], [drp1_df.angle[point_cond]], s=50, c="red", marker="x", zorder=6)

        current_id[0] = track_id
        current_t[0] = frame
        current_smooth[0] = smoothing

    slider.observe(recalc, widgets)
    ipywidgets.interact(recalc, **widgets)


def show_z_angle_plots(df: DataFrame, ids, spacing):
    """ Show of the trajectories with the given ids with z-axis and angle

    :param df: DRP1 tracking DataFrame with mitochondrial data
    :param ids: ids of the trajectories to show
    :param spacing: Spacing between pixels
    :return:
    """
    cols = 2
    rows = ceil(len(ids) / cols)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 3 * rows), layout="constrained", sharey=True)
    ax = axes.ravel()

    t_max = np_max(df.t)
    cmap = 'viridis'

    for a, i in enumerate(ids):
        dfi = df[df.id == i]
        zs = dfi.z * spacing
        an = dfi.angle

        ts = dfi.t
        norm = plt.Normalize(0, t_max)

        points = array([zs, an]).T.reshape(-1, 1, 2)
        segments = concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(ts)
        z_offset = (max(zs) - min(zs)) * .05
        ax[a].set_xlim([min(zs) - z_offset, max(zs) + z_offset])
        ax[a].set_ylim([-95, 95])
        ax[a].set_title(f"ID: {i}, OID: {dfi.oid.unique()[0]}, File: {dfi.File.unique()[0]}", fontsize=10)
        ax[a].set_yticks(range(-90, 91, 30))
        ax[a].hlines([-90, 0, 90], min(zs) - z_offset, max(zs) + z_offset, color="black", linestyles="dashed")
        ax[a].vlines(0, -90, 90, color="black", linestyles="dashed")
        ax[a].scatter(zs, an, s=5, c='red', marker='o')
        ax[a].add_collection(lc)
