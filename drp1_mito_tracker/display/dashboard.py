from math import ceil

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from numpy import max as np_max, array, concatenate
from pandas import DataFrame
from scipy.signal import savgol_filter


def show(mito_cts,
         drp1_cts,
         mtdna_cts,
         mito_df: DataFrame,
         drp1_df: DataFrame,
         mtdna_df: DataFrame,
         mito_cmap,
         drp1_cmap,
         mtdna_cmap,
         frame_interval,
         spacing,
         unit,
         drp1_track_cmap='viridis',
         drp1_track_point_c='white',
         mito_drp1_track_point_c='aqua',
         mtdna_track_cmap='plasma',
         mtdna_track_point_c='magenta',
         mito_mtdna_track_point_c='yellow',
         drp1_cond=None,
         mtdna_cond=None,
         window=11,
         immediate=False):
    max_t = max(mito_df.t)

    # region Widgets
    # Build ipywidgets sliders, one per non-planar dimension.
    play = ipywidgets.widgets.Play(
        value=0,
        min=0,
        max=max_t,
        step=1,
        interval=frame_interval * 1000,
        description="Press play",
        disabled=False
    )
    slider = ipywidgets.IntSlider(description="Frame", max=max_t, continuous_update=immediate)
    ipywidgets.widgets.jslink((play, 'value'), (slider, 'value'))
    widgets = {
        "Play": play,
        "Frame": slider
    }
    if drp1_cts is not None:
        widgets["drp1ID"] = ipywidgets.IntSlider(description="DRP1 ID", min=min(drp1_df.id), max=max(drp1_df.id),
                                                 continuous_update=immediate)
    if mtdna_cts is not None:
        widgets["mtdnaID"] = ipywidgets.IntSlider(description="mtDNA ID", min=min(mtdna_df.id), max=max(mtdna_df.id),
                                                  continuous_update=immediate)
    # widgets["All DRP1"] = ipywidgets.Checkbox()
    # widgets["All mtDNA"] = ipywidgets.Checkbox()
    widgets["Show DRP1"] = ipywidgets.Checkbox(True)
    widgets["Show mtDNA"] = ipywidgets.Checkbox(True)
    widgets["Mito radius"] = ipywidgets.Checkbox()
    widgets["Smoothing"] = ipywidgets.Checkbox()
    # endregion

    if drp1_cond is None:
        drp1_cond = drp1_df.id >= 0
    if mtdna_cond is None:
        mtdna_cond = mtdna_df.id >= 0

    saved = {
        "drp1_id": 0,
        "mtdna_id": 0,
        "t": -1,
        "show_drp1": True,
        "show_mtdna": True,
        "smooth": False
    }

    # Figure Setting
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    # plt.figure()
    fig = plt.figure(frameon=True, figsize=(16, 8), layout="constrained")
    fig.tight_layout(pad=0)
    # ax = axes.ravel()

    gs = GridSpec(4, 2, figure=fig)
    video = fig.add_subplot(gs[0:3, 0])
    # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
    angle = fig.add_subplot(gs[0, 1])
    frame_axis_label = angle.twiny()
    angular_velocity = fig.add_subplot(gs[1, 1], sharex=angle)
    z_axis = fig.add_subplot(gs[2, 1], sharex=angle)
    z_velocity = fig.add_subplot(gs[3, 1], sharex=angle)
    z_angle = fig.add_subplot(gs[3, 0])
    ax = {
        "video": video,
        "angle": angle,
        "z_axis": z_axis,
        "angular_velocity": angular_velocity,
        "z_velocity": z_velocity,
        "frame_axis_label": frame_axis_label,
        "z_angle": z_angle
    }

    video.set_xlim([0, mito_cts[0].shape[1] - 1])
    video.set_ylim([mito_cts[0].shape[0] - 1, 0])
    video.set_xlabel(f"x [px]")
    video.set_ylabel(f"y [px]")

    time = np.linspace(0, ceil(len(mito_cts) * frame_interval), 10)
    ticks = [t / frame_interval for t in time]

    angle.set_xlim([0, len(mito_cts)])
    angle.set_xticks(ticks, np.round(time))
    angle.set_ylim([-100, 100])
    angle.set_yticks(range(-90, 91, 30))
    angle.set_ylabel("Angle ϕ [deg]")
    # ang.get_xaxis().set_visible(False)
    angle.hlines(0, 0, len(mito_cts), color="black", linestyles="dashed")

    z_axis.set_ylabel(f"z [{unit}]")

    angular_velocity.set_ylabel("Angle vel [deg/s]")
    angular_velocity.hlines(0, 0, len(mito_cts), color="black", linestyles="dashed")

    z_velocity.set_xlabel("Time [s]")
    z_velocity.set_ylabel(f"z vel [{unit}/s]")
    z_velocity.hlines(0, 0, len(mito_cts), color="black", linestyles="dashed")

    frame_axis_label.set_xlabel("Time [Frame]")
    frame_axis_label.set_xlim([0, len(mito_cts)])
    # frame_axis_label.set_xticks(range(0, len(mito_cts), 20))

    z_angle.set_ylim([-100, 100])
    z_angle.set_yticks(range(-90, 91, 30))
    z_angle.set_xlabel(f"z [{unit}]")
    z_angle.set_ylabel("Angle ϕ [deg]")

    # region Mito / Drp1 Image
    # ax[0].imshow(zeros(cts_m[0].shape), cmap="magma", zorder=0)
    ax["video"].set_facecolor('black')
    mito_img = ax["video"].imshow(mito_cts[0], cmap=mito_cmap, zorder=1, vmin=0, vmax=np_max(mito_cts) * 3 / 5)
    drp1_img = ax["video"].imshow(drp1_cts[0], cmap=drp1_cmap, zorder=2, vmin=0, vmax=np_max(drp1_cts) * 2 / 5)
    mtdna_img = ax["video"].imshow(mtdna_cts[0], cmap=mtdna_cmap, zorder=2, vmin=0, vmax=np_max(mtdna_cts) * 2 / 5)
    # endregion

    # Scatter
    # Show what smoothing the radius did
    smooth_mito_skel = ax["video"].scatter([], [], s=1.5, color="aqua", alpha=.5, zorder=3)

    # Show tracked point
    # Tracked Drp1
    drp1_tracking_point = ax["video"].scatter([], [], s=50, c=drp1_track_point_c, marker="x", zorder=6)
    # Tracked drp1 point on Mito
    mito_drp1_tracking_point = ax["video"].scatter([], [], s=50, c=mito_drp1_track_point_c, marker="x", zorder=6)
    # Tracked mtDNA
    mtdna_tracking_point = ax["video"].scatter([], [], s=50, c=mtdna_track_point_c, marker="x", zorder=6)
    # Tracked point on Mito
    mito_mtdna_tracking_point = ax["video"].scatter([], [], s=50, c=mito_mtdna_track_point_c, marker="x", zorder=6)

    # Mito skel
    # sc4 = ax[0].scatter([], [], s=1.5, color="yellow", alpha=.5, zorder=3)

    # Tracked point on Mito in z-angle plot
    mito_drp1_z_angle_point = z_angle.scatter([], [], s=50, c="red", marker="x", zorder=6)
    mito_mtdna_z_angle_point = z_angle.scatter([], [], s=50, c="yellow", marker="x", zorder=6)

    scatter_plots = {"smooth mito skel": smooth_mito_skel,
                     "drp1 tracking point": drp1_tracking_point,
                     "mito drp1 tracking point": mito_drp1_tracking_point,
                     "mito drp1 z-angle point": mito_drp1_z_angle_point,
                     "mtdna tracking point": mtdna_tracking_point,
                     "mito mtdna tracking point": mito_mtdna_tracking_point,
                     "mito mtdna z-angle point": mito_mtdna_z_angle_point}

    col = {}

    mito_radius_lines = []

    # Dashed line of current frame on plot
    angle_frame_line = ax["angle"].vlines(0, 100, -100, color="black", linestyles="dashed")
    z_frame_line = ax["z_axis"].vlines(0, min(drp1_df.z) - 1, max(drp1_df.z) + 1, color="black", linestyles="dashed")
    angular_vel_frame_line = ax["angular_velocity"].vlines(0, -1, +1, color="black", linestyles="dashed")
    z_vel_frame_line = ax["z_velocity"].vlines(0, -1, +1, color="black", linestyles="dashed")

    # Create image plot with interactive sliders.
    def recalc(**kwargs):
        frame = kwargs["Frame"]

        mito_radius = kwargs["Mito radius"]

        smoothing = kwargs["Smoothing"]

        # region MITO
        if frame != saved["t"]:
            mito_img.set_data(mito_cts[frame])
            scatter_plots["smooth mito skel"].remove()
            scatter_plots["smooth mito skel"] = ax["video"].scatter(mito_df.x[mito_df.t == frame],
                                                                    mito_df.y[mito_df.t == frame],
                                                                    s=1.5, color="aqua", alpha=.5, zorder=3)
            angle_frame_line.set_segments([array([[frame, -100], [frame, 100]])])

        for i in range(len(mito_radius_lines)):
            if mito_radius_lines[i] is not None:
                # noinspection PyUnresolvedReferences
                for j in range(len(mito_radius_lines[i])):
                    mito_radius_lines[i][j].remove()
                mito_radius_lines[i] = None

        mito_radius_lines.clear()

        if mito_radius:
            # Show mito radius
            for tt, xx, yy, uu, rr, rl in mito_df[(mito_df.t == frame)].values:
                dx, dy = np.cos(uu), np.sin(uu)
                mito_radius_lines.append(
                    ax["video"].plot([xx + dy * rr, xx - dy * rl], [yy - dx * rr, yy + dx * rl], alpha=.25, c="orange",
                                     zorder=4))
        # endregion MITO

        # region DRP1
        drp1_id = kwargs["drp1ID"]
        show_drp1 = kwargs["Show DRP1"]

        drp1_range = draw_channel("drp1",
                                  drp1_df,
                                  drp1_cond,
                                  drp1_cts,
                                  drp1_img,
                                  drp1_id,
                                  frame,
                                  show_drp1,
                                  smoothing,
                                  window,
                                  spacing,
                                  frame_interval,
                                  saved,
                                  col,
                                  ax,
                                  scatter_plots,
                                  drp1_track_cmap,
                                  mito_drp1_track_point_c,
                                  drp1_track_point_c)
        # endregion DRP1

        # region mtDNA
        mtdna_id = kwargs["mtdnaID"]
        show_mtdna = kwargs["Show mtDNA"]

        mtdna_range = draw_channel("mtdna",
                                   mtdna_df,
                                   mtdna_cond,
                                   mtdna_cts,
                                   mtdna_img,
                                   mtdna_id,
                                   frame,
                                   show_mtdna,
                                   smoothing,
                                   window,
                                   spacing,
                                   frame_interval,
                                   saved,
                                   col,
                                   ax,
                                   scatter_plots,
                                   mtdna_track_cmap,
                                   mito_mtdna_track_point_c,
                                   mtdna_track_point_c)
        # endregion mtDNA

        if drp1_id == 7:
            print(drp1_range)

        z_min = min(drp1_range[0] if show_drp1 else 0, mtdna_range[0] if show_mtdna else 0)
        z_max = max(drp1_range[1] if show_drp1 else 0, mtdna_range[1] if show_mtdna else 0)
        z_margin = (z_max - z_min) * 0.025
        z_min -= z_margin
        z_max += z_margin

        dadt_min = min(drp1_range[2] if show_drp1 else 0, mtdna_range[2] if show_mtdna else 0)
        dadt_max = max(drp1_range[3] if show_drp1 else 0, mtdna_range[3] if show_mtdna else 0)
        dadt_margin = (dadt_max - dadt_min) * 0.025
        dadt_min -= dadt_margin
        dadt_max += dadt_margin

        dzdt_min = min(drp1_range[4] if show_drp1 else 0, mtdna_range[4] if show_mtdna else 0)
        dzdt_max = max(drp1_range[5] if show_drp1 else 0, mtdna_range[5] if show_mtdna else 0)
        dzdt_margin = (dzdt_max - dzdt_min) * 0.025
        dzdt_min -= dzdt_margin
        dzdt_max += dzdt_margin

        z_angle.set_xlim(z_min, z_max)
        angular_velocity.set_ylim(dadt_min, dadt_max)
        z_axis.set_ylim(z_min, z_max)
        z_velocity.set_ylim(dzdt_min, dzdt_max)

        z_frame_line.set_segments([array([[frame, z_min], [frame, z_max]])])
        angular_vel_frame_line.set_segments([array([[frame, dadt_min], [frame, dadt_max]])])
        z_vel_frame_line.set_segments([array([[frame, dzdt_min], [frame, dzdt_max]])])

        saved["t"] = frame
        saved["smooth"] = smoothing

        fig.suptitle(f"DRP1-ID: {drp1_id}, mtDNA-ID: {mtdna_id}, Time: {(frame * frame_interval):.2f}, Frame: {frame}")

    slider.observe(recalc, widgets)
    ipywidgets.interact(recalc, **widgets)


def draw_channel(c_type: str,
                 df: DataFrame,
                 cond,
                 cts,
                 img,
                 track_id,
                 frame,
                 c_show,
                 smoothing,
                 window,
                 spacing,
                 frame_interval,
                 saved,
                 col,
                 ax,
                 scatter_plots,
                 track_cmap,
                 mito_track_point_c,
                 track_point_c):
    track_n_angle_cond = (df.id == track_id) & cond
    has_data = not df[track_n_angle_cond].empty
    angle = df.angle[track_n_angle_cond]
    z_axis = df.z[track_n_angle_cond] * spacing
    try:
        dadt = savgol_filter(angle, window, 1, 1, frame_interval)
    except ValueError:
        dadt = None
    try:
        dzdt = savgol_filter(z_axis, window, 1, 1, frame_interval)
    except ValueError:
        dzdt = None
    if frame != saved["t"] or c_show != saved[f"show_{c_type}"]:
        img.set_data(cts[frame]) if c_show else img.set_data(np.zeros(cts[frame].shape))
    # Show blob tacking line
    if track_id != saved[f"{c_type}_id"] or saved["smooth"] != smoothing or saved[f"show_{c_type}"] != c_show:
        df_track_n_angle_cond = df[track_n_angle_cond]
        ts = df_track_n_angle_cond.t
        norm = plt.Normalize(0, len(cts))

        keys_to_remove = [
            f"{c_type}_video",
            f"{c_type}_angle",
            f"{c_type}_z_axis",
            f"{c_type}_z_angle",
            f"{c_type}_angular_velocity",
            f"{c_type}_z_velocity"
        ]

        for key in keys_to_remove:
            try:
                col[key].remove()
            except (KeyError, ValueError):
                pass

        if has_data and c_show:
            points = array([df_track_n_angle_cond.x, df_track_n_angle_cond.y]).T.reshape(-1, 1, 2)
            segments = concatenate([points[:-1], points[1:]], axis=1)

            bol = np.full(segments.shape, False)
            for i, t_i in enumerate(ts):
                if i == 0:
                    continue
                if ts.values[i] - ts.values[i - 1] != 1:
                    bol[i - 1, :, 0] = [True, True]

            lc = LineCollection(segments, cmap=track_cmap, norm=norm, zorder=5)
            lc.set_array(ts)
            col[f"{c_type}_video"] = ax["video"].add_collection(lc)

            points = array([ts, angle]).T.reshape(-1, 1, 2)
            segments = concatenate([points[:-1], points[1:]], axis=1)
            segments = np.ma.masked_where(bol, segments)
            lc = LineCollection(segments, cmap=track_cmap, norm=norm)
            lc.set_array(ts)
            col[f"{c_type}_angle"] = ax["angle"].add_collection(lc)

            points = array([ts, z_axis]).T.reshape(-1, 1, 2)
            segments = concatenate([points[:-1], points[1:]], axis=1)
            segments = np.ma.masked_where(bol, segments)
            lc = LineCollection(segments, cmap=track_cmap, norm=norm, zorder=5)
            lc.set_array(ts)
            col[f"{c_type}_z_axis"] = ax["z_axis"].add_collection(lc)

            points = array([z_axis, angle]).T.reshape(-1, 1, 2)
            segments = concatenate([points[:-1], points[1:]], axis=1)
            segments = np.ma.masked_where(bol, segments)
            lc = LineCollection(segments, cmap=track_cmap, norm=norm, zorder=5)
            lc.set_array(ts)
            col[f"{c_type}_z_angle"] = ax["z_angle"].add_collection(lc)

            if dadt is not None:
                points = array([ts, dadt]).T.reshape(-1, 1, 2)
                segments = concatenate([points[:-1], points[1:]], axis=1)
                segments = np.ma.masked_where(bol, segments)
                lc = LineCollection(segments, cmap=track_cmap, norm=norm, zorder=5)
                lc.set_array(ts)
                col[f"{c_type}_angular_velocity"] = ax["angular_velocity"].add_collection(lc)

            if dzdt is not None:
                points = array([ts, dzdt]).T.reshape(-1, 1, 2)
                segments = concatenate([points[:-1], points[1:]], axis=1)
                segments = np.ma.masked_where(bol, segments)
                lc = LineCollection(segments, cmap=track_cmap, norm=norm, zorder=5)
                lc.set_array(ts)
                col[f"{c_type}_z_velocity"] = ax["z_velocity"].add_collection(lc)
    # Show tracked blob point
    try:
        scatter_plots[f"mito {c_type} tracking point"].remove()
        scatter_plots[f"{c_type} tracking point"].remove()
        scatter_plots[f"mito {c_type} z-angle point"].remove()
    except ValueError:
        pass
    if c_show:
        point_cond = (df.id == track_id) & (df.t == frame)
        df_point_cond = df[point_cond]
        # Tracked blob point on Mito
        scatter_plots[f"mito {c_type} tracking point"] = ax["video"].scatter(df_point_cond.mx,
                                                                             df_point_cond.my,
                                                                             s=50,
                                                                             c=mito_track_point_c, marker="x",
                                                                             zorder=6)

        # Tracked blob
        scatter_plots[f"{c_type} tracking point"] = ax["video"].scatter(df_point_cond.x, df_point_cond.y, s=50,
                                                                        c=track_point_c, marker="x", zorder=7)

        # Tracked point on z-angle plot
        scatter_plots[f"mito {c_type} z-angle point"] = (ax["z_angle"]
                                                         .scatter([df_point_cond.z * spacing],
                                                                  [df_point_cond.angle],
                                                                  s=50, c=mito_track_point_c, marker="x", zorder=6))
    saved[f"{c_type}_id"] = track_id
    saved[f"show_{c_type}"] = c_show

    return (np.nan_to_num(z_axis.min()),
            np.nan_to_num(z_axis.max()),
            none_to_num(np.min(dadt)),
            none_to_num(np.max(dadt)),
            none_to_num(np.min(dzdt)),
            none_to_num(np.max(dzdt)))


def none_to_num(none, default=0):
    return default if none is None else none
