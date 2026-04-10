#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize 6-command predicted trajectories and 1 GT trajectory per frame.

This script is intended to live at the same level as the `Bench2DriveZoo/` folder.

Input JSON format (list of frames) is expected to contain:
  - predictions: list with 6 elements, each is a (T,2) trajectory
  - ground_truth: (T,2) trajectory
  - ego_fut_cmd_idx: int in [0..5] indicating the selected command (optional)

Example:
  conda run -n b2d_zoo python /home/deyun/git/B2DRepair/visualize_6cmd_trajs.py \
    --input /home/deyun/git/B2DRepair/Bench2DriveZoo/data/infos/b2d_repair_collect_tiny_traj.json \
    --output_dir /home/deyun/git/B2DRepair/trajectory_visualizations_6cmd
"""

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches

try:
    import pickle  # stdlib
except Exception:  # pragma: no cover
    pickle = None


# IMPORTANT: This ordering must match how `command_near` is converted to one-hot in B2D.
# See `Bench2DriveZoo/carla/PythonAPI/carla/agents/navigation/local_planner.py`:
#   RoadOption.LEFT=1, RIGHT=2, STRAIGHT=3, LANEFOLLOW=4, CHANGELANELEFT=5, CHANGELANERIGHT=6
# And `Bench2DriveZoo/mmcv/datasets/B2D_vad_dataset.py::command2hot` does `command -= 1`.
# Therefore indices map to:
#   0: LEFT, 1: RIGHT, 2: STRAIGHT, 3: LANEFOLLOW, 4: CHANGELANELEFT, 5: CHANGELANERIGHT
CMD_LABELS = [
    "Turn Left",  # idx 0
    "Turn Right",  # idx 1
    "Go Straight",  # idx 2
    "Follow Lane",  # idx 3 (LANEFOLLOW)
    "Change Lane Left",  # idx 4
    "Change Lane Right",  # idx 5
]

CMD_COLORS = [
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#ff7f0e",  # orange
    "#8c564b",  # brown
]

def _scatter_traj_band(ax, xy, color="#1f77b4", score=1.0, dot_size=18, points_per_step=18, zorder=6, alpha=1.0):
    """
    MomAD-style trajectory rendering: a smooth band of points along the polyline.
    xy: (T,2) absolute positions.
    score: [0..1], used to fade towards white.
    """
    if xy is None:
        return 0
    pts = np.asarray(xy, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return 0
    total_steps = (pts.shape[0] - 1) * int(points_per_step) + 1
    # Interpolate piecewise linearly to densify
    dense = np.zeros((total_steps, 2), dtype=float)
    for i in range(total_steps - 1):
        a = i // int(points_per_step)
        b = min(a + 1, pts.shape[0] - 1)
        t = (i / float(points_per_step)) - a
        dense[i] = (1.0 - t) * pts[a, :2] + t * pts[b, :2]
    dense[-1] = pts[-1, :2]

    # Fade to white by score (higher score = more saturated)
    rgb = np.array(matplotlib.colors.to_rgb(color), dtype=float)
    rgb = rgb * float(score) + (1.0 - float(score)) * np.ones_like(rgb)
    ax.scatter(dense[:, 0], dense[:, 1], s=float(dot_size), c=[rgb], alpha=float(alpha), linewidths=0, zorder=zorder)
    return int(dense.shape[0])

def _prepend_origin_if_missing(xy, eps=1e-3):
    arr = np.asarray(xy, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return arr
    if float(np.hypot(arr[0, 0], arr[0, 1])) <= float(eps):
        return arr
    zero = np.zeros((1, 2), dtype=float)
    return np.concatenate([zero, arr[:, :2]], axis=0)


def _apply_pretty_axes(ax, xlim, ylim):
    ax.set_aspect("equal", "box")
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.axis("off")

    # Enlarge legend box in pretty mode (applied if legend exists)
    leg = ax.get_legend()
    if leg is not None:
        frame = leg.get_frame()
        frame.set_linewidth(1.2)



def _safe_name(s):
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:180]


def _box_corners_xy(cx, cy, length, width, yaw):
    """
    Compute oriented box corners in XY plane.
    Assumes yaw in radians, x forward, y left (ego frame).
    Returns (5,2) array with closed polygon.
    """
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rot = np.array([[c, -s], [s, c]], dtype=float)
    # local corners (length along x, width along y)
    hl = 0.5 * float(length)
    hw = 0.5 * float(width)
    local = np.array(
        [
            [hl, hw],
            [hl, -hw],
            [-hl, -hw],
            [-hl, hw],
        ],
        dtype=float,
    )
    corners = local @ rot.T + np.array([float(cx), float(cy)], dtype=float)
    corners = np.vstack([corners, corners[0:1]])
    return corners


def _resolve_ego_icon_path(icon_path):
    """
    Resolve relative paths like ego.png: absolute path, then cwd, then this script's directory.
    """
    p = Path(icon_path).expanduser()
    if p.is_file():
        return str(p.resolve())
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        cand = base / p
        if cand.is_file():
            return str(cand.resolve())
    return str(p)


def _autocrop_icon(img, alpha_thresh=0.02, white_thresh=0.98, pad_px=2):
    """
    Auto-crop an icon to its non-background region.

    - If image has alpha channel and is not fully opaque: crop by alpha > alpha_thresh.
    - Otherwise: crop by non-white heuristic (any RGB channel < white_thresh).

    Works best with transparent-background PNGs (like your `ego.png`).
    """
    arr = np.asarray(img)
    if arr.ndim < 3 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return arr

    h, w = int(arr.shape[0]), int(arr.shape[1])
    mask = None

    # Prefer alpha mask when present and meaningful
    if arr.shape[-1] == 4:
        a = arr[..., 3]
        if float(np.mean(a)) < 0.995:
            mask = a > float(alpha_thresh)

    # Fallback: non-white mask
    if mask is None:
        rgb = arr[..., :3]
        mask = np.any(rgb < float(white_thresh), axis=-1)

    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return arr

    y0 = max(int(ys.min()) - int(pad_px), 0)
    y1 = min(int(ys.max()) + int(pad_px) + 1, h)
    x0 = max(int(xs.min()) - int(pad_px), 0)
    x1 = min(int(xs.max()) + int(pad_px) + 1, w)
    if (y1 - y0) < 2 or (x1 - x0) < 2:
        return arr
    return arr[y0:y1, x0:x1].copy()


def _pad_icon_to_aspect(img, target_aspect_wh, pad_value=0.0):
    """
    Pad an image (H,W,C) to a target aspect ratio (width/height) without stretching.
    Keeps content centered; pads with transparent (alpha=0) if RGBA, else pad_value.
    """
    arr = np.asarray(img)
    if arr.ndim < 3 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return arr
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h <= 0 or w <= 0:
        return arr

    target_aspect_wh = float(target_aspect_wh)
    if not np.isfinite(target_aspect_wh) or target_aspect_wh <= 0:
        return arr

    cur_aspect = float(w) / float(h)
    if abs(cur_aspect - target_aspect_wh) / max(target_aspect_wh, 1e-6) < 0.02:
        return arr

    if cur_aspect < target_aspect_wh:
        # Need wider: pad left/right
        new_w = int(np.ceil(float(h) * target_aspect_wh))
        pad_total = max(new_w - w, 0)
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        pad = ((0, 0), (pad_l, pad_r), (0, 0))
    else:
        # Need taller: pad top/bottom
        new_h = int(np.ceil(float(w) / target_aspect_wh))
        pad_total = max(new_h - h, 0)
        pad_t = pad_total // 2
        pad_b = pad_total - pad_t
        pad = ((pad_t, pad_b), (0, 0), (0, 0))

    # Single scalar for all channels (RGBA: 0 => transparent black pixels)
    # A tuple of per-channel constants breaks np.pad on 3D arrays and triggers fallback to black vector silhouette.
    return np.pad(arr, pad, mode="constant", constant_values=0)


def _draw_ego_car(ax, icon_path="", yaw=0.0, zorder=10, anchor="center", forward_axis="y"):
    """
    Draw the ego car at origin in LiDAR BEV frame (x forward, y left).
    - If icon_path is provided, render it via imshow with a fixed meter extent.
    - Otherwise, draw a simple car silhouette (rectangle + nose).
    """
    # Typical passenger car footprint (meters)
    length = 4.6
    width = 2.0
    nose = 0.9

    anchor = str(anchor or "center").lower()
    if anchor not in ["center", "front", "rear"]:
        anchor = "center"

    forward_axis = str(forward_axis or "y").lower()
    if forward_axis not in ["x", "y"]:
        forward_axis = "y"

    # Define where (0,0) attaches to the car in plot coordinates.
    # In our BEV plots: x is horizontal, y is vertical. "forward" is typically +y.
    if forward_axis == "y":
        if anchor == "front":
            ax0, ay0 = 0.0, 0.5 * length  # front edge center (up)
        elif anchor == "rear":
            ax0, ay0 = 0.0, -0.5 * length
        else:
            ax0, ay0 = 0.0, 0.0
        extent = [-0.5 * width - ax0, 0.5 * width - ax0, -0.5 * length - ay0, 0.5 * length - ay0]
        target_aspect_wh = float(width) / float(length)
    else:
        if anchor == "front":
            ax0, ay0 = 0.5 * length, 0.0
        elif anchor == "rear":
            ax0, ay0 = -0.5 * length, 0.0
        else:
            ax0, ay0 = 0.0, 0.0
        extent = [-0.5 * length - ax0, 0.5 * length - ax0, -0.5 * width - ay0, 0.5 * width - ay0]
        target_aspect_wh = float(length) / float(width)

    icon_path = str(icon_path or "")
    if icon_path:
        resolved = _resolve_ego_icon_path(icon_path)
        try:
            img = plt.imread(resolved)
            img = _autocrop_icon(img)
            img = _pad_icon_to_aspect(img, target_aspect_wh)
            # PNG 俯视图资源通常「车头在图片上方」；origin=upper 让首行对齐 extent 的 ymax，
            # 与 forward_axis=y（前方 = +y 向上）一致。origin=lower 会把车头翻到 -y，后视镜看起来像在车尾。
            ax.imshow(
                img,
                extent=extent,
                origin="upper" if forward_axis == "y" else "lower",
                zorder=int(zorder),
                interpolation="bilinear",
            )
            return True
        except Exception as e:
            print(
                "WARNING: ego icon load failed ({} -> {}). Using black vector silhouette. Error: {}".format(
                    icon_path, resolved, e
                )
            )

    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rot = np.array([[c, -s], [s, c]], dtype=float)

    hl = 0.5 * length
    hw = 0.5 * width
    if forward_axis == "y":
        # front at +y
        body = np.array([[hw, hl], [hw, -hl], [-hw, -hl], [-hw, hl]], dtype=float)
        tri = np.array([[0.0, hl + nose], [hw * 0.75, hl], [-hw * 0.75, hl]], dtype=float)
    else:
        # front at +x
        body = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=float)
        tri = np.array([[hl + nose, 0.0], [hl, hw * 0.75], [hl, -hw * 0.75]], dtype=float)

    body_xy = body @ rot.T - np.array([ax0, ay0], dtype=float)
    tri_xy = tri @ rot.T - np.array([ax0, ay0], dtype=float)

    body_patch = mpatches.Polygon(body_xy, closed=True, facecolor="#111111", edgecolor="#111111", linewidth=1.2, zorder=int(zorder))
    nose_patch = mpatches.Polygon(tri_xy, closed=True, facecolor="#111111", edgecolor="#111111", linewidth=1.2, zorder=int(zorder))
    ax.add_patch(body_patch)
    ax.add_patch(nose_patch)
    return True


def _draw_gt_boxes(ax, info, max_boxes=80, alpha=0.6):
    """
    Draw gt boxes from one info dict (as in b2d_infos_*.pkl).
    Expects info['gt_boxes'] shape (N, >=7) and optionally info['gt_names'].
    Convention is assumed: [x, y, z, dx, dy, dz, yaw, ...] in ego frame.
    """
    if not info:
        return 0
    gt_boxes = info.get("gt_boxes", None)
    if gt_boxes is None:
        return 0
    boxes = np.asarray(gt_boxes)
    if boxes.ndim != 2 or boxes.shape[0] == 0 or boxes.shape[1] < 7:
        return 0
    n = int(min(boxes.shape[0], max_boxes))
    for i in range(n):
        x, y = float(boxes[i, 0]), float(boxes[i, 1])
        # NOTE: In B2D infos generated by `prepare_B2D.py`, size is stored as [w, l, h]
        # (converted from extent and then *2). Our box helper expects (length along x, width along y),
        # so we map: length=col4 (l), width=col3 (w).
        dx, dy = float(boxes[i, 4]), float(boxes[i, 3])
        # NOTE: In `prepare_B2D.py`, the stored yaw is `yaw_local_in_lidar_box = -yaw_local - pi/2`.
        # For BEV drawing in LiDAR frame (x forward, y left), we convert back to `yaw_local`.
        yaw_in_box = float(boxes[i, 6])
        yaw = -(yaw_in_box + np.pi / 2.0)
        poly = _box_corners_xy(x, y, dx, dy, yaw)
        ax.plot(poly[:, 0], poly[:, 1], color="#666666", linewidth=1.0, alpha=alpha, zorder=2)
        # Indicate heading: a short line from center to front edge center (+x in box local frame)
        # Our `_box_corners_xy` ordering uses front edge at corners[0] -> corners[1].
        front_center = 0.5 * (poly[0, :2] + poly[1, :2])
        ax.plot([x, front_center[0]], [y, front_center[1]], color="#666666", linewidth=1.6, alpha=min(1.0, alpha + 0.25), zorder=3)
        # Emphasize the front edge so "head/tail" is visually obvious
        ax.plot([poly[0, 0], poly[1, 0]], [poly[0, 1], poly[1, 1]], color="#666666", linewidth=2.4, alpha=min(1.0, alpha + 0.25), zorder=3)
        ax.plot(x, y, marker=".", color="#666666", markersize=2, alpha=alpha, zorder=2)
    return n


def _load_occ_npz(occ_root, scene_token, frame_idx, version_prefix="v1", reduce_mode="max"):
    """
    Load VAD occ cache (.npz) saved by Bench2DriveZoo/adzoo/vad/test.py.
    Path layout observed in this repo:
      {occ_root}/{version_prefix}/{scene_name}/{frame_idx:06d}.npz
    where scene_token in json may be like "v1/SceneName"; we strip leading "v1/".
    The saved npz contains key 'occ' with shape (T, H, W) (T=6).
    """
    if not occ_root:
        return None
    st = str(scene_token or "")
    if st.startswith(f"{version_prefix}/"):
        st = st[len(version_prefix) + 1 :]
    npz_path = os.path.join(str(occ_root), version_prefix, st, f"{int(frame_idx):06d}.npz")
    if not os.path.exists(npz_path):
        return None
    try:
        z = np.load(npz_path)
        occ = z.get("occ", None)
        if occ is None:
            return None
        occ = np.asarray(occ)
        if occ.ndim == 3:
            if reduce_mode == "t0":
                occ2 = occ[0]
            else:
                occ2 = np.max(occ, axis=0)
        elif occ.ndim == 2:
            occ2 = occ
        else:
            return None
        return occ2.astype(float)
    except Exception:
        return None


def _draw_occ_background(ax, occ2d, alpha=0.45, draw_contour=True):
    """
    Draw occupancy grid background aligned to x/y meters.
    PlanningMetric uses:
      X_BOUND=[-50,50,0.5], Y_BOUND=[-50,50,0.5]
    and writes into an image where axis1 ~ x, axis0 ~ -y (cv2 coords),
    so we flip axis0 to align with y upward.
    """
    if occ2d is None:
        return False
    occ_img = np.flipud(occ2d)  # axis0 corresponds to -y
    x_min, x_max = -50.0, 50.0
    y_min, y_max = -50.0, 50.0
    ax.imshow(
        occ_img,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap="Reds",
        alpha=float(alpha),
        zorder=1,
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    if draw_contour:
        try:
            xs = np.linspace(x_min, x_max, occ_img.shape[1])
            ys = np.linspace(y_min, y_max, occ_img.shape[0])
            ax.contour(xs, ys, occ_img, levels=[0.5], colors=["#aa0000"], linewidths=1.0, alpha=0.9, zorder=2)
        except Exception:
            pass
    return True


def _load_infos_pkl(pkl_path):
    if pickle is None:
        raise RuntimeError("pickle is not available in this Python environment.")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    # expected: list of dicts (infos) or dict with 'infos'
    if isinstance(data, dict) and "infos" in data:
        infos = data["infos"]
    else:
        infos = data
    if not isinstance(infos, list):
        raise ValueError(f"Unexpected pkl format: {type(infos)}")
    return infos


def _build_info_index(infos):
    """
    Build index for quick lookup: (folder, frame_idx) -> info dict
    """
    idx = {}
    for it in infos:
        if not isinstance(it, dict):
            continue
        folder = it.get("folder", None)
        frame_idx = it.get("frame_idx", None)
        if folder is None or frame_idx is None:
            continue
        idx[(str(folder), int(frame_idx))] = it
    return idx


def _load_map_infos_pkl(map_infos_pkl):
    if pickle is None:
        raise RuntimeError("pickle is not available in this Python environment.")
    with open(map_infos_pkl, "rb") as f:
        m = pickle.load(f)
    if not isinstance(m, dict):
        raise ValueError(f"Unexpected map_infos format: {type(m)}")
    return m


def _transform_world_to_frame_xy(points_xyz, T_world_to_frame):
    """
    points_xyz: (N,3) in world/map coordinates
    T_world_to_frame: 4x4, same convention as B2D_vad_dataset.get_map_info:
        p_frame = (T @ p_homo.T).T  with p_homo = [x,y,z,1]
    returns (N,2) xy in that frame (LiDAR BEV when T is world2lidar)
    """
    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return None
    T = np.asarray(T_world_to_frame, dtype=float)
    if T.shape != (4, 4):
        return None
    ones = np.ones((pts.shape[0], 1), dtype=float)
    homo = np.concatenate([pts[:, :3], ones], axis=1)  # (N,4)
    out = (T @ homo.T).T
    return out[:, :2]


def _world2lidar_from_b2d_info(info):
    """4x4 world2lidar from b2d_infos pkl (same frame as gt_boxes / VAD ego_fut trajs)."""
    if not isinstance(info, dict):
        return None
    sensors = info.get("sensors", None)
    if not isinstance(sensors, dict):
        return None
    lidar = sensors.get("LIDAR_TOP", None)
    if not isinstance(lidar, dict):
        return None
    w2l = lidar.get("world2lidar", None)
    if w2l is None:
        return None
    T = np.asarray(w2l, dtype=float)
    if T.shape != (4, 4):
        return None
    return T


def _plot_contiguous_runs(ax, xy, in_mask, color, linewidth, alpha, zorder):
    if xy is None or in_mask is None:
        return 0
    xy = np.asarray(xy, dtype=float)
    m = np.asarray(in_mask, dtype=bool)
    if xy.ndim != 2 or xy.shape[0] == 0 or m.shape[0] != xy.shape[0]:
        return 0
    nseg = 0
    start = None
    for i in range(len(m)):
        if m[i] and start is None:
            start = i
        if (not m[i] or i == len(m) - 1) and start is not None:
            end = i if not m[i] else i + 1
            if end - start >= 2:
                seg = xy[start:end]
                ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
                nseg += 1
            start = None
    return nseg


def _draw_map_polylines_bev(
    ax,
    map_info,
    world2lidar,
    xlim=(-50, 50),
    ylim=(-50, 50),
    alpha=0.55,
    pretty=False,
    draw_trigger_volumes=True,
):
    """
    Draw lane / trigger volume polylines as background.
    Map points in b2d_map_infos.pkl are world/map coords; B2D_vad_dataset.get_map_info uses
    sensors['LIDAR_TOP']['world2lidar'] — same frame as gt_boxes and collected ego_fut trajs.
    """
    if not isinstance(map_info, dict):
        return 0
    if world2lidar is None:
        return 0
    n = 0

    # In pretty mode, prefer *sampled* polylines (MomAD-like clean look).
    keys = [("lane_sample_points", "#7a7a7a", 2.4 if pretty else 0.9)]
    if not pretty:
        keys = [
            ("lane_points", "#7a7a7a", 1.2),
            ("lane_sample_points", "#7a7a7a", 0.9),
        ]
    if draw_trigger_volumes:
        keys += [
            ("trigger_volumes_points", "#4c78a8", 2.2 if pretty else 1.2),
            ("trigger_volumes_sample_points", "#4c78a8", 1.6 if pretty else 0.9),
        ]

    # IMPORTANT: By default we DO NOT drop any map/lane information.
    # We only crop by view window (xlim/ylim) and plot contiguous in-view segments.
    for key, color, lw in keys:
        pts_list = map_info.get(key, None)
        if not isinstance(pts_list, list):
            continue
        for pts in pts_list:
            arr = np.asarray(pts, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] < 2:
                continue
            xy = _transform_world_to_frame_xy(arr[:, :3], world2lidar)
            if xy is None:
                continue
            in_mask = (
                (xy[:, 0] >= float(xlim[0]))
                & (xy[:, 0] <= float(xlim[1]))
                & (xy[:, 1] >= float(ylim[0]))
                & (xy[:, 1] <= float(ylim[1]))
            )
            # If nothing is in view, skip.
            if not bool(np.any(in_mask)):
                continue
            n += _plot_contiguous_runs(
                ax,
                xy,
                in_mask,
                color=color,
                linewidth=lw,
                alpha=alpha,
                zorder=0,
            )
            # MomAD-like: add small markers on sampled points (purely aesthetic, no filtering).
            if pretty and key.endswith("sample_points"):
                sub = xy[in_mask]
                if sub.shape[0] > 0:
                    step = max(1, int(sub.shape[0] // 140))  # avoid too many markers
                    ax.plot(
                        sub[::step, 0],
                        sub[::step, 1],
                        linestyle="None",
                        marker="o",
                        markersize=2.6,
                        color=color,
                        alpha=min(1.0, alpha + 0.10),
                        zorder=1,
                    )
    return n

def compute_global_limits(
    frames,
    only_valid=False,
    pad_ratio=0.08,
    min_range=5.0,
    info_index=None,
    include_boxes=False,
):
    """Compute global x/y limits across all frames so every saved figure shares the same scale."""
    xs = []
    ys = []

    for frame in frames:
        if only_valid and not frame.get("fut_valid_flag", False):
            continue

        preds = frame.get("predictions", []) or []
        gt = frame.get("ground_truth", []) or []

        for cmd_idx in range(6):
            traj = preds[cmd_idx] if cmd_idx < len(preds) else []
            if not traj:
                continue
            arr = np.asarray(traj, dtype=float)
            xs.append(arr[:, 0])
            ys.append(arr[:, 1])

        if gt:
            gt_arr = np.asarray(gt, dtype=float)
            xs.append(gt_arr[:, 0])
            ys.append(gt_arr[:, 1])

        if include_boxes and info_index is not None:
            token = frame.get("folder", None) or frame.get("scene_token", None)
            frame_idx = frame.get("frame_idx", None)
            if token is not None and frame_idx is not None:
                info = info_index.get((str(token), int(frame_idx)), None)
                if isinstance(info, dict) and info.get("gt_boxes", None) is not None:
                    b = np.asarray(info["gt_boxes"], dtype=float)
                    if b.ndim == 2 and b.shape[0] > 0 and b.shape[1] >= 2:
                        xs.append(b[:, 0])
                        ys.append(b[:, 1])

    # Fallback: if nothing valid, keep a default symmetric window around origin
    if not xs or not ys:
        half = min_range
        return (-half, half), (-half, half)

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    ymin = float(np.min(y))
    ymax = float(np.max(y))

    # Add padding; also ensure a minimum visible range
    xr = max(xmax - xmin, min_range)
    yr = max(ymax - ymin, min_range)
    pad_x = xr * pad_ratio
    pad_y = yr * pad_ratio

    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)

    half_x = 0.5 * xr + pad_x
    half_y = 0.5 * yr + pad_y

    return (xmid - half_x, xmid + half_x), (ymid - half_y, ymid + half_y)


def visualize_one_frame(
    frame_data,
    frame_global_idx,
    output_dir,
    draw_title=True,
    xlim=None,
    ylim=None,
    info_index=None,
    draw_boxes=False,
    occ_root="",
    draw_occ=False,
    occ_reduce="max",
    map_infos=None,
    draw_map=False,
    style="default",
    pretty_range=40.0,
    map_draw_trigger_volumes=False,
    compare_frame_data=None,
    primary_name="Base",
    compare_name="Repaired",
    ego_icon="",
    ego_anchor="center",
    ego_forward_axis="y",
):
    predictions = frame_data.get("predictions", [])
    ground_truth = frame_data.get("ground_truth", [])
    ego_fut_cmd_idx = frame_data.get("ego_fut_cmd_idx", -1)
    predictions_b = (compare_frame_data or {}).get("predictions", []) if compare_frame_data else []
    ego_fut_cmd_idx_b = (compare_frame_data or {}).get("ego_fut_cmd_idx", -1) if compare_frame_data else -1

    is_pretty = str(style).lower() in ["momad", "pretty", "clean"]
    fig_size = (20, 20) if is_pretty else (10, 10)
    fig, ax = plt.subplots(figsize=fig_size)

    # Current pose origin: LiDAR BEV frame (same as VAD gt_boxes / ego_fut in b2d_infos & collect JSON)
    _draw_ego_car(ax, icon_path=ego_icon, yaw=0.0, zorder=10, anchor=ego_anchor, forward_axis=ego_forward_axis)
    if not is_pretty:
        # Keep legend entry for ego in non-pretty mode (pretty mode uses custom legend items)
        ax.plot([], [], "ko", markersize=10, label="Ego (Current)")

    # Optional: draw occupancy background from VAD occ cache (not roads; dynamic obstacles/ped)
    if draw_occ:
        scene_token = frame_data.get("scene_token", "")
        frame_idx = frame_data.get("frame_idx", frame_global_idx)
        occ2d = _load_occ_npz(occ_root, scene_token, frame_idx, version_prefix="v1", reduce_mode=occ_reduce)
        _draw_occ_background(ax, occ2d)

    # Optional: map polylines: world -> LiDAR BEV via world2lidar (not world2ego; matches VAD dataset)
    if draw_map and map_infos is not None and info_index is not None:
        town_name = frame_data.get("town_name", "") or ""
        token = frame_data.get("folder", None) or frame_data.get("scene_token", None)
        frame_idx = frame_data.get("frame_idx", None)
        world2lidar = None
        if token is not None and frame_idx is not None:
            info = info_index.get((str(token), int(frame_idx)), None)
            if isinstance(info, dict):
                world2lidar = _world2lidar_from_b2d_info(info)
                if not town_name:
                    town_name = info.get("town_name", "") or ""
        if town_name and town_name in map_infos and world2lidar is not None:
            # Make map visually stronger in pretty mode
            _draw_map_polylines_bev(
                ax,
                map_infos[town_name],
                world2lidar,
                xlim=(-50, 50),
                ylim=(-50, 50),
                alpha=0.75 if is_pretty else 0.55,
                pretty=is_pretty,
                draw_trigger_volumes=(map_draw_trigger_volumes if is_pretty else True),
            )

    # Optional: draw surrounding agents' GT boxes
    if draw_boxes and info_index is not None:
        # Prefer folder (B2D infos key), fallback to scene_token (baseline JSON uses this)
        token = frame_data.get("folder", None) or frame_data.get("scene_token", None)
        frame_idx = frame_data.get("frame_idx", None)
        if token is not None and frame_idx is not None:
            info = info_index.get((str(token), int(frame_idx)), None)
            _draw_gt_boxes(ax, info)

    # Predictions (6 commands)
    legend_items = []
    any_pred_drawn = False
    any_selected_drawn = False
    for cmd_idx in range(6):
        pred_traj = predictions[cmd_idx] if cmd_idx < len(predictions) else []
        if not pred_traj:
            continue

        pred = _prepend_origin_if_missing(pred_traj)
        is_selected = cmd_idx == ego_fut_cmd_idx
        lw = 3.2 if is_selected else 2.0
        alpha = 1.0 if is_selected else (0.75 if is_pretty else 0.55)
        ls = "-" if is_selected else ("-" if is_pretty else "--")

        label = CMD_LABELS[cmd_idx] if cmd_idx < len(CMD_LABELS) else "Cmd {}".format(cmd_idx)
        if is_selected:
            label = "{} (Selected)".format(label)

        c = CMD_COLORS[cmd_idx % len(CMD_COLORS)]
        if is_pretty:
            # Render as a point band; selected command more saturated & slightly larger
            score = 1.0 if is_selected else 0.85
            _scatter_traj_band(
                ax,
                pred[:, :2],
                color=c,
                score=score,
                dot_size=18 if is_selected else 12,
                points_per_step=18,
                zorder=7 if is_selected else 6,
                alpha=1.0 if is_selected else 0.95,
            )
            any_pred_drawn = True
            cmd_name = CMD_LABELS[cmd_idx] if cmd_idx < len(CMD_LABELS) else "Cmd {}".format(cmd_idx)
            if is_selected:
                cmd_name = "{} (Selected)".format(cmd_name)
                any_selected_drawn = True
            legend_items.append((f"{primary_name}: {cmd_name}", c))
        else:
            ax.plot(
                pred[:, 0],
                pred[:, 1],
                color=c,
                linewidth=lw,
                alpha=alpha,
                linestyle=ls,
                marker="o",
                markersize=4,
                label=label,
                zorder=5,
            )
            ax.plot(
                pred[-1, 0],
                pred[-1, 1],
                color=c,
                marker="s",
                markersize=8,
                alpha=alpha,
                zorder=6,
            )

    # Compare predictions (2nd JSON) overlay
    if compare_frame_data is not None:
        for cmd_idx in range(6):
            pred_traj_b = predictions_b[cmd_idx] if cmd_idx < len(predictions_b) else []
            if not pred_traj_b:
                continue
            pred_b = _prepend_origin_if_missing(pred_traj_b)
            is_selected_b = cmd_idx == ego_fut_cmd_idx_b
            c = CMD_COLORS[cmd_idx % len(CMD_COLORS)]
            if is_pretty:
                _scatter_traj_band(
                    ax,
                    pred_b[:, :2],
                    color=c,
                    score=1.0 if is_selected_b else 0.80,
                    dot_size=14 if is_selected_b else 9,
                    points_per_step=18,
                    zorder=9 if is_selected_b else 8,
                    alpha=0.90 if is_selected_b else 0.70,
                )
                cmd_name = CMD_LABELS[cmd_idx] if cmd_idx < len(CMD_LABELS) else "Cmd {}".format(cmd_idx)
                if is_selected_b:
                    cmd_name = "{} (Selected)".format(cmd_name)
                legend_items.append((f"{compare_name}: {cmd_name}", c))
            else:
                ax.plot(
                    pred_b[:, 0],
                    pred_b[:, 1],
                    color=c,
                    linewidth=2.2 if is_selected_b else 1.6,
                    alpha=0.85 if is_selected_b else 0.45,
                    linestyle=":" if is_selected_b else "--",
                    zorder=4,
                )

    # Ground truth (single)
    if ground_truth:
        gt = _prepend_origin_if_missing(ground_truth)
        if is_pretty:
            _scatter_traj_band(ax, gt[:, :2], color="#111111", score=1.0, dot_size=18, points_per_step=18, zorder=9, alpha=1.0)
            legend_items.append(("GT", "#111111"))
        else:
            ax.plot(
                gt[:, 0],
                gt[:, 1],
                color="black",
                linewidth=3,
                linestyle="-",
                marker="x",
                markersize=6,
                label="Ground Truth",
                zorder=9,
            )
            ax.plot(gt[-1, 0], gt[-1, 1], color="black", marker="D", markersize=8, zorder=9)

    # Metrics text (optional)
    metrics = []
    for k, name in [
        ("plan_L2_1s", "L2 (1s)"),
        ("plan_L2_2s", "L2 (2s)"),
        ("plan_L2_3s", "L2 (3s)"),
        ("fut_valid_flag", "Valid"),
    ]:
        if k in frame_data:
            metrics.append("{}: {}".format(name, frame_data[k]))
    if metrics:
        ax.text(
            0.02 if not is_pretty else 0.02,
            0.98 if not is_pretty else 0.98,
            "\n".join(metrics),
            transform=ax.transAxes,
            fontsize=11 if not is_pretty else 32,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    if not is_pretty:
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.25)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    if draw_title and (not is_pretty):
        scene_token = frame_data.get("scene_token", "")
        frame_idx = frame_data.get("frame_idx", frame_global_idx)
        title = "Frame {} | ego_fut_cmd_idx={} | {}".format(frame_idx, ego_fut_cmd_idx, scene_token)
        ax.set_title(title, fontsize=11)

    # Legend is mandatory.
    if not is_pretty:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    else:
        # Build a small, clean legend for pretty mode
        from matplotlib.lines import Line2D

        handles = []
        labels = []
        seen = set()
        for name, col in legend_items:
            if name in seen:
                continue
            seen.add(name)
            handles.append(Line2D([0], [0], color=col, linewidth=6))
            labels.append(name)
        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=32, framealpha=0.88)

    # Fixed scale across frames
    if is_pretty:
        # MomAD uses a fixed local window; using global limits makes the road look tiny.
        pr = float(pretty_range) if pretty_range else 40.0
        _apply_pretty_axes(ax, (-pr, pr), (-pr, pr))
    else:
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    scene_token = _safe_name(frame_data.get("scene_token", "scene"))
    frame_idx = int(frame_data.get("frame_idx", frame_global_idx))
    out_ext = "jpg" if is_pretty else "png"
    out_path = os.path.join(output_dir, "{}_frame_{:05d}.{}".format(scene_token, frame_idx, out_ext))
    if is_pretty:
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(out_path, dpi=150)
    else:
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    repo_root = Path(__file__).resolve().parent
    b2d_root = repo_root / "Bench2DriveZoo"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(b2d_root / "data/infos/b2d_repair_collect_tiny_traj.json"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(repo_root / "trajectory_visualizations_6cmd"),
    )
    parser.add_argument(
        "--style",
        default="momad",
        choices=["default", "momad", "pretty", "clean"],
        help="Visualization style. Default momad: large canvas, axis-off, point-band trajectories (简洁). "
        "Use default for legacy matplotlib axes/grid style.",
    )
    parser.add_argument(
        "--compare_input",
        default="",
        help="Optional: 2nd open-loop JSON to overlay (e.g. repaired model). Matched by (scene_token, frame_idx).",
    )
    parser.add_argument("--primary_name", default="Base", help="Legend prefix for primary input (default: Base).")
    parser.add_argument("--compare_name", default="Repaired", help="Legend prefix for compare_input (default: Repaired).")
    parser.add_argument(
        "--pretty_range",
        type=float,
        default=40.0,
        help="When --style momad/pretty/clean: use fixed BEV window [-R,R] meters (default 40).",
    )
    parser.add_argument(
        "--map_draw_trigger_volumes",
        action="store_true",
        help="When --style momad/pretty/clean and --draw_map: also draw trigger volume polylines (default off).",
    )
    parser.add_argument(
        "--occ_root",
        default="",
        help="Optional: root dir for VAD occ cache (e.g. baseline/VAD/vad_occ_cache). Used with --draw_occ.",
    )
    parser.add_argument(
        "--draw_occ",
        action="store_true",
        help="Draw VAD occ cache as BEV background (dynamic obstacles/ped occupancy, not road curbs).",
    )
    parser.add_argument(
        "--occ_reduce",
        default="max",
        choices=["max", "t0"],
        help="How to reduce occ over time dimension: max (default) or t0.",
    )
    parser.add_argument(
        "--b2d_infos_pkl",
        default="",
        help="Optional: path to b2d_infos_*.pkl to draw gt_boxes/gt_names by matching (scene_token, frame_idx).",
    )
    parser.add_argument(
        "--draw_gt_boxes",
        action="store_true",
        help="Draw GT boxes from b2d_infos_pkl (matches scene_token+frame_idx).",
    )
    parser.add_argument(
        "--map_infos_pkl",
        default="",
        help="Optional: path to b2d_map_infos.pkl for map polylines (world coords).",
    )
    parser.add_argument(
        "--draw_map",
        action="store_true",
        help="Draw map polylines in LiDAR BEV using world2lidar (LIDAR_TOP) from b2d_infos_pkl, same as B2D_vad_dataset.",
    )
    parser.add_argument(
        "--ego_icon",
        default="",
        help="Optional: path to an ego car icon image (png/jpg). If empty, draw a vector car silhouette.",
    )
    parser.add_argument(
        "--ego_anchor",
        default="center",
        choices=["center", "front", "rear"],
        help="Where to place (0,0) on the ego car: center (default), front (front-center), or rear (rear-center).",
    )
    parser.add_argument(
        "--ego_forward_axis",
        default="y",
        choices=["x", "y"],
        help="Which plot axis is considered 'forward' for ego icon placement. Default y (up).",
    )
    parser.add_argument(
        "--only_scene_token",
        default="",
        help="Optional: only visualize frames with this exact scene_token (or folder), e.g. v1/SignalizedJunctionLeftTurn_Town04_Route173_Weather26.",
    )
    parser.add_argument(
        "--only_frame_idx",
        type=int,
        default=None,
        help="Optional: only visualize frames with this frame_idx (use with --only_scene_token for a single frame).",
    )
    args = parser.parse_args()

    print("Loading: {}".format(args.input))
    with open(args.input, "r") as f:
        data = json.load(f)

    compare_index = None
    if args.compare_input:
        print("Loading compare_input: {}".format(args.compare_input))
        with open(args.compare_input, "r") as f:
            data_b = json.load(f)
        compare_index = {}
        for it in data_b:
            if not isinstance(it, dict):
                continue
            tok = it.get("scene_token", None) or it.get("folder", None)
            fi = it.get("frame_idx", None)
            if tok is None or fi is None:
                continue
            compare_index[(str(tok), int(fi))] = it
        print("Built compare index: {} entries".format(len(compare_index)))

    only_tok = str(args.only_scene_token or "").strip()
    only_fi = args.only_frame_idx
    filtered = []
    for orig_i, fr in enumerate(data):
        if not isinstance(fr, dict):
            continue
        if only_tok:
            tok = str(fr.get("scene_token", None) or fr.get("folder", None) or "")
            if tok != only_tok:
                continue
        if only_fi is not None:
            if int(fr.get("frame_idx", -999999)) != int(only_fi):
                continue
        filtered.append((orig_i, fr))

    if only_tok or only_fi is not None:
        if not filtered:
            print("ERROR: no frames match --only_scene_token / --only_frame_idx filter; exit.")
            return
        print(
            "Filter active: {} frames (only_scene_token={!r}, only_frame_idx={})".format(
                len(filtered), only_tok or None, only_fi
            )
        )

    n = len(data)
    print("Loaded {} frames".format(n))
    print("Saving to: {}".format(args.output_dir))

    info_index = None
    if args.draw_gt_boxes:
        if not args.b2d_infos_pkl:
            print("WARNING: --draw_gt_boxes set but --b2d_infos_pkl is empty; skip.")
        else:
            print("Loading b2d infos pkl for gt_boxes: {}".format(args.b2d_infos_pkl))
            infos = _load_infos_pkl(args.b2d_infos_pkl)
            info_index = _build_info_index(infos)
            print("Built info index (gt_boxes): {} entries".format(len(info_index)))

    map_infos = None
    if args.draw_map:
        if not args.map_infos_pkl:
            print("WARNING: --draw_map set but --map_infos_pkl is empty; skip map polylines.")
        else:
            print("Loading map infos pkl: {}".format(args.map_infos_pkl))
            map_infos = _load_map_infos_pkl(args.map_infos_pkl)
            print("Loaded map infos: {} towns".format(len(map_infos)))

    # Compute global limits once so all frames share the same scale/view.
    # If we draw GT boxes, include them in view limits too (otherwise boxes may fall outside).
    limit_frames = [fr for _, fr in filtered] if filtered else data
    xlim, ylim = compute_global_limits(
        limit_frames,
        info_index=info_index,
        include_boxes=bool(args.draw_gt_boxes),
        min_range=20.0,
    )
    print("Using fixed limits: xlim={} ylim={}".format(xlim, ylim))

    count = 0
    loop_items = filtered if filtered else [(i, data[i]) for i in range(n)]
    for i, frame in loop_items:

        frame_b = None
        if compare_index is not None:
            tok = frame.get("scene_token", None) or frame.get("folder", None)
            fi = frame.get("frame_idx", None)
            if tok is not None and fi is not None:
                frame_b = compare_index.get((str(tok), int(fi)), None)

        visualize_one_frame(
            frame,
            i,
            args.output_dir,
            xlim=xlim,
            ylim=ylim,
            info_index=info_index,
            draw_boxes=bool(args.draw_gt_boxes),
            occ_root=args.occ_root,
            draw_occ=args.draw_occ,
            occ_reduce=args.occ_reduce,
            map_infos=map_infos,
            draw_map=args.draw_map,
            style=args.style,
            pretty_range=args.pretty_range,
            map_draw_trigger_volumes=args.map_draw_trigger_volumes,
            compare_frame_data=frame_b,
            primary_name=args.primary_name,
            compare_name=args.compare_name,
            ego_icon=args.ego_icon,
            ego_anchor=args.ego_anchor,
            ego_forward_axis=args.ego_forward_axis,
        )
        count += 1
        if count == 1 or count % 50 == 0:
            print("  visualized {} (idx={})".format(count, i))

    print("Done. Total visualized frames: {}".format(count))


if __name__ == "__main__":
    main()
