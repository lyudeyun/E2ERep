#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize predicted trajectories (e.g. VAD: six per-command trajectories, UniAD: one trajectory) and GT per frame.

Run from the repository root (the directory that contains ``mytools/`` and ``Bench2DriveZoo/``).

Input JSON format (list of frames) is expected to contain:
  - predictions: list of trajectories; length 6 (VAD-style) or 1 (UniAD-style), each (T,2)
  - ground_truth: (T,2) trajectory
  - ego_fut_cmd_idx: int in [0..5] indicating the selected motion command when there are six per-command trajectories (optional)

Example (minimal — ``--model`` is required; omit ``--output_dir`` to auto-name under repo root from ``len(predictions)``):

  conda run -n b2d_zoo python mytools/visualize_openloop.py --input Bench2DriveZoo/data/infos/b2d_repair_collect_tiny_traj.json --model vad --vad-pred-draw all

Example (VAD baseline + ``--compare_input`` UniAD JSON, ``--vad-pred-draw selected``, one frame; run from repository root).
Below, ``\\`` at line end is one backslash in this file (bash line continuation)::

  python mytools/visualize_openloop.py \\
    --input baseline/VAD/vad_base_baseline_b2d_infos_val_partB_25clips.json \\
    --compare_input /home/deyun/git/B2DRepair/B2DRepair_Data/vad_base_Arachne_v2_DE_results/middle/VAD_base_REP_VAL_t3s_Arachne_v2_DE_w52_p104_i50_es5_CONT2_9/open_loop_eval/vad_base_rep_val.json \\
    --model vad \\
    --vad-pred-draw selected \\
    --scene \\
    --output_dir vad_base_cmp_ori_rep \\
    --view-forward-center-offset-m 10 \\
    --only_scene_token 'v1/OppositeVehicleTakingPriority_Town04_Route214_Weather6' \\
    --only_frame_idx 20
  
  python mytools/visualize_openloop.py \\
    --input baseline/UniAD/uniad_base_baseline_b2d_infos_val_partB_25clips.json \\
    --compare_input /home/deyun/git/B2DRepair/B2DRepair_Data/uniad_base_Arachne_v2_DE_results/large/UniAD_base_REP_VAL_t3s_Arachne_v2_DE_w26_p52_i50_es5_CONT2_7/open_loop_eval/uniad_base_rep_val.json \\
    --model uniad \\
    --scene \\
    --output_dir uniad_base_cmp_ori_rep \\
    --view-forward-center-offset-m 10 \\
    --only_scene_token 'v1/OppositeVehicleTakingPriority_Town04_Route214_Weather6' \\
    --only_frame_idx 20

Example (UniAD only, no ``--compare_input``; no ``--vad-pred-draw``). Replace ``--input`` with your UniAD open-loop JSON path::

  python mytools/visualize_openloop.py \\
    --input B2DRepair_Data/uniad_base_Arachne_v2_DE_results/large/UniAD_base_REP_VAL_t3s_Arachne_v2_DE_w26_p52_i50_es5_CONT2_7/open_loop_eval/uniad_base_rep_val.json \\
    --model uniad \\
    --scene \\
    --output_dir viz_openloop_smoke \\
    --only_scene_token 'v1/OppositeVehicleTakingPriority_Town04_Route214_Weather6' \\
    --only_frame_idx 0

Example (VAD only, no ``--compare_input``; ``--vad-pred-draw`` is still required for ``--model vad``)::

  python mytools/visualize_openloop.py \\
    --input baseline/VAD/vad_base_baseline_b2d_infos_val_partB_25clips.json \\
    --model vad \\
    --vad-pred-draw all \\
    --scene \\
    --output_dir viz_openloop_smoke \\
    --view-forward-center-offset-m 10 \\
    --only_scene_token 'v1/OppositeVehicleTakingPriority_Town04_Route214_Weather6' \\
    --only_frame_idx 0

With lane map, other agents (GT boxes), and ``mytools/ego.png`` when data exists, add ``--scene`` (and optional ``--output_dir``).

Parameter meanings and defaults: each flag is documented in ``main()`` via
``parser.add_argument(..., help=...)`` (see around the ``ArgumentParser`` block), or run:

  python mytools/visualize_openloop.py --help
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
from matplotlib.lines import Line2D

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

# With --compare_input and a matched frame: primary preds vs compare preds (fixed colors, uniform width).
COMPARE_TRAJ_LW = 3.0
COMPARE_TRAJ_ORIG_COLOR = "#d62728"  # red
COMPARE_TRAJ_REP_COLOR = "#2ca02c"  # green


def _rgb_hex_blended_on_white(hex_color, alpha):
    """
    Opaque RGB that matches how ``ax.plot(..., color=hex, alpha=a)`` looks on a white axes background.
    Legend handles often draw fully opaque strokes; without this, swatches look darker than the plot line.
    """
    h = str(hex_color or "").strip().lstrip("#")
    if len(h) != 6 or not all(c in "0123456789abcdefABCDEF" for c in h):
        return str(hex_color)
    a = max(0.0, min(1.0, float(alpha)))
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r2 = int(round(r * a + 255.0 * (1.0 - a)))
    g2 = int(round(g * a + 255.0 * (1.0 - a)))
    b2 = int(round(b * a + 255.0 * (1.0 - a)))
    return "#{:02x}{:02x}{:02x}".format(r2, g2, b2)


FIG_SIZE = (16.0, 16.0)
VIZ_FONT_PT = 30.0
# Upper-left metrics box and upper-right legend (axes/ticks stay VIZ_FONT_PT).
CORNER_FONT_PT = 30.0
# Extra space above axes between title and plot (points); avoids overlap with top y tick labels.
TITLE_PAD_PT = 22.0
# Shift ego icon/silhouette toward -forward (m) so the nose does not cover the first pred segment; tune if needed.
EGO_SHIFT_BACK_M = 2.0


def _prepend_origin_if_missing(xy, eps=1e-3):
    arr = np.asarray(xy, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return arr
    if float(np.hypot(arr[0, 0], arr[0, 1])) <= float(eps):
        return arr
    zero = np.zeros((1, 2), dtype=float)
    return np.concatenate([zero, arr[:, :2]], axis=0)


def _resolve_repo_path(repo_root: Path, path_str: str) -> str:
    """Absolute paths unchanged; relative paths are under ``repo_root`` (same rule as ``--output_dir``)."""
    p = (path_str or "").strip()
    if not p:
        return ""
    pt = Path(p).expanduser()
    if pt.is_absolute():
        return str(pt)
    return str((repo_root / pt).resolve())


def _default_b2d_infos_pkl_path(repo_root: Path) -> str:
    for rel in (
        "Bench2DriveZoo/data/infos/b2d_infos_val_partB_25clips.pkl",
        "Bench2DriveZoo/data/infos/b2d_infos_val_partA_25clips.pkl",
        "Bench2DriveZoo/data/infos/b2d_infos_val.pkl",
    ):
        cand = repo_root / rel
        if cand.is_file():
            return str(cand.resolve())
    return ""


def _default_map_infos_pkl_path(repo_root: Path) -> str:
    cand = repo_root / "Bench2DriveZoo/data/infos/b2d_map_infos.pkl"
    return str(cand.resolve()) if cand.is_file() else ""


def _default_ego_icon_path() -> str:
    """``mytools/ego.png`` or ``mytools/ego.jpg`` next to this script."""
    mytools = Path(__file__).resolve().parent
    for name in ("ego.png", "ego.jpg"):
        cand = mytools / name
        if cand.is_file():
            return str(cand.resolve())
    return ""


def _bev_axis_limits(view_half, forward_axis, forward_center_offset_m=0.0):
    """
    Build (xlim, ylim) for BEV. Lateral and forward each span ``2 * view_half`` (same scale).

    ``forward_center_offset_m`` slides the forward-axis limits along +forward while ego stays at 0:
    forward range is ``[-view_half + offset, view_half + offset]``.
    Example: ``view_half=40``, ``offset=+30`` → 70 m ahead, 10 m behind on that axis.
    ``offset=0`` → symmetric ±view_half (same as the old balance=0.5 behavior).
    """
    vh = float(view_half)
    off = float(forward_center_offset_m)
    fa = str(forward_axis or "y").lower()
    lo = -vh + off
    hi = vh + off
    if fa == "x":
        return (lo, hi), (-vh, vh)
    return (-vh, vh), (lo, hi)


def _safe_name(s):
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:180]


def _strip_scene_version_prefix_for_display(s):
    """Remove leading ``v1/``, ``v2/``, … from scene_token for titles only (paths / keys unchanged)."""
    return re.sub(r"^[vV][0-9]+/", "", str(s or ""))


MODEL_CHOICES = frozenset(("uniad", "vad"))


def _normalize_model(s):
    m = str(s or "").strip().lower()
    if m not in MODEL_CHOICES:
        raise ValueError(
            "model must be one of {{{}}} (pass --model)".format(", ".join(sorted(MODEL_CHOICES)))
        )
    return m


def _recompute_vad_l2_metrics(frame, model_norm):
    """与 ``baseline/convert_uniad_to_vad_metrics.py::compute_vad_l2`` 相同；VAD 多分支用 ``ego_fut_cmd_idx``。"""
    if not isinstance(frame, dict):
        return None
    preds = frame.get("predictions") or []
    gt = frame.get("ground_truth")
    mask = frame.get("ego_fut_masks")
    if not preds or gt is None or mask is None:
        return None
    if model_norm == "vad" and len(preds) > 1:
        try:
            ei = int(frame.get("ego_fut_cmd_idx", -1))
        except (TypeError, ValueError):
            ei = -1
        pred_src = preds[ei] if 0 <= ei < len(preds) else preds[0]
    else:
        pred_src = preds[0]

    pred_traj = np.asarray(pred_src, dtype=np.float32)[:6, :2]
    gt_traj = np.asarray(gt, dtype=np.float32)
    if gt_traj.ndim == 3:
        gt_traj = gt_traj[0, :6, :2]
    else:
        gt_traj = gt_traj[:6, :2]
    gt_mask = np.asarray(mask, dtype=np.float32)
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[0, :6, :2]
    elif gt_mask.ndim == 2:
        gt_mask = gt_mask[:6, :2] if gt_mask.shape[1] >= 2 else gt_mask[:6, None]
    elif gt_mask.ndim == 1:
        gt_mask = gt_mask[:6, None]
    else:
        return None

    pred_traj[:, 0] = -pred_traj[:, 0]
    gt_traj[:, 0] = -gt_traj[:, 0]
    l2 = np.sqrt((((pred_traj - gt_traj) ** 2) * gt_mask).sum(axis=-1))
    return {
        "plan_L2_1s": float(np.mean(l2[:2])),
        "plan_L2_2s": float(np.mean(l2[:4])),
        "plan_L2_3s": float(np.mean(l2[:6])),
    }


def _pred_cmd_indices(model_norm, vad_pred_draw, predictions, ego_fut_cmd_idx):
    """
    Which ``predictions[k]`` indices to plot. UniAD: always try 0..5 (only non-empty are drawn).
    VAD: ``vad_pred_draw`` is ``all`` or ``selected`` (``ego_fut_cmd_idx`` for the latter).
    """
    ei = int(ego_fut_cmd_idx) if ego_fut_cmd_idx is not None else -1
    if model_norm == "uniad":
        return list(range(6))
    if vad_pred_draw == "all":
        return list(range(6))
    if vad_pred_draw == "selected" and 0 <= ei < 6 and ei < len(predictions):
        return [ei]
    return list(range(6))


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
    Resolve relative paths like ego.png: absolute path, then cwd, then this script's directory (``mytools/``).
    """
    p = Path(icon_path).expanduser()
    if p.is_file():
        return str(p.resolve())
    bases = [Path.cwd(), Path(__file__).resolve().parent]
    for base in bases:
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


def _draw_ego_car(
    ax,
    icon_path="",
    yaw=0.0,
    zorder=10,
    anchor="center",
    forward_axis="y",
    along_shift_m=0.0,
):
    """
    Draw the ego car at origin in LiDAR BEV frame (x forward, y left).
    - If icon_path is provided, render it via imshow with a fixed meter extent.
    - Otherwise, draw a simple car silhouette (rectangle + nose).
    ``along_shift_m`` (m): move the drawn car toward -forward so traj near (0,0) stays visible (see ``EGO_SHIFT_BACK_M``).
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
        # Rearward along +y = decrease both y bounds by the same amount.
        sh = float(along_shift_m)
        extent[2] -= sh
        extent[3] -= sh
        target_aspect_wh = float(width) / float(length)
    else:
        if anchor == "front":
            ax0, ay0 = 0.5 * length, 0.0
        elif anchor == "rear":
            ax0, ay0 = -0.5 * length, 0.0
        else:
            ax0, ay0 = 0.0, 0.0
        extent = [-0.5 * length - ax0, 0.5 * length - ax0, -0.5 * width - ay0, 0.5 * width - ay0]
        sh = float(along_shift_m)
        extent[0] -= sh
        extent[1] -= sh
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
    sh = float(along_shift_m)
    if sh != 0.0:
        if forward_axis == "y":
            body_xy[:, 1] -= sh
            tri_xy[:, 1] -= sh
        else:
            body_xy[:, 0] -= sh
            tri_xy[:, 0] -= sh

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
    Build index for quick lookup: (token, frame_idx) -> info dict.
    Keys include ``folder`` and ``scene_token`` (when present) so baseline JSON can match either field.
    """
    idx = {}
    for it in infos:
        if not isinstance(it, dict):
            continue
        frame_idx = it.get("frame_idx", None)
        if frame_idx is None:
            continue
        fi = int(frame_idx)
        folder = it.get("folder", None)
        if folder is not None:
            idx[(str(folder), fi)] = it
        st = it.get("scene_token", None)
        if st is not None:
            idx[(str(st), fi)] = it
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

    # When pretty=True, prefer sampled polylines only (sparser map).
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


def visualize_one_frame(
    frame_data,
    frame_global_idx,
    output_dir,
    draw_title=True,
    info_index=None,
    draw_boxes=False,
    occ_root="",
    draw_occ=False,
    occ_reduce="max",
    map_infos=None,
    draw_map=False,
    view_m=40.0,
    forward_center_offset_m=0.0,
    compare_frame_data=None,
    compare_legend=False,
    ego_icon="",
    ego_anchor="center",
    ego_forward_axis="y",
    model=None,
    vad_pred_draw=None,
):
    mnorm = _normalize_model(model)
    use_compact_compare_legend = bool(compare_legend) and compare_frame_data is not None
    predictions = frame_data.get("predictions", [])
    ground_truth = frame_data.get("ground_truth", [])
    ego_fut_cmd_idx = frame_data.get("ego_fut_cmd_idx", -1)
    predictions_b = (compare_frame_data or {}).get("predictions", []) if compare_frame_data else []
    ego_fut_cmd_idx_b = (compare_frame_data or {}).get("ego_fut_cmd_idx", -1) if compare_frame_data else -1

    view_half = float(view_m) if view_m else 40.0
    xlim, ylim = _bev_axis_limits(view_half, ego_forward_axis, forward_center_offset_m)
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Current pose origin: LiDAR BEV frame (same as VAD gt_boxes / ego_fut in b2d_infos & collect JSON).
    # zorder below pred lines so traj near origin is drawn on top; slight rear shift keeps the icon off the first segment.
    _draw_ego_car(
        ax,
        icon_path=ego_icon,
        yaw=0.0,
        zorder=3,
        anchor=ego_anchor,
        forward_axis=ego_forward_axis,
        along_shift_m=EGO_SHIFT_BACK_M,
    )
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
            _draw_map_polylines_bev(
                ax,
                map_infos[town_name],
                world2lidar,
                xlim=xlim,
                ylim=ylim,
                alpha=0.55,
                pretty=False,
                draw_trigger_volumes=True,
            )

    # Optional: draw surrounding agents' GT boxes
    if draw_boxes and info_index is not None:
        # Prefer folder (B2D infos key), fallback to scene_token (baseline JSON uses this)
        token = frame_data.get("folder", None) or frame_data.get("scene_token", None)
        frame_idx = frame_data.get("frame_idx", None)
        if token is not None and frame_idx is not None:
            info = info_index.get((str(token), int(frame_idx)), None)
            _draw_gt_boxes(ax, info)

    pred_cmd_indices = _pred_cmd_indices(mnorm, vad_pred_draw, predictions, ego_fut_cmd_idx)
    hide_selected_suffix = mnorm == "vad" and vad_pred_draw == "selected" and len(pred_cmd_indices) == 1

    # Predictions (VAD: six or one per --vad-pred-draw; UniAD: slots 0..5, typically only 0 filled)
    for cmd_idx in pred_cmd_indices:
        pred_traj = predictions[cmd_idx] if cmd_idx < len(predictions) else []
        if not pred_traj:
            continue

        pred = _prepend_origin_if_missing(pred_traj)
        is_selected = cmd_idx == ego_fut_cmd_idx
        if use_compact_compare_legend:
            lw = COMPARE_TRAJ_LW
            alpha = 1.0 if is_selected else 0.55
            ls = "-" if is_selected else "--"
            c = COMPARE_TRAJ_ORIG_COLOR
        else:
            lw = 3.2 if is_selected else 2.0
            alpha = 1.0 if is_selected else 0.55
            ls = "-" if is_selected else "--"
            c = CMD_COLORS[cmd_idx % len(CMD_COLORS)]

        label = CMD_LABELS[cmd_idx] if cmd_idx < len(CMD_LABELS) else "Cmd {}".format(cmd_idx)
        if is_selected and not hide_selected_suffix:
            label = "{} (Selected)".format(label)
        if use_compact_compare_legend:
            label = "_nolegend_"
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
        compare_cmd_indices = _pred_cmd_indices(mnorm, vad_pred_draw, predictions_b, ego_fut_cmd_idx_b)
        for cmd_idx in compare_cmd_indices:
            pred_traj_b = predictions_b[cmd_idx] if cmd_idx < len(predictions_b) else []
            if not pred_traj_b:
                continue
            pred_b = _prepend_origin_if_missing(pred_traj_b)
            is_selected_b = cmd_idx == ego_fut_cmd_idx_b
            if use_compact_compare_legend:
                c = COMPARE_TRAJ_REP_COLOR
                lw_b = COMPARE_TRAJ_LW
            else:
                c = CMD_COLORS[cmd_idx % len(CMD_COLORS)]
                lw_b = 2.2 if is_selected_b else 1.6
            ax.plot(
                pred_b[:, 0],
                pred_b[:, 1],
                color=c,
                linewidth=2.2 if is_selected_b else 1.6,
                alpha=0.85 if is_selected_b else 0.45,
                linestyle=":" if is_selected_b else "--",
                zorder=4,
            )

    # Ground truth (single): only draw when fut_valid_flag is true (per-frame validity in B2D collect / baseline JSON).
    if ground_truth and bool(frame_data.get("fut_valid_flag", False)):
        gt = _prepend_origin_if_missing(ground_truth)
        ax.plot(
            gt[:, 0],
            gt[:, 1],
            color="black",
            linewidth=COMPARE_TRAJ_LW if use_compact_compare_legend else 3,
            linestyle="-",
            marker="x",
            markersize=6,
            label=("_nolegend_" if use_compact_compare_legend else "Ground Truth"),
            zorder=9,
        )
        ax.plot(gt[-1, 0], gt[-1, 1], color="black", marker="D", markersize=8, zorder=9)

    # Metrics：``--model uniad`` 且本帧 ``fut_valid_flag`` 为 True 时，用 VAD 口径从轨迹重算 L2（与 convert_uniad_to_vad_metrics 一致）；
    # ``--model vad`` 始终 VAD 口径；UniAD 且 valid=False 时只读 JSON。
    metrics = []

    def _l2_dict_for_corner(fr):
        if not isinstance(fr, dict):
            return None, "none"
        if mnorm == "uniad" and not bool(fr.get("fut_valid_flag", False)):
            return None, "uniad_file"
        if mnorm == "uniad":
            v = _recompute_vad_l2_metrics(fr, "uniad")
            if v is not None:
                return v, "vad_rule"
            return None, "uniad_recalc_fail"
        v = _recompute_vad_l2_metrics(fr, mnorm)
        if v is not None:
            return v, "vad_rule"
        return None, "vad_file"

    def _fmt_l2_val(v):
        if v is None:
            return "n/a"
        try:
            return "{:.4f}".format(float(v))
        except (TypeError, ValueError):
            return str(v)

    def _l2_error_3s_line(title, fr):
        if not isinstance(fr, dict):
            return "{}: n/a".format(title)
        d, tag = _l2_dict_for_corner(fr)
        j = fr.get("plan_L2_3s")
        if tag == "vad_rule" and d is not None:
            return "{}: {}".format(title, _fmt_l2_val(d["plan_L2_3s"]))
        if j is not None:
            return "{}: {}".format(title, _fmt_l2_val(j))
        return "{}: n/a".format(title)

    if use_compact_compare_legend:
        metrics.append(_l2_error_3s_line("L2 error (3s) original", frame_data))
        metrics.append(_l2_error_3s_line("L2 error (3s) repaired", compare_frame_data))
    else:
        d, tag = _l2_dict_for_corner(frame_data)
        for k, label in (
            ("plan_L2_1s", "L2 (1s)"),
            ("plan_L2_2s", "L2 (2s)"),
            ("plan_L2_3s", "L2 (3s)"),
        ):
            j = frame_data.get(k)
            if tag == "vad_rule" and d is not None:
                metrics.append("{}: {}".format(label, _fmt_l2_val(d[k])))
            elif j is not None:
                metrics.append("{}: {}".format(label, _fmt_l2_val(j)))
            else:
                metrics.append("{}: n/a".format(label))
        if "fut_valid_flag" in frame_data:
            metrics.append("Valid: {}".format(frame_data["fut_valid_flag"]))
    if metrics:
        ax.text(
            0.02,
            0.98,
            "\n".join(metrics),
            transform=ax.transAxes,
            fontsize=CORNER_FONT_PT,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.25)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("X (m)", fontsize=VIZ_FONT_PT)
    ax.set_ylabel("Y (m)", fontsize=VIZ_FONT_PT)
    ax.tick_params(axis="both", labelsize=VIZ_FONT_PT)

    if draw_title:
        scene_token = _strip_scene_version_prefix_for_display(frame_data.get("scene_token", ""))
        frame_idx = frame_data.get("frame_idx", frame_global_idx)
        title = "Frame {} | {}".format(frame_idx, scene_token)
        ax.set_title(title, fontsize=VIZ_FONT_PT, pad=TITLE_PAD_PT)

    if use_compact_compare_legend:
        eib = int(ego_fut_cmd_idx_b) if ego_fut_cmd_idx_b is not None else -1
        orig_c = COMPARE_TRAJ_ORIG_COLOR
        orig_ls = "-"
        orig_lw = COMPARE_TRAJ_LW
        orig_a = 1.0
        rep_c, rep_ls, rep_lw, rep_a = "#555555", "--", COMPARE_TRAJ_LW, 1.0
        if compare_frame_data is not None:
            for cmd_idx in _pred_cmd_indices(mnorm, vad_pred_draw, predictions_b, ego_fut_cmd_idx_b):
                pred_traj_b = predictions_b[cmd_idx] if cmd_idx < len(predictions_b) else []
                if not pred_traj_b:
                    continue
                sel_b = cmd_idx == eib
                rep_c = COMPARE_TRAJ_REP_COLOR
                rep_ls = ":" if sel_b else "--"
                rep_lw = COMPARE_TRAJ_LW
                rep_a = 0.85 if sel_b else 0.45
                break
        orig_leg_color = _rgb_hex_blended_on_white(orig_c, orig_a)
        rep_leg_color = _rgb_hex_blended_on_white(rep_c, rep_a)
        leg_handles = []
        if ground_truth and bool(frame_data.get("fut_valid_flag", False)):
            leg_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle="-",
                    linewidth=COMPARE_TRAJ_LW,
                    marker="x",
                    markersize=9,
                    label="Ground Truth",
                )
            )
        leg_handles.append(
            Line2D(
                [0],
                [0],
                color=orig_leg_color,
                linestyle=orig_ls,
                linewidth=orig_lw,
                label="Original",
            ),
        )
        leg_handles.append(
            Line2D(
                [0],
                [0],
                color=rep_leg_color,
                linestyle=rep_ls,
                linewidth=rep_lw,
                label="Repaired",
            ),
        )
        ax.legend(
            handles=leg_handles,
            loc="upper right",
            fontsize=CORNER_FONT_PT,
            framealpha=0.9,
        )
    else:
        ax.legend(loc="upper right", fontsize=CORNER_FONT_PT, framealpha=0.9)

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    scene_token = _safe_name(frame_data.get("scene_token", "scene"))
    frame_idx = int(frame_data.get("frame_idx", frame_global_idx))
    out_path = os.path.join(output_dir, "{}_frame_{:05d}.png".format(scene_token, frame_idx))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def infer_default_output_dir(frames, repo_root: Path) -> Path:
    """Pick ``trajectory_visualizations_*`` under repo root from the first non-empty predictions list."""
    n = 0
    for fr in frames:
        if not isinstance(fr, dict):
            continue
        preds = fr.get("predictions")
        if isinstance(preds, list) and len(preds) > 0:
            n = len(preds)
            break
    if n == 6:
        tag = "vad_6traj"
    elif n == 1:
        tag = "uniad_1traj"
    elif n > 0:
        tag = f"preds_{n}"
    else:
        tag = "openloop"
    return repo_root / f"trajectory_visualizations_{tag}"


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    b2d_root = repo_root / "Bench2DriveZoo"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(b2d_root / "data/infos/b2d_repair_collect_tiny_traj.json"),
        help="Open-loop JSON (list of frames): predictions + ground_truth per frame.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Output directory. If omitted or empty, auto: <repo>/trajectory_visualizations_vad_6traj "
        "(len(predictions)==6), ..._uniad_1traj (len==1), or ..._preds_<N>. "
        "Relative paths are resolved under the repository root (parent of mytools/); absolute paths unchanged.",
    )
    parser.add_argument(
        "--compare_input",
        default="",
        help="Optional: 2nd open-loop JSON to overlay (e.g. repaired model). Matched by (scene_token, frame_idx). "
        "When set and a frame matches: primary preds are drawn red, compare preds green (same linewidth as GT); "
        "legend shows Original / Ground Truth (if fut_valid_flag) / Repaired.",
    )
    parser.add_argument(
        "--view-m",
        type=float,
        default=15.0,
        metavar="R",
        help="BEV half-span (m): lateral axis is always [-R,R]; forward axis is [-R,+R] plus "
        "--view-forward-center-offset-m (total length still 2R). Used to clip map polylines when --draw_map.",
    )
    parser.add_argument(
        "--view-forward-center-offset-m",
        type=float,
        default=0.0,
        metavar="D",
        help="Slide the forward-axis window (see --ego-forward-axis) by D meters toward +forward: limits become "
        "[-R+D, R+D] so ego at 0 still sees more ahead and less behind when D>0. Example: R=40 and D=+30 → "
        "70 m ahead, 10 m behind. D=0 centers the window on the ego.",
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
        "--scene",
        action="store_true",
        help="Draw lane map + surrounding agent GT boxes. Uses --b2d_infos_pkl / --map_infos_pkl when set; "
        "otherwise tries Bench2DriveZoo/data/infos/ defaults if those files exist.",
    )
    parser.add_argument(
        "--b2d_infos_pkl",
        default="",
        help="b2d_infos_*.pkl: GT boxes, world2lidar for map. Relative paths are under repo root. "
        "If empty and --scene (or --draw_map / --draw_gt_boxes), a default partB/partA pkl is used when present.",
    )
    parser.add_argument(
        "--draw_gt_boxes",
        action="store_true",
        help="Draw GT boxes from b2d_infos_pkl (matches scene_token or folder + frame_idx).",
    )
    parser.add_argument(
        "--map_infos_pkl",
        default="",
        help="b2d_map_infos.pkl (world lane / trigger polylines). Relative paths are under repo root. "
        "If empty with --scene/--draw_map, uses Bench2DriveZoo/data/infos/b2d_map_infos.pkl when present.",
    )
    parser.add_argument(
        "--draw_map",
        action="store_true",
        help="Draw map polylines in LiDAR BEV (needs b2d_infos_pkl for world2lidar). Implied by --scene.",
    )
    parser.add_argument(
        "--ego_icon",
        default="",
        help="Ego car icon (png/jpg). Relative paths are under repo root. If empty, uses mytools/ego.png or mytools/ego.jpg when present; else vector silhouette.",
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
        "--model",
        required=True,
        choices=["uniad", "vad"],
        help="Required. uniad: single-trajectory JSON. vad: six-way JSON; use --vad-pred-draw to plot all six or only the selected command.",
    )
    parser.add_argument(
        "--vad-pred-draw",
        default=None,
        choices=["all", "selected"],
        metavar="MODE",
        help="Only for --model vad: draw all six motion priors (all) or only predictions[ego_fut_cmd_idx] (selected). "
        "Required with --model vad; must be omitted with --model uniad. Invalid ego_fut_cmd_idx falls back to all six.",
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

    if args.model == "vad":
        if args.vad_pred_draw is None:
            parser.error("--vad-pred-draw is required when --model vad (all or selected)")
    elif args.vad_pred_draw is not None:
        parser.error("--vad-pred-draw is only valid with --model vad")

    draw_map = bool(args.draw_map or args.scene)
    draw_gt_boxes = bool(args.draw_gt_boxes or args.scene)

    b2d_infos_pkl = _resolve_repo_path(repo_root, args.b2d_infos_pkl) if (args.b2d_infos_pkl or "").strip() else ""
    if (draw_map or draw_gt_boxes) and not b2d_infos_pkl:
        b2d_infos_pkl = _default_b2d_infos_pkl_path(repo_root)
        if b2d_infos_pkl:
            print("Using default b2d_infos_pkl: {}".format(b2d_infos_pkl))

    map_infos_pkl = _resolve_repo_path(repo_root, args.map_infos_pkl) if (args.map_infos_pkl or "").strip() else ""
    if draw_map and not map_infos_pkl:
        map_infos_pkl = _default_map_infos_pkl_path(repo_root)
        if map_infos_pkl:
            print("Using default map_infos_pkl: {}".format(map_infos_pkl))

    ego_raw = (args.ego_icon or "").strip()
    if ego_raw:
        ego_icon_final = _resolve_repo_path(repo_root, ego_raw)
    else:
        ego_icon_final = _default_ego_icon_path()
        if ego_icon_final:
            print("Using default ego icon: {}".format(ego_icon_final))

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
    out_raw = (args.output_dir or "").strip()
    if out_raw:
        p = Path(out_raw).expanduser()
        output_dir = p if p.is_absolute() else (repo_root / p)
    else:
        output_dir = infer_default_output_dir(data, repo_root)
        print("Auto output_dir from len(predictions): {}".format(output_dir))
    print("Saving to: {}".format(output_dir))

    info_index = None
    if b2d_infos_pkl and (draw_map or draw_gt_boxes):
        print("Loading b2d infos pkl: {}".format(b2d_infos_pkl))
        infos = _load_infos_pkl(b2d_infos_pkl)
        info_index = _build_info_index(infos)
        print("Built info index: {} keys".format(len(info_index)))
    elif draw_map or draw_gt_boxes:
        print(
            "WARNING: map / GT boxes requested but no b2d_infos_*.pkl found (--b2d_infos_pkl or default under Bench2DriveZoo/data/infos/)."
        )

    map_infos = None
    if draw_map:
        if not map_infos_pkl:
            print("WARNING: --draw_map / --scene but no b2d_map_infos.pkl; skip map polylines.")
        else:
            print("Loading map infos pkl: {}".format(map_infos_pkl))
            map_infos = _load_map_infos_pkl(map_infos_pkl)
            print("Loaded map infos: {} towns".format(len(map_infos)))

    vh = float(args.view_m)
    d = float(args.view_forward_center_offset_m)
    print(
        "BEV: lateral ±{:.1f} m; forward axis [{:.1f}, {:.1f}] m (--view-forward-center-offset-m={:+.1f})".format(
            vh,
            -vh + d,
            vh + d,
            d,
        )
    )
    mnorm = _normalize_model(args.model)
    print("Open-loop JSON: --model {}".format(mnorm))
    if mnorm == "vad":
        print("VAD: --vad-pred-draw {}".format(args.vad_pred_draw))

    count = 0
    loop_items = filtered if filtered else [(i, data[i]) for i in range(n)]
    compare_legend = bool((args.compare_input or "").strip())
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
            output_dir,
            info_index=info_index,
            draw_boxes=draw_gt_boxes,
            occ_root=args.occ_root,
            draw_occ=args.draw_occ,
            occ_reduce=args.occ_reduce,
            map_infos=map_infos,
            draw_map=draw_map,
            view_m=args.view_m,
            forward_center_offset_m=args.view_forward_center_offset_m,
            compare_frame_data=frame_b,
            compare_legend=compare_legend,
            ego_icon=ego_icon_final,
            ego_anchor=args.ego_anchor,
            ego_forward_axis=args.ego_forward_axis,
            model=mnorm,
            vad_pred_draw=args.vad_pred_draw,
        )
        count += 1
        if count == 1 or count % 50 == 0:
            print("  visualized {} (idx={})".format(count, i))

    print("Done. Total visualized frames: {}".format(count))


if __name__ == "__main__":
    main()
