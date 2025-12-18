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


def _safe_name(s):
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:180]


def compute_global_limits(frames, only_valid=False, pad_ratio=0.08, min_range=5.0):
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


def visualize_one_frame(frame_data, frame_global_idx, output_dir, draw_title=True, xlim=None, ylim=None):
    predictions = frame_data.get("predictions", [])
    ground_truth = frame_data.get("ground_truth", [])
    ego_fut_cmd_idx = frame_data.get("ego_fut_cmd_idx", -1)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Ego current position at origin (dataset is in ego frame)
    ax.plot(0, 0, "ko", markersize=10, label="Ego (Current)", zorder=10)

    # Predictions (6 commands)
    for cmd_idx in range(6):
        pred_traj = predictions[cmd_idx] if cmd_idx < len(predictions) else []
        if not pred_traj:
            continue

        pred = np.asarray(pred_traj, dtype=float)
        is_selected = cmd_idx == ego_fut_cmd_idx
        lw = 3.0 if is_selected else 1.8
        alpha = 1.0 if is_selected else 0.55
        ls = "-" if is_selected else "--"

        label = CMD_LABELS[cmd_idx] if cmd_idx < len(CMD_LABELS) else "Cmd {}".format(cmd_idx)
        if is_selected:
            label = "{} (Selected)".format(label)

        ax.plot(
            pred[:, 0],
            pred[:, 1],
            color=CMD_COLORS[cmd_idx % len(CMD_COLORS)],
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
            color=CMD_COLORS[cmd_idx % len(CMD_COLORS)],
            marker="s",
            markersize=8,
            alpha=alpha,
            zorder=6,
        )

    # Ground truth (single)
    if ground_truth:
        gt = np.asarray(ground_truth, dtype=float)
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
            0.02,
            0.98,
            "\n".join(metrics),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.25)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    if draw_title:
        scene_token = frame_data.get("scene_token", "")
        frame_idx = frame_data.get("frame_idx", frame_global_idx)
        title = "Frame {} | ego_fut_cmd_idx={} | {}".format(frame_idx, ego_fut_cmd_idx, scene_token)
        ax.set_title(title, fontsize=11)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Fixed scale across frames
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    scene_token = _safe_name(frame_data.get("scene_token", "scene"))
    frame_idx = int(frame_data.get("frame_idx", frame_global_idx))
    out_path = os.path.join(output_dir, "{}_frame_{:05d}.png".format(scene_token, frame_idx))
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
    parser.add_argument("--only_valid", action="store_true", help="only frames with fut_valid_flag==True")
    args = parser.parse_args()

    print("Loading: {}".format(args.input))
    with open(args.input, "r") as f:
        data = json.load(f)

    n = len(data)
    print("Loaded {} frames".format(n))
    print("Saving to: {}".format(args.output_dir))

    # Compute global limits once so all frames share the same scale/view.
    xlim, ylim = compute_global_limits(data, only_valid=args.only_valid)
    print("Using fixed limits: xlim={} ylim={}".format(xlim, ylim))

    count = 0
    for i in range(n):
        frame = data[i]
        if args.only_valid and not frame.get("fut_valid_flag", False):
            continue

        visualize_one_frame(frame, i, args.output_dir, xlim=xlim, ylim=ylim)
        count += 1
        if count == 1 or count % 50 == 0:
            print("  visualized {} (idx={})".format(count, i))

    print("Done. Total visualized frames: {}".format(count))


if __name__ == "__main__":
    main()
