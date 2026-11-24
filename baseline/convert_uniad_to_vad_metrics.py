#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def compute_vad_l2(item):
    preds = item.get("predictions")
    gt = item.get("ground_truth")
    mask = item.get("ego_fut_masks")
    if preds is None or gt is None or mask is None:
        return None

    pred_traj = np.asarray(preds[0], dtype=np.float32)[:6, :2]
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

    # UniAD planning metric flips x before L2
    pred_traj[:, 0] = -pred_traj[:, 0]
    gt_traj[:, 0] = -gt_traj[:, 0]

    # IMPORTANT: The sum must be INSIDE sqrt, not outside!
    # Correct: sqrt(sum(...)), NOT sqrt(...).sum()
    # This matches the correct formula used in _compute_l2_error_gpu
    l2 = np.sqrt((((pred_traj - gt_traj) ** 2) * gt_mask).sum(axis=-1))
    return {
        "plan_L2_1s": float(np.mean(l2[:2])),
        "plan_L2_2s": float(np.mean(l2[:4])),
        "plan_L2_3s": float(np.mean(l2[:6])),
    }, float(np.sum(gt_mask))


def build_scene_index(items):
    by_scene = defaultdict(list)
    for idx, item in enumerate(items):
        by_scene[item.get("scene_token")].append((idx, item))

    scene_maps = {}
    for scene, pairs in by_scene.items():
        pairs.sort(key=lambda x: x[1].get("frame_idx", 0))
        frame_to_pos = {}
        for pos, (_, it) in enumerate(pairs):
            frame_idx = it.get("frame_idx")
            if isinstance(frame_idx, int):
                frame_to_pos[frame_idx] = pos
        scene_maps[scene] = {
            "pairs": pairs,
            "frame_to_pos": frame_to_pos,
        }
    return scene_maps


def has_full_future(scene_map, item, sample_interval, future_steps):
    """Match dataset logic: use list index (temporal order), not frame_idx value."""
    pairs = scene_map["pairs"]
    try:
        pos = next(i for i, (_, it) in enumerate(pairs) if it is item)
    except StopIteration:
        return False
    for step in range(1, future_steps + 1):
        if pos + step * sample_interval >= len(pairs):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert UniAD JSON metrics to VAD-style L2 and valid-flag rule."
    )
    parser.add_argument("input_json", help="UniAD JSON from open-loop eval")
    parser.add_argument("output_json", help="Output JSON path")
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=5,
        help="Frame interval between future steps (default: 5 for 10Hz->2Hz)",
    )
    parser.add_argument(
        "--future-steps",
        type=int,
        default=6,
        help="Number of future steps (default: 6 for 3s at 2Hz)",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional path to write summary metrics (txt)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    scene_maps = build_scene_index(data)

    # Convert per-item fields
    for scene, scene_map in scene_maps.items():
        for _, item in scene_map["pairs"]:
            # Preserve original flag
            if "fut_valid_flag_uniad" not in item and "fut_valid_flag" in item:
                item["fut_valid_flag_uniad"] = item["fut_valid_flag"]

            full_future = has_full_future(
                scene_map, item, args.sample_interval, args.future_steps
            )
            result = compute_vad_l2(item)
            if result is None:
                # no trajectory to score: keep the UniAD L2 fields, frame is not valid
                mask_sum = 0.0
            else:
                vad_l2, mask_sum = result
                item.update(vad_l2)

            item["fut_valid_flag"] = bool(full_future and mask_sum > 0)

    # Summary stats
    keys = ["plan_L2_1s", "plan_L2_2s", "plan_L2_3s"]
    col_steps = {
        "0.5s": "plan_obj_col_0.5s",
        "1s": "plan_obj_col_1s",
        "1.5s": "plan_obj_col_1.5s",
        "2s": "plan_obj_col_2s",
        "2.5s": "plan_obj_col_2.5s",
        "3s": "plan_obj_col_3s",
    }
    box_steps = {
        "0.5s": "plan_obj_box_col_0.5s",
        "1s": "plan_obj_box_col_1s",
        "1.5s": "plan_obj_box_col_1.5s",
        "2s": "plan_obj_box_col_2s",
        "2.5s": "plan_obj_box_col_2.5s",
        "3s": "plan_obj_box_col_3s",
    }
    interval_map = {
        "1s": ["0.5s", "1s"],
        "2s": ["0.5s", "1s", "1.5s", "2s"],
        "3s": ["0.5s", "1s", "1.5s", "2s", "2.5s", "3s"],
    }
    # 只对 fut_valid_flag 为 True 的帧统计 L2 和 collision
    vals_all = {k: [] for k in keys}
    col_all = {k: [] for k in interval_map}
    box_all = {k: [] for k in interval_map}
    valid_count = 0
    for item in data:
        is_valid = bool(item.get("fut_valid_flag"))
        if not is_valid:
            continue
        valid_count += 1
        # L2 指标：仅统计有效帧
        for k in keys:
            v = item.get(k)
            if isinstance(v, (int, float)):
                vals_all[k].append(v)
        # Collision interval averages（仅统计有效帧，且丢弃负值）
        for horizon, steps in interval_map.items():
            step_vals = []
            box_vals = []
            for s in steps:
                cv = item.get(col_steps[s])
                bv = item.get(box_steps[s])
                # 无效 collision 通常为负数（如 -1），这里显式只保留非负值
                if isinstance(cv, (int, float)) and cv >= 0:
                    step_vals.append(cv)
                if isinstance(bv, (int, float)) and bv >= 0:
                    box_vals.append(bv)
            if step_vals:
                col_all[horizon].append(sum(step_vals) / len(step_vals))
            if box_vals:
                box_all[horizon].append(sum(box_vals) / len(box_vals))

    # 由于我们已经在聚合阶段只保留了有效帧，这里的 avg_all / avg_valid
    # 在数值上是相同的，但为了向后兼容，仍然保留两个 section。
    lines = [
        f"input_json: {input_path}",
        f"output_json: {output_path}",
        f"total_items: {len(data)}",
        f"valid_items (VAD rule): {valid_count}",
        "",
        "avg_all (VAD L2):",
    ]
    for k in keys:
        avg = sum(vals_all[k]) / len(vals_all[k]) if vals_all[k] else 0.0
        lines.append(f"  {k}: {avg} (n={len(vals_all[k])})")
    lines.append("")
    lines.append("avg_valid (VAD L2):")
    for k in keys:
        avg = sum(vals_all[k]) / len(vals_all[k]) if vals_all[k] else 0.0
        lines.append(f"  {k}: {avg} (n={len(vals_all[k])})")

    lines.append("")
    lines.append("avg_all (VAD collision interval):")
    for horizon in ["1s", "2s", "3s"]:
        avg = sum(col_all[horizon]) / len(col_all[horizon]) if col_all[horizon] else 0.0
        lines.append(f"  plan_obj_col_{horizon}: {avg} (n={len(col_all[horizon])})")
    for horizon in ["1s", "2s", "3s"]:
        avg = sum(box_all[horizon]) / len(box_all[horizon]) if box_all[horizon] else 0.0
        lines.append(f"  plan_obj_box_col_{horizon}: {avg} (n={len(box_all[horizon])})")

    lines.append("")
    lines.append("avg_valid (VAD collision interval):")
    for horizon in ["1s", "2s", "3s"]:
        avg = sum(col_all[horizon]) / len(col_all[horizon]) if col_all[horizon] else 0.0
        lines.append(f"  plan_obj_col_{horizon}: {avg} (n={len(col_all[horizon])})")
    for horizon in ["1s", "2s", "3s"]:
        avg = sum(box_all[horizon]) / len(box_all[horizon]) if box_all[horizon] else 0.0
        lines.append(f"  plan_obj_box_col_{horizon}: {avg} (n={len(box_all[horizon])})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("\n".join(lines), encoding="utf-8")
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
