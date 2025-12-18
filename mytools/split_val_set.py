#!/usr/bin/env python
"""
将 validation set 的 pkl 文件拆分成 repair set 和新的 validation set
拆分的最小单位是 clip，而不是帧
"""
import pickle
import argparse
import random
import os
from collections import defaultdict


def get_clips_from_pkl(pkl_file):
    """
    从 pkl 文件中提取所有 clips 及其帧索引
    
    Returns:
        clips_dict: {clip_id: [frame_indices]}
        clip_list: [(clip_id, frame_count, frame_indices)]
    """
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # 按 clip 分组（使用 folder 作为 clip 标识）
    clips_dict = defaultdict(list)
    for i, d in enumerate(data):
        clip_id = d.get('folder', '')
        clips_dict[clip_id].append(i)
    
    # 转换为列表，便于排序和选择
    clip_list = [(clip_id, len(indices), indices) 
                 for clip_id, indices in clips_dict.items()]
    clip_list.sort(key=lambda x: x[0])  # 按 clip_id 排序
    
    return clips_dict, clip_list


def split_by_clips(clip_list, mode='ratio', repair_ratio=0.5, 
                   repair_clip_count=None, repair_clip_ids=None, 
                   random_seed=42, shuffle=True):
    """
    按 clip 拆分数据
    
    Args:
        clip_list: [(clip_id, frame_count, frame_indices)]
        mode: 'ratio', 'count', 'manual'
        repair_ratio: repair set 的 clip 比例（mode='ratio' 时使用）
        repair_clip_count: repair set 的 clip 数量（mode='count' 时使用）
        repair_clip_ids: repair set 的 clip ID 列表（mode='manual' 时使用）
        random_seed: 随机种子
        shuffle: 是否随机打乱
    
    Returns:
        repair_clips: [(clip_id, frame_count, frame_indices)]
        val_clips: [(clip_id, frame_count, frame_indices)]
    """
    total_clips = len(clip_list)
    
    if mode == 'manual':
        # 手动选择模式
        if repair_clip_ids is None:
            raise ValueError("repair_clip_ids must be provided when mode='manual'")
        
        repair_clip_set = set(repair_clip_ids)
        repair_clips = [c for c in clip_list if c[0] in repair_clip_set]
        val_clips = [c for c in clip_list if c[0] not in repair_clip_set]
        
        if len(repair_clips) != len(repair_clip_set):
            missing = repair_clip_set - set(c[0] for c in repair_clips)
            print(f"Warning: Some clip IDs not found: {missing}")
    
    elif mode == 'count':
        # 按 clip 数量模式
        if repair_clip_count is None:
            raise ValueError("repair_clip_count must be provided when mode='count'")
        
        if repair_clip_count >= total_clips:
            raise ValueError(f"repair_clip_count ({repair_clip_count}) >= total clips ({total_clips})")
        
        if shuffle:
            random.seed(random_seed)
            clip_list_shuffled = clip_list.copy()
            random.shuffle(clip_list_shuffled)
        else:
            clip_list_shuffled = clip_list
        
        repair_clips = clip_list_shuffled[:repair_clip_count]
        val_clips = clip_list_shuffled[repair_clip_count:]
    
    else:  # mode == 'ratio'
        # 按比例模式
        repair_clip_count = int(total_clips * repair_ratio)
        
        if shuffle:
            random.seed(random_seed)
            clip_list_shuffled = clip_list.copy()
            random.shuffle(clip_list_shuffled)
        else:
            clip_list_shuffled = clip_list
        
        repair_clips = clip_list_shuffled[:repair_clip_count]
        val_clips = clip_list_shuffled[repair_clip_count:]
    
    return repair_clips, val_clips


def split_by_shortest_clips(clip_list, k=1):
    """
    Select the shortest K clips as repair set (deterministic).

    Args:
        clip_list: [(clip_id, frame_count, frame_indices)]
        k: number of shortest clips to select

    Returns:
        repair_clips, val_clips
    """
    if k is None:
        k = 1
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= len(clip_list):
        raise ValueError(f"k ({k}) >= total clips ({len(clip_list)})")

    # Sort by (frame_count asc, clip_id asc) for deterministic selection
    sorted_by_len = sorted(clip_list, key=lambda x: (x[1], x[0]))
    repair_clips = sorted_by_len[:k]
    repair_set = set(c[0] for c in repair_clips)
    val_clips = [c for c in clip_list if c[0] not in repair_set]
    return repair_clips, val_clips


def save_split_pkl(input_pkl, output_pkl, selected_clips):
    """
    保存选中的 clips 到新的 pkl 文件
    
    Args:
        input_pkl: 输入的 pkl 文件
        output_pkl: 输出的 pkl 文件
        selected_clips: [(clip_id, frame_count, frame_indices)]
    """
    with open(input_pkl, 'rb') as f:
        all_data = pickle.load(f)
    
    # 收集所有选中的帧索引
    selected_indices = []
    for clip_id, frame_count, frame_indices in selected_clips:
        selected_indices.extend(frame_indices)
    
    # 按原始顺序排序
    selected_indices.sort()
    
    # 提取对应的数据
    selected_data = [all_data[i] for i in selected_indices]
    
    # 保存
    os.makedirs(os.path.dirname(output_pkl) if os.path.dirname(output_pkl) else '.', exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(selected_data, f)
    
    return len(selected_data)


def list_clips(pkl_file):
    """列出所有 clips 的信息"""
    _, clip_list = get_clips_from_pkl(pkl_file)
    
    print(f"Total clips: {len(clip_list)}")
    print(f"\nClip information:")
    print(f"{'Index':<6} {'Clip ID':<60} {'Frames':<8}")
    print("-" * 80)
    for idx, (clip_id, frame_count, _) in enumerate(clip_list):
        print(f"{idx:<6} {clip_id:<60} {frame_count:<8}")
    
    total_frames = sum(frame_count for _, frame_count, _ in clip_list)
    print(f"\nTotal frames: {total_frames}")


def main():
    parser = argparse.ArgumentParser(
        description='Split validation pkl file into repair set and new validation set by clips',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all clips
  python split_val_set.py --list
  
  # Split by ratio (50% repair, 50% validation)
  python split_val_set.py --repair-ratio 0.5
  
  # Split by clip count (15 clips for repair)
  python split_val_set.py --mode count --repair-clip-count 15
  
  # Manual selection (specify clip IDs)
  python split_val_set.py --mode manual --repair-clip-ids "clip1,clip2,clip3"
  
  # Manual selection from file (one clip ID per line)
  python split_val_set.py --mode manual --repair-clip-file repair_clips.txt
        """
    )
    
    parser.add_argument('--input', type=str, default='data/infos/b2d_infos_val.pkl',
                     help='Input validation pkl file path')
    parser.add_argument('--output-repair', type=str, default='data/infos/b2d_infos_repair.pkl',
                     help='Output repair set pkl file path')
    parser.add_argument('--output-val', type=str, default='data/infos/b2d_infos_val_new.pkl',
                     help='Output new validation set pkl file path')
    parser.add_argument('--mode', type=str, choices=['ratio', 'count', 'manual'], default='ratio',
                     help='Split mode: ratio (by ratio), count (by clip count), manual (by clip IDs)')
    parser.add_argument('--shortest-k', type=int, default=None,
                     help='Select shortest K clips as repair set (enables shortest mode).')
    parser.add_argument('--repair-ratio', type=float, default=0.5,
                     help='Ratio of repair set clips (0-1, default: 0.5, used in ratio mode)')
    parser.add_argument('--repair-clip-count', type=int,
                     help='Number of clips for repair set (used in count mode)')
    parser.add_argument('--repair-clip-ids', type=str,
                     help='Comma-separated clip IDs for repair set (used in manual mode)')
    parser.add_argument('--repair-clip-file', type=str,
                     help='File containing clip IDs (one per line) for repair set (used in manual mode)')
    parser.add_argument('--seed', type=int, default=42,
                     help='Random seed for shuffling (default: 42)')
    parser.add_argument('--no-shuffle', action='store_true',
                     help='Do not shuffle, split sequentially (only for ratio/count modes)')
    parser.add_argument('--list', action='store_true',
                     help='List all clips and exit')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # 列出所有 clips
    if args.list:
        list_clips(args.input)
        return
    
    # 获取所有 clips
    print(f"Loading validation pkl file: {args.input}")
    clips_dict, clip_list = get_clips_from_pkl(args.input)
    total_clips = len(clip_list)
    total_frames = sum(frame_count for _, frame_count, _ in clip_list)
    print(f"Total clips: {total_clips}, Total frames: {total_frames}\n")
    
    # shortest-k overrides mode (for fast iteration)
    use_shortest = args.shortest_k is not None

    # 处理 manual 模式
    repair_clip_ids = None
    if (not use_shortest) and args.mode == 'manual':
        if args.repair_clip_file:
            with open(args.repair_clip_file, 'r') as f:
                repair_clip_ids = [line.strip() for line in f if line.strip()]
        elif args.repair_clip_ids:
            repair_clip_ids = [cid.strip() for cid in args.repair_clip_ids.split(',')]
        else:
            print("Error: --repair-clip-ids or --repair-clip-file must be provided in manual mode")
            return
    
    # 验证参数
    if (not use_shortest) and args.mode == 'ratio' and not (0 < args.repair_ratio < 1):
        print(f"Error: repair_ratio must be between 0 and 1, got {args.repair_ratio}")
        return
    
    if (not use_shortest) and args.mode == 'count' and args.repair_clip_count is None:
        print("Error: --repair-clip-count must be provided in count mode")
        return
    
    # 拆分 clips
    if use_shortest:
        repair_clips, val_clips = split_by_shortest_clips(clip_list, k=args.shortest_k)
    else:
        repair_clips, val_clips = split_by_clips(
            clip_list,
            mode=args.mode,
            repair_ratio=args.repair_ratio,
            repair_clip_count=args.repair_clip_count,
            repair_clip_ids=repair_clip_ids,
            random_seed=args.seed,
            shuffle=not args.no_shuffle
        )
    
    # 统计信息
    repair_clip_count = len(repair_clips)
    repair_frame_count = sum(frame_count for _, frame_count, _ in repair_clips)
    val_clip_count = len(val_clips)
    val_frame_count = sum(frame_count for _, frame_count, _ in val_clips)
    
    print(f"Split results:")
    print(f"  Repair set: {repair_clip_count} clips, {repair_frame_count} frames ({repair_frame_count/total_frames*100:.1f}%)")
    print(f"  New validation set: {val_clip_count} clips, {val_frame_count} frames ({val_frame_count/total_frames*100:.1f}%)")
    
    # 显示 repair set 的 clip IDs
    print(f"\nRepair set clip IDs:")
    for clip_id, frame_count, _ in repair_clips:
        print(f"  {clip_id} ({frame_count} frames)")
    
    # 保存文件
    print(f"\nSaving repair set to: {args.output_repair}")
    repair_saved_frames = save_split_pkl(args.input, args.output_repair, repair_clips)
    
    print(f"Saving new validation set to: {args.output_val}")
    val_saved_frames = save_split_pkl(args.input, args.output_val, val_clips)
    
    print("\nSplit completed successfully!")
    print(f"  - Repair set: {repair_clip_count} clips, {repair_saved_frames} frames")
    print(f"  - New validation set: {val_clip_count} clips, {val_saved_frames} frames")


if __name__ == '__main__':
    main()
