# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import argparse
import os
import json
import torch
import warnings
from mmcv.models.dense_heads.planning_head_plugin.metric_stp3 import PlanningMetric
import numpy as np
from mmcv.utils import get_dist_info, init_dist, wrap_fp16_model, set_random_seed, Config, DictAction, load_checkpoint
from mmcv.models import build_model, fuse_conv_bn
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from mmcv.datasets import build_dataset, build_dataloader, replace_ImageToTensor
import time
import os.path as osp
from adzoo.vad.apis.test import custom_multi_gpu_test, single_gpu_test

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--json_dir', help='json parent dir name file') # NOTE: json file parent folder name
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)
    # Data collection for repair (optional; no behavior change when disabled)
    parser.add_argument(
        '--collect-data',
        action='store_true',
        help='Collect per-frame planning metrics and intermediate info into a JSON file.')
    parser.add_argument(
        '--data-output',
        type=str,
        default=None,
        help='Output JSON path used with --collect-data (e.g., data/infos/repair_preprocessing.json).')
    parser.add_argument(
        '--occ-output-dir',
        type=str,
        default=None,
        help='Optional directory to save per-frame occupancy/segmentation for collision recomputation.')
    parser.add_argument(
        '--print-collected',
        type=int,
        default=0,
        help='Print first N collected items to stdout (only works with --collect-data).')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    # if hasattr(cfg, 'plugin'):
    #     if cfg.plugin:
    #         import importlib
    #         if hasattr(cfg, 'plugin_dir'):
    #             plugin_dir = cfg.plugin_dir
    #             _module_dir = os.path.dirname(plugin_dir)
    #             _module_dir = _module_dir.split('/')
    #             _module_path = _module_dir[0]

    #             for m in _module_dir[1:]:
    #                 _module_path = _module_path + '.' + m
    #             print(_module_path)
    #             plg_lib = importlib.import_module(_module_path)
    #         else:
    #             # import dir is the dirpath for the config file
    #             _module_dir = os.path.dirname(args.config)
    #             _module_dir = _module_dir.split('/')
    #             _module_path = _module_dir[0]
    #             for m in _module_dir[1:]:
    #                 _module_path = _module_path + '.' + m
    #             print(_module_path)
    #             plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # Load the checkpoint directly to the GPU if available
    if torch.cuda.is_available():
        map_location = f'cuda:{torch.cuda.current_device()}'
    else:
        map_location = 'cpu'
    checkpoint = load_checkpoint(model, args.checkpoint, map_location=map_location)
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        # Deyun: fix a bug
        # Explicitly move the model to the GPU
        if torch.cuda.is_available():
            model = model.cuda()
        model = DataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = DistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                        args.gpu_collect)



    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            print(dataset.evaluate(outputs['bbox_results'], **eval_kwargs))

        # Optional: collect per-frame info (planning metrics etc.) into a JSON list
        if args.collect_data:
            if not args.data_output:
                print('WARNING: --collect-data set but --data-output not provided; skip saving JSON.')
            else:
                # Helpers for JSON serialization
                def _to_jsonable(x):
                    if torch.is_tensor(x):
                        return x.detach().cpu().numpy().tolist()
                    if isinstance(x, np.ndarray):
                        return x.tolist()
                    if isinstance(x, (np.integer, )):
                        return int(x)
                    if isinstance(x, (np.floating, )):
                        return float(x)
                    if isinstance(x, dict):
                        return {k: _to_jsonable(v) for k, v in x.items()}
                    if isinstance(x, (list, tuple)):
                        return [_to_jsonable(v) for v in x]
                    return x

                # Build scenario_idx by clip (folder) in order of appearance
                folder_to_scenario = {}
                next_scenario = 0
                collected = []
                results_list = outputs.get('bbox_results', [])
                for batch_idx, res in enumerate(results_list):
                    # Get info from dataset if available (B2D repair pkl provides folder/town_name/frame_idx)
                    info = None
                    if hasattr(dataset, 'data_infos') and batch_idx < len(dataset.data_infos):
                        info = dataset.data_infos[batch_idx]

                    folder = ''
                    town_name = ''
                    frame_idx = batch_idx
                    timestamp = 0.0
                    if isinstance(info, dict):
                        folder = info.get('folder', '') or ''
                        town_name = info.get('town_name', '') or ''
                        frame_idx = int(info.get('frame_idx', frame_idx))
                        timestamp = float(info.get('timestamp', timestamp)) if 'timestamp' in info else timestamp

                    if folder not in folder_to_scenario:
                        folder_to_scenario[folder] = next_scenario
                        next_scenario += 1
                    scenario_idx = folder_to_scenario[folder]

                    metric = res.get('metric_results', {}) if isinstance(res, dict) else {}
                    fut_valid_flag = bool(metric.get('fut_valid_flag', False))
                    # --- Extract ego GT / prediction trajectories (nuScenes-style naming) ---
                    # GT ego future trajs are derived from adjacent frames in the clip (not stored in a single frame).
                    gt_ego_fut_trajs = []
                    ego_fut_masks = []
                    ego_fut_cmd = []
                    ego_fut_cmd_idx = -1
                    # Align naming with nuScenes:
                    # - ego_features: 512-D model feature (input to last planning layers)
                    # - ego_lcf_feat: 9-D handcrafted ego state
                    ego_features = []
                    ego_lcf_feat = []
                    predictions = []

                    # (1) GT from dataset utilities (same logic as dataloader uses)
                    try:
                        if hasattr(dataset, 'get_ego_trajs'):
                            _, ego_fut_trajs_off, ego_fut_masks_arr, cmd_onehot = dataset.get_ego_trajs(
                                batch_idx, dataset.sample_interval, dataset.past_frames, dataset.future_frames
                            )
                            # offsets -> absolute in ego frame
                            gt_ego_fut_trajs = np.cumsum(np.asarray(ego_fut_trajs_off, dtype=np.float32), axis=0).tolist()
                            ego_fut_masks = np.asarray(ego_fut_masks_arr, dtype=np.float32).tolist()
                            ego_fut_cmd = np.asarray(cmd_onehot, dtype=np.float32).tolist()
                            if len(ego_fut_cmd) > 0:
                                ego_fut_cmd_idx = int(np.argmax(np.asarray(ego_fut_cmd)))
                    except Exception:
                        # Keep empty if anything goes wrong; collection should not break evaluation.
                        gt_ego_fut_trajs = []
                        ego_fut_masks = []
                        ego_fut_cmd = []
                        ego_fut_cmd_idx = -1

                    # (2) Ego handcrafted features (B2D ego_lcf_feat-style 9D vector)
                    if isinstance(info, dict):
                        try:
                            ego_translation = np.asarray(info.get('ego_translation', [0, 0, 0]), dtype=np.float32).reshape(-1)
                            ego_accel = np.asarray(info.get('ego_accel', [0, 0, 0, 0]), dtype=np.float32).reshape(-1)
                            if ego_accel.shape[0] < 4:
                                ego_accel = np.pad(ego_accel, (0, 4 - ego_accel.shape[0]), mode='constant')
                            ego_rotation_rate = np.asarray(info.get('ego_rotation_rate', [0, 0, 0]), dtype=np.float32).reshape(-1)
                            ego_size = np.asarray(info.get('ego_size', [0, 0]), dtype=np.float32).reshape(-1)
                            steer = float(info.get('steer', 0.0) or 0.0)
                            ego_lcf_feat = np.zeros(9, dtype=np.float32)
                            ego_lcf_feat[0:2] = ego_translation[0:2] if ego_translation.shape[0] >= 2 else 0.0
                            ego_lcf_feat[2:4] = ego_accel[2:4]
                            ego_lcf_feat[4] = float(ego_rotation_rate[-1]) if ego_rotation_rate.shape[0] > 0 else 0.0
                            ego_lcf_feat[5] = float(ego_size[1]) if ego_size.shape[0] > 1 else 0.0
                            ego_lcf_feat[6] = float(ego_size[0]) if ego_size.shape[0] > 0 else 0.0
                            ego_lcf_feat[7] = float(np.sqrt((ego_translation[0] if ego_translation.shape[0] > 0 else 0.0) ** 2 +
                                                            (ego_translation[1] if ego_translation.shape[0] > 1 else 0.0) ** 2))
                            ego_lcf_feat[8] = float(steer)
                            ego_lcf_feat = ego_lcf_feat.tolist()
                        except Exception:
                            ego_lcf_feat = []

                    # (3) Ego_features from model outputs (expected 512-D; saved as nuScenes-style ego_features)
                    try:
                        feat_src = None
                        if isinstance(res, dict):
                            if 'ego_features' in res:
                                feat_src = res
                            elif 'pts_bbox' in res and isinstance(res['pts_bbox'], dict) and 'ego_features' in res['pts_bbox']:
                                feat_src = res['pts_bbox']
                        if feat_src is not None and feat_src.get('ego_features', None) is not None:
                            ef = feat_src['ego_features']
                            if torch.is_tensor(ef):
                                ef = ef.detach().cpu().numpy()
                            ef = np.asarray(ef, dtype=np.float32).reshape(-1)
                            ego_features = ef.tolist()
                    except Exception:
                        ego_features = []

                    # (4) Predictions from model outputs (ego_fut_preds: per-command future offsets)
                    try:
                        pred_src = None
                        if isinstance(res, dict):
                            # Some codepaths wrap bbox results under 'pts_bbox'
                            if 'ego_fut_preds' in res:
                                pred_src = res
                            elif 'pts_bbox' in res and isinstance(res['pts_bbox'], dict) and 'ego_fut_preds' in res['pts_bbox']:
                                pred_src = res['pts_bbox']

                        if pred_src is not None and pred_src.get('ego_fut_preds', None) is not None:
                            pred = pred_src['ego_fut_preds']
                            if torch.is_tensor(pred):
                                pred = pred.detach().cpu().numpy()
                            pred = np.asarray(pred, dtype=np.float32)
                            # offsets -> absolute in ego frame
                            pred = np.cumsum(pred, axis=-2)
                            predictions = pred.tolist()
                    except Exception as e:
                        # Do not break eval; but print a one-time hint for debugging.
                        if batch_idx == 0:
                            try:
                                v = None
                                if isinstance(res, dict):
                                    v = res.get('ego_fut_preds', None)
                                    if v is None and 'pts_bbox' in res and isinstance(res['pts_bbox'], dict):
                                        v = res['pts_bbox'].get('ego_fut_preds', None)
                                if torch.is_tensor(v):
                                    v_desc = 'torch(shape={}, dtype={})'.format(tuple(v.shape), v.dtype)
                                elif v is None:
                                    v_desc = 'None'
                                else:
                                    try:
                                        v_np = np.asarray(v)
                                        v_desc = 'ndarray(shape={}, dtype={})'.format(v_np.shape, v_np.dtype)
                                    except Exception:
                                        v_desc = str(type(v))
                                print('[collect-data] WARN: failed to parse ego_fut_preds for predictions: {} (ego_fut_preds={})'.format(e, v_desc))
                            except Exception:
                                pass
                        predictions = []

                    # Optional: save occupancy/segmentation for collision recomputation
                    occ_path = None
                    if args.occ_output_dir:
                        try:
                            anns = dataset.get_ann_info(batch_idx)
                            gt_bboxes_3d = anns.get('gt_bboxes_3d', None)
                            attr_labels = anns.get('attr_labels', None)
                            if gt_bboxes_3d is not None and attr_labels is not None:
                                pm = PlanningMetric()
                                attr_t = torch.as_tensor(attr_labels, dtype=torch.float32).unsqueeze(0)
                                seg, ped = pm.get_label(gt_bboxes_3d, attr_t)
                                occ = torch.logical_or(seg, ped).to(torch.uint8).squeeze(0).cpu().numpy()
                                os.makedirs(args.occ_output_dir, exist_ok=True)
                                safe_folder = folder if folder else 'unknown_scene'
                                occ_scene_dir = osp.join(args.occ_output_dir, safe_folder)
                                os.makedirs(occ_scene_dir, exist_ok=True)
                                occ_path = osp.join(occ_scene_dir, f"{int(frame_idx):06d}.npz")
                                if not osp.exists(occ_path):
                                    np.savez_compressed(occ_path, occ=occ)
                        except Exception as e:
                            print(f"[collect-data] WARN: failed to save occ/seg for frame {batch_idx}: {e}")

                    item = {
                        'batch_idx': int(batch_idx),
                        'scenario_idx': int(scenario_idx),
                        'scenario_abs_idx': int(scenario_idx + 1),
                        'frame_idx': int(frame_idx),
                        # For B2D we use folder as the scene identifier (clip-level)
                        'scene_token': str(folder),
                        'timestamp': float(timestamp),
                        # Planning metrics (match nuScenes-style naming)
                        'plan_L2_1s': float(_to_jsonable(metric.get('plan_L2_1s', 0.0)) or 0.0),
                        'plan_L2_2s': float(_to_jsonable(metric.get('plan_L2_2s', 0.0)) or 0.0),
                        'plan_L2_3s': float(_to_jsonable(metric.get('plan_L2_3s', 0.0)) or 0.0),
                        'plan_obj_box_col_1s': float(_to_jsonable(metric.get('plan_obj_box_col_1s', 0.0)) or 0.0) if fut_valid_flag else -1.0,
                        'plan_obj_box_col_2s': float(_to_jsonable(metric.get('plan_obj_box_col_2s', 0.0)) or 0.0) if fut_valid_flag else -1.0,
                        'plan_obj_box_col_3s': float(_to_jsonable(metric.get('plan_obj_box_col_3s', 0.0)) or 0.0) if fut_valid_flag else -1.0,
                        'fut_valid_flag': bool(fut_valid_flag),
                        # Match nuScenes preprocessing schema
                        'predictions': predictions,
                        'ground_truth': gt_ego_fut_trajs,
                        'ego_features': ego_features,
                        'ego_lcf_feat': ego_lcf_feat,
                        'ego_fut_cmd': ego_fut_cmd,
                        'ego_fut_cmd_idx': int(ego_fut_cmd_idx),
                        # Extra B2D-specific helper (not in nuScenes json): future mask for debugging/repair
                        'ego_fut_masks': ego_fut_masks,
                        'town_name': str(town_name),
                    }
                    if occ_path is not None:
                        item['occ_path'] = str(occ_path)
                    collected.append(item)

                os.makedirs(os.path.dirname(args.data_output) or '.', exist_ok=True)
                with open(args.data_output, 'w', encoding='utf-8') as f:
                    json.dump(collected, f, indent=2)
                print('Saved collected data to {} (num_frames={})'.format(args.data_output, len(collected)))

                # Optional: print a small preview for quick verification
                if getattr(args, 'print_collected', 0) and int(args.print_collected) > 0:
                    n = min(int(args.print_collected), len(collected))
                    print('Preview first {} collected items:'.format(n))
                    for i in range(n):
                        it = collected[i]
                        print(
                            '  [{}] scene_token={} scenario_idx={} frame_idx={} '
                            'L2(1/2/3s)=({:.4f},{:.4f},{:.4f}) col(1/2/3s)=({:.1f},{:.1f},{:.1f}) fut_valid={}'.format(
                                i,
                                it.get('scene_token', ''),
                                it.get('scenario_idx', -1),
                                it.get('frame_idx', -1),
                                float(it.get('plan_L2_1s', 0.0)),
                                float(it.get('plan_L2_2s', 0.0)),
                                float(it.get('plan_L2_3s', 0.0)),
                                float(it.get('plan_obj_box_col_1s', -1.0)),
                                float(it.get('plan_obj_box_col_2s', -1.0)),
                                float(it.get('plan_obj_box_col_3s', -1.0)),
                                bool(it.get('fut_valid_flag', False)),
                            )
                        )
    
        # # # NOTE: record to json
        # json_path = args.json_dir
        # if not os.path.exists(json_path):
        #     os.makedirs(json_path)
        
        # metric_all = []
        # for res in outputs['bbox_results']:
        #     for k in res['metric_results'].keys():
        #         if type(res['metric_results'][k]) is np.ndarray:
        #             res['metric_results'][k] = res['metric_results'][k].tolist()
        #     metric_all.append(res['metric_results'])
        
        # print('start saving to json done')
        # with open(json_path+'/metric_record.json', "w", encoding="utf-8") as f2:
        #     json.dump(metric_all, f2, indent=4)
        # print('save to json done')

if __name__ == '__main__':
    main()
