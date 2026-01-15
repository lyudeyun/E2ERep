import os
import os.path as osp
import pickle
import shutil
import tempfile
import time

import torch
import torch.distributed as dist

from mmcv.models.dense_heads.occ_head_plugin import IntersectionOverUnion, PanopticMetric
from mmcv.models.dense_heads.planning_head_plugin import UniADPlanningMetric
from mmcv.utils import ProgressBar, mkdir_or_exist, get_dist_info
from mmcv.fileio.io import load, dump
import numpy as np
import pycocotools.mask as mask_util

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, collect_data=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()

    # Occ eval init
    eval_occ = hasattr(model.module, 'with_occ_head') \
                and model.module.with_occ_head
    if eval_occ:
        # 30mx30m, 100mx100m at 50cm resolution
        EVALUATION_RANGES = {'30x30': (70, 130),
                            '100x100': (0, 200)}
        n_classes = 2
        iou_metrics = {}
        for key in EVALUATION_RANGES.keys():
            iou_metrics[key] = IntersectionOverUnion(n_classes).cuda()
        panoptic_metrics = {}
        for key in EVALUATION_RANGES.keys():
            panoptic_metrics[key] = PanopticMetric(n_classes=n_classes, temporally_consistent=True).cuda()
    
    # Plan eval init
    eval_planning =  hasattr(model.module, 'with_planning_head') \
                      and model.module.with_planning_head
    if eval_planning:
        planning_metrics = UniADPlanningMetric().cuda()
        
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    num_occ = 0
    collected_data = []  # For per-frame data collection
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(data, return_loss=False, rescale=True)


            #import pdb;pdb.set_trace()
            
            # # EVAL planning
            if eval_planning:
                # TODO: Wrap below into a func
                segmentation = result[0]['planning']['planning_gt']['segmentation']
                sdc_planning = result[0]['planning']['planning_gt']['sdc_planning']
                sdc_planning_mask = result[0]['planning']['planning_gt']['sdc_planning_mask']
                pred_sdc_traj = result[0]['planning']['result_planning']['sdc_traj']
                result[0]['planning_traj'] = result[0]['planning']['result_planning']['sdc_traj']
                result[0]['planning_traj_gt'] = result[0]['planning']['planning_gt']['sdc_planning']
                result[0]['command'] = result[0]['planning']['planning_gt']['command']
                planning_metrics(pred_sdc_traj[:, :6, :2], sdc_planning[0][0,:, :6, :2], sdc_planning_mask[0][0,:, :6, :2], segmentation[0][:, [1,2,3,4,5,6]])
                
                # Collect per-frame planning data if requested
                if collect_data and rank == 0:
                    # Compute per-frame metrics
                    pred_traj = pred_sdc_traj[0, :6, :2].cpu().numpy()  # (6, 2)
                    # Extract gt_traj using the same pattern as planning_metrics (line 104)
                    # planning_metrics uses: sdc_planning[0][0,:, :6, :2]
                    # This means: sdc_planning[0] -> (1, num_commands, planning_steps, 3)
                    #            sdc_planning[0][0] -> (num_commands, planning_steps, 3)
                    #            sdc_planning[0][0,:] -> (num_commands, planning_steps, 3)
                    #            sdc_planning[0][0,:, :6, :2] -> (num_commands, 6, 2)
                    try:
                        gt_traj_all = sdc_planning[0][0, :, :6, :2].cpu().numpy()  # (num_commands, 6, 2)
                        gt_mask_all = sdc_planning_mask[0][0, :, :6].cpu().numpy()  # (num_commands, 6)
                        # UniAD is single-mode, take first command
                        gt_traj = gt_traj_all[0]  # (6, 2)
                        gt_mask = gt_mask_all[0]  # (6,)
                    except (IndexError, RuntimeError) as e:
                        # Fallback: try alternative indexing if shape is different
                        # Maybe sdc_planning[0] is (1, planning_steps, 3) instead
                        try:
                            gt_traj = sdc_planning[0][0, :6, :2].cpu().numpy()  # (6, 2)
                            gt_mask = sdc_planning_mask[0][0, :6].cpu().numpy()  # (6,)
                        except (IndexError, RuntimeError):
                            # Last resort: try to extract directly
                            sp = sdc_planning[0].cpu().numpy()
                            spm = sdc_planning_mask[0].cpu().numpy()
                            # Flatten and reshape if needed
                            if sp.ndim == 3 and sp.shape[1] >= 6:
                                gt_traj = sp[0, :6, :2]  # (6, 2)
                                gt_mask = spm[0, :6] if spm.ndim >= 2 else spm[:6]  # (6,)
                            else:
                                # Create default values
                                gt_traj = np.zeros((6, 2), dtype=np.float32)
                                gt_mask = np.zeros(6, dtype=np.float32)
                                if rank == 0:
                                    print(f"Warning: Could not extract gt_traj, shape={sp.shape}, using zeros")
                    
                    # Compute L2 error for each timestep
                    l2_errors = np.sqrt(np.sum((pred_traj - gt_traj) ** 2, axis=-1)) * gt_mask
                    plan_L2_1s = float(l2_errors[1]) if len(l2_errors) > 1 else 0.0  # 1.0s = index 1 (0.5s intervals)
                    plan_L2_2s = float(l2_errors[3]) if len(l2_errors) > 3 else 0.0  # 2.0s = index 3
                    plan_L2_3s = float(l2_errors[5]) if len(l2_errors) > 5 else 0.0  # 3.0s = index 5
                    
                    # Compute collision metrics (simplified - use planning_metrics logic)
                    # For now, set to -1.0 if invalid, will be computed properly if needed
                    fut_valid_flag = bool(gt_mask.sum() > 0)
                    plan_obj_box_col_1s = -1.0 if not fut_valid_flag else 0.0  # Placeholder
                    plan_obj_box_col_2s = -1.0 if not fut_valid_flag else 0.0
                    plan_obj_box_col_3s = -1.0 if not fut_valid_flag else 0.0
                    
                    # Get dataset info if available
                    info = None
                    if hasattr(dataset, 'data_infos') and i < len(dataset.data_infos):
                        info = dataset.data_infos[i]
                    
                    folder = ''
                    town_name = ''
                    frame_idx = i
                    timestamp = 0.0
                    if isinstance(info, dict):
                        folder = info.get('folder', '') or ''
                        town_name = info.get('town_name', '') or ''
                        frame_idx = int(info.get('frame_idx', frame_idx))
                        timestamp = float(info.get('timestamp', timestamp)) if 'timestamp' in info else timestamp
                    
                    # --- Extract ego GT / prediction trajectories ---
                    # GT ego future trajs from dataset (if available)
                    gt_ego_fut_trajs = gt_traj.tolist()  # Default: use sdc_planning
                    ego_fut_masks = gt_mask.tolist()  # Default: use sdc_planning_mask
                    try:
                        if hasattr(dataset, 'get_ego_future_xy'):
                            # B2D_e2e_dataset has get_ego_future_xy method
                            ego_fut_trajs_off, ego_fut_masks_arr = dataset.get_ego_future_xy(
                                i, dataset.sample_interval, dataset.future_frames
                            )
                            # offsets -> absolute in ego frame
                            if len(ego_fut_trajs_off) > 0 and ego_fut_trajs_off.shape[1] >= 6:
                                gt_ego_fut_trajs = np.cumsum(np.asarray(ego_fut_trajs_off[0, :6, :2], dtype=np.float32), axis=0).tolist()
                                ego_fut_masks = np.asarray(ego_fut_masks_arr[0, :6, :], dtype=np.float32).tolist()
                    except Exception:
                        pass  # Use default values from sdc_planning
                    
                    # Ego_features from planning head (equivalent to VAD's ego_features)
                    ego_features = None
                    try:
                        planning_result = result[0].get('planning', {}).get('result_planning', {})
                        if 'ego_features' in planning_result:
                            ef = planning_result['ego_features']
                            if torch.is_tensor(ef):
                                ef = ef.detach().cpu().numpy()
                            ef = np.asarray(ef, dtype=np.float32).reshape(-1)
                            ego_features = ef.tolist()
                    except Exception:
                        pass  # Skip if not available
                    
                    # Ego handcrafted features (only if available in dataset info)
                    ego_lcf_feat = None
                    if isinstance(info, dict):
                        try:
                            # Check if required fields exist
                            if all(key in info for key in ['ego_translation', 'ego_accel', 'ego_rotation_rate', 'ego_size', 'steer']):
                                ego_translation = np.asarray(info['ego_translation'], dtype=np.float32).reshape(-1)
                                ego_accel = np.asarray(info['ego_accel'], dtype=np.float32).reshape(-1)
                                if ego_accel.shape[0] < 4:
                                    ego_accel = np.pad(ego_accel, (0, 4 - ego_accel.shape[0]), mode='constant')
                                ego_rotation_rate = np.asarray(info['ego_rotation_rate'], dtype=np.float32).reshape(-1)
                                ego_size = np.asarray(info['ego_size'], dtype=np.float32).reshape(-1)
                                steer = float(info['steer'] or 0.0)
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
                            pass  # Skip if not available
                    
                    # Command from planning result (only if available)
                    # Get command_dim dynamically from model configuration
                    ego_fut_cmd = None
                    ego_fut_cmd_idx = None
                    try:
                        command = result[0].get('command', None)
                        if command is not None:
                            if torch.is_tensor(command):
                                command = command.cpu().numpy()
                            command = np.asarray(command, dtype=np.int32).reshape(-1)
                            if len(command) > 0:
                                # Get command_dim from model's planning_head configuration
                                command_dim = 3  # Default fallback
                                try:
                                    if hasattr(model, 'module') and hasattr(model.module, 'planning_head'):
                                        if hasattr(model.module.planning_head, 'navi_embed'):
                                            command_dim = model.module.planning_head.navi_embed.num_embeddings
                                except Exception:
                                    pass  # Use default if cannot get from model
                                
                                # Convert to one-hot based on actual command_dim
                                cmd_onehot = np.zeros(command_dim, dtype=np.float32)
                                cmd_idx = int(command[0]) if 0 <= command[0] < command_dim else 0
                                cmd_onehot[cmd_idx] = 1.0
                                ego_fut_cmd = cmd_onehot.tolist()
                                ego_fut_cmd_idx = int(cmd_idx)
                    except Exception:
                        pass  # Skip if not available
                    
                    # Convert trajectories to list format (cumulative offsets -> absolute positions)
                    # pred_traj is already in absolute coordinates
                    # Shape: (6, 2) -> list of 6 [x, y] pairs
                    predictions = pred_traj.tolist()
                    # Note: UniAD outputs 1 trajectory (single mode), while VAD outputs 6 trajectories (multi-mode)
                    # To match VAD's format, wrap the single trajectory in a list: [[trajectory]]
                    # This makes predictions a list containing one trajectory, matching VAD's structure
                    predictions = [predictions]
                    
                    # Build scenario_idx by folder
                    folder_to_scenario = {}
                    for j, collected_item in enumerate(collected_data):
                        folder_j = collected_item.get('scene_token', '')
                        if folder_j not in folder_to_scenario:
                            folder_to_scenario[folder_j] = len(folder_to_scenario)
                    if folder not in folder_to_scenario:
                        folder_to_scenario[folder] = len(folder_to_scenario)
                    scenario_idx = folder_to_scenario[folder]
                    
                    # Build item dict with only available fields
                    item = {
                        'batch_idx': int(i),
                        'scenario_idx': int(scenario_idx),
                        'scenario_abs_idx': int(scenario_idx + 1),
                        'frame_idx': int(frame_idx),
                        'scene_token': str(folder),
                        'timestamp': float(timestamp),
                        'plan_L2_1s': plan_L2_1s,
                        'plan_L2_2s': plan_L2_2s,
                        'plan_L2_3s': plan_L2_3s,
                        'plan_obj_box_col_1s': plan_obj_box_col_1s,
                        'plan_obj_box_col_2s': plan_obj_box_col_2s,
                        'plan_obj_box_col_3s': plan_obj_box_col_3s,
                        'fut_valid_flag': fut_valid_flag,
                        'predictions': predictions,
                        'ground_truth': gt_ego_fut_trajs,
                        'ego_fut_masks': ego_fut_masks,
                        'town_name': str(town_name),
                    }
                    
                    # Only add optional fields if they are available
                    if ego_features is not None:
                        item['ego_features'] = ego_features
                    if ego_lcf_feat is not None:
                        item['ego_lcf_feat'] = ego_lcf_feat
                    if ego_fut_cmd is not None:
                        item['ego_fut_cmd'] = ego_fut_cmd
                    if ego_fut_cmd_idx is not None:
                        item['ego_fut_cmd_idx'] = int(ego_fut_cmd_idx)
                    
                    collected_data.append(item)

            # # Eval Occ
            if eval_occ:
                occ_has_invalid_frame = data['gt_occ_has_invalid_frame'][0]
                occ_to_eval = not occ_has_invalid_frame.item()
                if occ_to_eval and 'occ' in result[0].keys():
                    try:
                        num_occ += 1
                        for key, grid in EVALUATION_RANGES.items():
                            limits = slice(grid[0], grid[1])
                            iou_metrics[key](result[0]['occ']['seg_out'][..., limits, limits].contiguous(),
                                            result[0]['occ']['seg_gt'][..., limits, limits].contiguous())
                            # Check dimensions before calling panoptic_metrics
                            ins_seg_out_sliced = result[0]['occ']['ins_seg_out'][..., limits, limits].contiguous().detach()
                            ins_seg_gt_sliced = result[0]['occ']['ins_seg_gt'][..., limits, limits].contiguous()
                            # Ensure dimensions are correct: should be (b, s, h, w) where b>=1, s>=1
                            if ins_seg_out_sliced.dim() == 4 and ins_seg_gt_sliced.dim() == 4:
                                panoptic_metrics[key](ins_seg_out_sliced, ins_seg_gt_sliced)
                            else:
                                # Skip panoptic evaluation for this sample if dimensions don't match
                                if rank == 0:
                                    print(f"Warning: Skipping panoptic evaluation for sample {i}, "
                                          f"ins_seg_out dim={ins_seg_out_sliced.dim()}, "
                                          f"ins_seg_gt dim={ins_seg_gt_sliced.dim()}")
                    except Exception as e:
                        # Skip occ evaluation for this sample if there's an error
                        # Note: Do NOT use 'continue' here, as it would skip the entire frame evaluation
                        # Only skip occ evaluation, but continue with bbox and planning evaluation
                        if rank == 0:
                            print(f"Warning: Skipping occ evaluation for sample {i} due to error: {e}")
                        # Don't use continue - let the frame continue to be evaluated for bbox/planning

            # Pop out unnecessary occ results, avoid appending it to cpu when collect_results_cpu
            if os.environ.get('ENABLE_PLOT_MODE', None) is None:
                result[0].pop('occ', None)
                result[0].pop('planning', None)
            else:
                for k in ['seg_gt', 'ins_seg_gt', 'pred_ins_sigmoid', 'seg_out', 'ins_seg_out']:
                    if k in result[0]['occ']:
                        result[0]['occ'][k] = result[0]['occ'][k].detach().cpu()
                for k in ['bbox', 'segm', 'labels', 'panoptic', 'drivable', 'score_list', 'lane', 'lane_score', 'stuff_score_list']:
                    if k in result[0]['pts_bbox'] and isinstance(result[0]['pts_bbox'][k], torch.Tensor):
                        result[0]['pts_bbox'][k] = result[0]['pts_bbox'][k].detach().cpu()

            # # encode mask results
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    batch_size = len(result['bbox_results'])
                    bbox_results.extend(bbox_result)
                if 'mask_results' in result.keys() and result['mask_results'] is not None:
                    mask_result = custom_encode_mask_results(result['mask_results'])
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                batch_size = len(result)
                bbox_results.extend(result)


        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()
                
        # break

    # collect results from all ranks
    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        if have_mask:
            mask_results = collect_results_gpu(mask_results, len(dataset))
        else:
            mask_results = None
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        if have_mask:
            mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
        else:
            mask_results = None

    if eval_planning:
        planning_results = planning_metrics.compute()
        planning_metrics.reset()

    ret_results = dict()
    ret_results['bbox_results'] = bbox_results
    if eval_occ:
        occ_results = {}
        for key, grid in EVALUATION_RANGES.items():
            panoptic_scores = panoptic_metrics[key].compute()
            for panoptic_key, value in panoptic_scores.items():
                occ_results[f'{panoptic_key}'] = occ_results.get(f'{panoptic_key}', []) + [100 * value[1].item()]
            panoptic_metrics[key].reset()

            iou_scores = iou_metrics[key].compute()
            occ_results['iou'] = occ_results.get('iou', []) + [100 * iou_scores[1].item()]
            iou_metrics[key].reset()

        occ_results['num_occ'] = num_occ  # count on one gpu
        occ_results['ratio_occ'] = num_occ / len(dataset)  # count on one gpu, but reflect the relative ratio
        ret_results['occ_results_computed'] = occ_results
    if eval_planning:
        ret_results['planning_results_computed'] = planning_results

    if mask_results is not None:
        ret_results['mask_results'] = mask_results
    if collect_data:
        # Collect collected_data from all ranks if needed
        if gpu_collect:
            # For simplicity, only rank 0 collects data
            if rank == 0:
                ret_results['collected_data'] = collected_data
            else:
                ret_results['collected_data'] = []
        else:
            # Collect from all ranks
            if rank == 0:
                all_collected = [collected_data]
                for _ in range(1, world_size):
                    tmp_file = osp.join(tmpdir or tempfile.gettempdir(), f'collected_data_rank_{_}.pkl')
                    if osp.exists(tmp_file):
                        all_collected.append(load(tmp_file))
                # Flatten and sort by batch_idx
                all_collected_flat = []
                for rank_data in all_collected:
                    all_collected_flat.extend(rank_data)
                all_collected_flat.sort(key=lambda x: x['batch_idx'])
                ret_results['collected_data'] = all_collected_flat
            else:
                tmp_file = osp.join(tmpdir or tempfile.gettempdir(), f'collected_data_rank_{rank}.pkl')
                dump(collected_data, tmp_file)
                ret_results['collected_data'] = []
    return ret_results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)

def custom_single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(data, result, out_dir=out_dir)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results