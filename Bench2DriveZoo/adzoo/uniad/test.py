import argparse
import torch
import os
import warnings
import json
import numpy as np
from torch.nn.parallel.distributed import DistributedDataParallel
from mmcv.utils import get_dist_info, init_dist, wrap_fp16_model, set_random_seed, Config, DictAction, load_checkpoint
from mmcv.fileio.io import dump
from mmcv.datasets import build_dataset, build_dataloader, replace_ImageToTensor
from mmcv.models import build_model, fuse_conv_bn
import time
import os.path as osp
from adzoo.uniad.test_utils import custom_multi_gpu_test, custom_single_gpu_test
import cv2
cv2.setNumThreads(1)

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default='output/results.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
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
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
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
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--collect-data',
        action='store_true',
        help='Collect per-frame planning metrics and intermediate info into a JSON file.')
    parser.add_argument(
        '--data-output',
        type=str,
        default=None,
        help='Output JSON path used with --collect-data (e.g., baseline/uniad_baseline.json).')
    parser.add_argument(
        '--occ-output-dir',
        type=str,
        default=None,
        help='Optional directory to save per-frame occupancy/segmentation for collision recomputation.')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        torch.backends.cudnn.benchmark = True
        init_dist(args.launcher, **cfg.dist_params)
        rank, world_size = get_dist_info()

    set_random_seed(args.seed, deterministic=args.deterministic)

    # Dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset,
                                    samples_per_gpu=samples_per_gpu,
                                    workers_per_gpu=cfg.data.workers_per_gpu,
                                    dist=distributed,
                                    shuffle=False,
                                    nonshuffler_sampler=cfg.data.nonshuffler_sampler,
                                    )

    # Model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    
    # Add classese info
    if 'CLASSES' in checkpoint.get('meta', {}): # for det
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):  # for seg
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        model.PALETTE = dataset.PALETTE

    if not distributed:
        assert False #TODO(yzj)
        # model = MMDataParallel(model, device_ids=[0])
        # outputs = custom_single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = DistributedDataParallel(model.cuda(),
                                        device_ids=[torch.cuda.current_device()],
                                        broadcast_buffers=False,
                                        )
        outputs = custom_multi_gpu_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
            collect_data=args.collect_data,
            occ_output_dir=args.occ_output_dir,
        )



    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if 'jsonfile_prefix' not in kwargs:
            kwargs['jsonfile_prefix'] = osp.join('test', args.config.split('/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))

        # Optional: collect per-frame info (planning metrics etc.) into a JSON list
        if args.collect_data:
            if not args.data_output:
                print('WARNING: --collect-data set but --data-output not provided; skip saving JSON.')
            else:
                # Get collected data from outputs if available
                collected_data = outputs.get('collected_data', [])
                if collected_data:
                    os.makedirs(os.path.dirname(args.data_output) or '.', exist_ok=True)
                    with open(args.data_output, 'w', encoding='utf-8') as f:
                        json.dump(collected_data, f, indent=2)
                    print(f'\nCollected data saved to {args.data_output} (num_frames={len(collected_data)})')
                else:
                    print('WARNING: No collected data found in outputs. Make sure test_utils.py collects per-frame data.')


if __name__ == '__main__':
    main()
