"""GPU image pre-processing shared by vad_b2d_agent_gpu.py and uniad_b2d_agent_gpu.py.

GpuPreprocMixin, placed before the original agent class (VadAgent, UniadAgent), changes only
how the camera images are prepared for the model; the original agents are unchanged.

1. Each camera image is processed as soon as it arrives. CARLA renders and sends the
   cameras one after another (about 8 ms apart); the original agents wait for all sensor
   data of a tick before they start. Here each image is handed to a worker thread on
   arrival for the BGR->RGB conversion and the JPEG round trip (quality 20) of tick(), so
   most of that work overlaps with the rendering of the later cameras.
2. The image pipeline (normalise, pad to a multiple of 32, [VAD: resize to 0.8x, pad
   again,] stack) runs on the GPU instead of numpy/cv2. Each tick uploads the six uint8
   images once; what is the same in every tick (mean and 1/std, the indices and weights of
   the resize) is prepared once and kept in GPU memory. The image metadata (shapes,
   lidar2img, ...) is still produced by the original transform code.

Normalisation, padding and stacking are bit-identical to the original agents. The resize
(VAD only) follows the bilinear formula of cv2.resize but not its optimised kernels, so
pixel values differ by up to ~6e-5 after normalisation (~0.003 grey levels).
"""
import math
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from queue import Empty

import cv2
import numpy as np
import torch

from leaderboard.envs.sensor_interface import SensorReceivedNoData
from mmcv.datasets.pipelines.formating import DefaultFormatBundle3D, VADFormatBundle3D
from mmcv.datasets.pipelines.test_time_aug import MultiScaleFlipAug3D
from mmcv.datasets.pipelines.transforms_3d import (NormalizeMultiviewImage, PadMultiViewImage,
                                                   RandomScaleImageMultiViewImage)
from mmcv.parallel import DataContainer as DC
from srunner.scenariomanager.timer import GameTime

CAMS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
_ENCODE = [int(cv2.IMWRITE_JPEG_QUALITY), 20]


def _jpeg_round_trip(bgra):
    """The per-camera image step of VadAgent.tick and UniadAgent.tick."""
    img = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB)
    _, img = cv2.imencode('.jpg', img, _ENCODE)
    return cv2.imdecode(img, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------------------------
# Image pipeline on the GPU
# ---------------------------------------------------------------------------------------------
class _GpuImage:
    """An (H, W, C) image tensor on the GPU with a numpy-like .shape for the metadata code."""
    __slots__ = ('t',)

    def __init__(self, t):
        self.t = t

    @property
    def shape(self):
        return tuple(int(s) for s in self.t.shape)


class _GpuOps:
    """Resize tables, prepared once per (input size, output size), on one device."""

    def __init__(self, device):
        self.device = device
        self._resize = {}

    @staticmethod
    def pad(t, shape):
        out = t.new_zeros((shape[0], shape[1]) + tuple(t.shape[2:]))
        out[:t.shape[0], :t.shape[1]] = t
        return out

    def _axis(self, src, dst):
        """Source indices and weights of a bilinear resize along one axis (half-pixel centres)."""
        f = ((np.arange(dst) + 0.5) * (src / dst) - 0.5).astype(np.float32)
        s = np.floor(f).astype(np.int64)
        f = (f - s).astype(np.float32)
        f[s < 0] = 0
        s[s < 0] = 0
        hi = s >= src - 1
        f[hi] = 0
        s[hi] = src - 1
        s1 = np.minimum(s + 1, src - 1)
        return (torch.tensor(s, device=self.device), torch.tensor(s1, device=self.device),
                torch.tensor(1 - f, device=self.device), torch.tensor(f, device=self.device))

    def resize(self, t, w, h):
        key = (t.shape[0], t.shape[1], h, w)
        if key not in self._resize:
            self._resize[key] = (self._axis(t.shape[1], w), self._axis(t.shape[0], h))
        (x0, x1, ax0, ax1), (y0, y1, ay0, ay1) = self._resize[key]
        tmp = t[:, x0] * ax0[None, :, None] + t[:, x1] * ax1[None, :, None]
        return tmp[y0] * ay0[:, None, None] + tmp[y1] * ay1[:, None, None]


class _GpuNormalize(NormalizeMultiviewImage):
    def _normalize(self, img_u8):
        # float64 arithmetic rounded to float32, as cv2.subtract/cv2.multiply with float64 scalars
        x = torch.from_numpy(img_u8).to(self._mean.device, non_blocking=True).double()
        if self.to_rgb:
            x = x[..., [2, 1, 0]]
        return ((x - self._mean) * self._stdinv).float()

    def __call__(self, results):
        results['img'] = [_GpuImage(self._normalize(img)) for img in results['img']]
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


class _GpuPad(PadMultiViewImage):
    def _pad_img(self, results):
        d = self.size_divisor
        padded_img = [_GpuImage(self._ops.pad(img.t, (int(np.ceil(img.shape[0] / d)) * d,
                                                      int(np.ceil(img.shape[1] / d)) * d)))
                      for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor


class _GpuRandomScale(RandomScaleImageMultiViewImage):
    def __call__(self, results):
        rand_ind = np.random.permutation(range(len(self.scales)))[0]   # same RNG use as the original
        rand_scale = self.scales[rand_ind]
        y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['img']]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results['img'] = [_GpuImage(self._ops.resize(img.t, x_size[i], y_size[i]))
                          for i, img in enumerate(results['img'])]
        results['lidar2img'] = [scale_factor @ l2i for l2i in results['lidar2img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]
        return results


def _stack_on_gpu(results, imgs):
    # as np.ascontiguousarray(np.stack([img.transpose(2, 0, 1) ...])) -> tensor, but on the GPU
    results['img'] = DC(torch.stack([img.t.permute(2, 0, 1) for img in imgs]).contiguous(), stack=True)
    return results


class _GpuFormatBundle(DefaultFormatBundle3D):
    def __call__(self, results):
        imgs = results.pop('img')
        return _stack_on_gpu(super().__call__(results), imgs)


class _GpuVADFormatBundle(VADFormatBundle3D):
    def __call__(self, results):
        imgs = results.pop('img')
        return _stack_on_gpu(super().__call__(results), imgs)


class _MultiScaleFlipAug3DNoImageCopy(MultiScaleFlipAug3D):
    """MultiScaleFlipAug3D for one scale without flipping (the VAD and UniAD configs), whose deep copy of the
    results skips the images: the following transforms return new images, never modify them."""

    def __call__(self, results):
        _results = deepcopy(results, {id(img): img for img in results['img']})
        _results.update(scale=self.img_scale[0], flip=False, pcd_scale_factor=self.pts_scale_ratio[0],
                        flip_direction=self.flip_direction[0], pcd_horizontal_flip=False, pcd_vertical_flip=False)
        return {key: [val] for key, val in self.transforms(_results).items()}


_GPU = {NormalizeMultiviewImage: _GpuNormalize, PadMultiViewImage: _GpuPad,
        RandomScaleImageMultiViewImage: _GpuRandomScale, DefaultFormatBundle3D: _GpuFormatBundle,
        VADFormatBundle3D: _GpuVADFormatBundle}


def _to_gpu(compose, ops):
    """Swap the image transforms of the pipeline (in place) for their GPU versions."""
    for t in compose.transforms:
        if type(t) in _GPU:
            t.__class__ = _GPU[type(t)]
            t._ops = ops
            if type(t) is _GpuNormalize:
                t._mean = torch.tensor(np.float64(t.mean), device=ops.device)
                t._stdinv = 1 / torch.tensor(np.float64(t.std), device=ops.device)
        elif type(t) is MultiScaleFlipAug3D:
            t.__class__ = _MultiScaleFlipAug3DNoImageCopy
            _to_gpu(t.transforms, ops)
    return compose


# ---------------------------------------------------------------------------------------------
# Agent mixin
# ---------------------------------------------------------------------------------------------
class GpuPreprocMixin:
    """Put before the original agent class: class VadAgentGPU(GpuPreprocMixin, VadAgent)."""

    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        self._pool = ThreadPoolExecutor(max_workers=len(CAMS))
        _to_gpu(self.inference_only_pipeline, _GpuOps(torch.device('cuda')))

    def _get_data(self, frame):
        """SensorInterface.get_data, but each camera image is handed to the pool on arrival."""
        si = self.sensor_interface
        data, self._jpeg = {}, {}
        try:
            while len(data) < len(si._sensors_objects):
                tag, f, d = si._data_buffers.get(True, si._queue_timeout)
                if f != frame:
                    continue
                data[tag] = (f, d)
                if tag in CAMS:
                    self._jpeg[tag] = self._pool.submit(_jpeg_round_trip, d)
        except Empty:
            raise SensorReceivedNoData("A sensor took too long to send their data")
        return data

    def __call__(self):
        # AutonomousAgent.__call__ with _get_data instead of sensor_interface.get_data
        input_data = self._get_data(GameTime.get_frame())
        timestamp = GameTime.get_time()
        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()
        wallclock = GameTime.get_wallclocktime()
        wallclock_diff = (wallclock - self.wallclock_t0).total_seconds()
        sim_ratio = 0 if wallclock_diff == 0 else timestamp / wallclock_diff
        print('=== [Agent] -- Wallclock = {} -- System time = {} -- Game time = {} -- Ratio = {}x'.format(
            str(wallclock)[:-3], format(wallclock_diff, '.3f'), format(timestamp, '.3f'), format(sim_ratio, '.3f')),
            flush=True)
        control = self.run_step(input_data, timestamp)
        control.manual_gear_shift = False
        return control

    def tick(self, input_data):
        # VadAgent.tick / UniadAgent.tick (identical), using the JPEG round trips started on arrival
        self.step += 1
        imgs = {cam: self._jpeg[cam].result() for cam in CAMS}
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]

        pos = self.gps_to_location(gps)
        near_node, near_command = self._route_planner.run_step(pos)

        if (math.isnan(compass) == True):  # noqa: E712  (as in VadAgent) the compass can be nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        return {
            'imgs': imgs,
            'gps': gps,
            'pos': pos,
            'speed': speed,
            'compass': compass,
            'bev': bev,
            'acceleration': acceleration,
            'angular_velocity': angular_velocity,
            'command_near': near_command,
            'command_near_xy': near_node,
        }

    def destroy(self):
        self._pool.shutdown(wait=False)
        super().destroy()
