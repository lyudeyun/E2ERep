"""GPU cache of the open-loop repair frames, shared by the VAD and UniAD fitness evaluation.

The decoder inputs, expert trajectories, commands and occupancy grids of all frames are
loaded onto the GPU once per repair run. A fitness evaluation then needs no file access
and no per-frame Python code: collisions of all frames are checked with a few tensor
operations. The collision check reproduces the planning metrics of Bench2DriveZoo
(PlanningMetric for VAD, UniADPlanningMetric for UniAD) exactly: the same ego
footprint, float32 division by the cell size, float64 addition of the footprint offsets,
truncation to int32 and clipping to the grid. Only the mapping from trajectory
coordinates to grid cells differs between the two models (subclasses).
"""
import os
import time

import numpy as np
import torch
from skimage.draw import polygon

TIMESTEPS = {1: 2, 2: 4, 3: 6}   # time horizon (s) -> number of 0.5 s planning steps


class FrameCache:
    """All frames of a frame_data_dict as tensors on `device`.

    metric: the planning metric whose collision check is reproduced (for the ego size,
    grid origin, cell size and grid dimension). resolve_occ_path: maps the occ_path
    stored in the JSON to a file.
    """

    occ_key = 'occ'              # array name in the occupancy .npz files

    def __init__(self, frame_data_dict, device, metric, resolve_occ_path):
        t0 = time.time()
        frames = list(frame_data_dict.values())
        self.key = (id(frame_data_dict), len(frames), str(device))
        self.frames = frames
        self.n = len(frames)
        self.device = device

        self.feats = torch.tensor(np.array([np.asarray(f['ego_features'], dtype=np.float32) for f in frames]),
                                  dtype=torch.float32, device=device)
        gt = np.zeros((self.n, 6, 2), dtype=np.float32)
        for i, f in enumerate(frames):
            g = np.asarray(f.get('gt_future_traj', []), dtype=np.float32)
            if g.ndim == 1:
                g = g.reshape(-1, 2)
            k = min(g.shape[0], 6)
            gt[i, :k] = g[:k]
        self.gt = torch.from_numpy(gt).to(device)
        self.cmd = torch.tensor([f.get('ego_fut_cmd_idx', 0) for f in frames], dtype=torch.long, device=device)
        self.has_collision = np.array([bool(f.get('has_collision', False)) for f in frames])
        self._fields = {}

        # Occupancy of every frame, (N, 6, 200, 200) bool
        self.occ = torch.zeros((self.n, 6, 200, 200), dtype=torch.bool, device=device)
        for i, f in enumerate(frames):
            occ_path = resolve_occ_path(f.get('occ_path', None))
            if not occ_path or not os.path.exists(occ_path):
                raise ValueError(f"Missing occ_path for collision computation: {occ_path}. "
                                 "Please regenerate JSON with --occ-output-dir.")
            occ = np.load(occ_path)[self.occ_key]
            if occ.ndim == 4:
                occ = occ[0]
            self.occ[i] = torch.from_numpy(occ[:6] != 0)

        # Ego footprint in grid cells, as in evaluate_single_coll of both metrics
        bx = torch.as_tensor(metric.bx).detach().cpu().numpy()
        dx = torch.as_tensor(metric.dx).detach().cpu().numpy()
        pts = np.array([[-metric.H / 2. + 0.5, metric.W / 2.], [metric.H / 2. + 0.5, metric.W / 2.],
                        [metric.H / 2. + 0.5, -metric.W / 2.], [-metric.H / 2. + 0.5, -metric.W / 2.]])
        pts = (pts - bx) / dx
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:, 1], pts[:, 0])
        self.footprint = torch.tensor(np.stack([rr, cc], -1), dtype=torch.float64, device=device)  # (K, 2)
        self.dx = torch.as_tensor(metric.dx).detach().to(device)                                       # float32
        self.bev_h, self.bev_w = int(metric.bev_dimension[0]), int(metric.bev_dimension[1])

        self._init_extra(frames)
        # The expert trajectory never changes: its colliding steps are computed once
        self.gt_occupied = torch.cat([self._occupied(self.gt[s:s + 4096], s) for s in range(0, self.n, 4096)]) \
            if self.n else torch.zeros((0, 6), dtype=torch.bool, device=device)
        print(f"[fitness] cached {self.n} frames on {device} "
              f"(occupancy {self.occ.numel() / 1e9:.2f} GB) in {time.time() - t0:.0f}s", flush=True)

    def _init_extra(self, frames):
        """Model-specific per-frame data."""

    def _cells(self, traj):
        """(row, col) grid indices of the ego footprint at every step: two (B, 6, K) long tensors."""
        raise NotImplementedError

    def _occupied(self, traj, start):
        """(B, 6) bool: does the ego box at each step of traj overlap an occupied cell?"""
        r, c = self._cells(traj)
        b = torch.arange(start, start + traj.shape[0], device=traj.device)[:, None, None]
        t = torch.arange(traj.shape[1], device=traj.device)[None, :, None]
        return self.occ[b, t, r, c].any(-1)

    def collision(self, pred_abs, start, n_steps):
        """(B,) bool: the plan collides at a step where the expert does not, within n_steps."""
        end = start + pred_abs.shape[0]
        hit = self._occupied(pred_abs, start) & ~self.gt_occupied[start:end]
        return hit[:, :n_steps].any(1)

    def field(self, name):
        """A numeric per-frame value stored in the JSON (e.g. plan_L2_3s); NaN where missing."""
        if name not in self._fields:
            self._fields[name] = np.array([f.get(name, None) for f in self.frames], dtype=np.float64)
        return self._fields[name]


class VADFrameCache(FrameCache):
    """Cells as in metric_stp3.PlanningMetric.evaluate_single_coll (VAD)."""

    occ_key = 'occ'

    def _cells(self, traj):
        q = (traj[..., [1, 0]] / self.dx).double().unsqueeze(2) + self.footprint      # (B, 6, K, 2)
        r = (self.bev_h - q[..., 0]).to(torch.int32).clamp_(0, self.bev_h - 1).long()
        c = q[..., 1].to(torch.int32).clamp_(0, self.bev_w - 1).long()
        return r, c


class UniADFrameCache(FrameCache):
    """Cells as in planning_metrics.UniADPlanningMetric.evaluate_coll (UniAD): x is negated first."""

    occ_key = 'seg'

    def _init_extra(self, frames):
        self.flip = torch.tensor([-1, 1], device=self.device)
        self.gt_mask = torch.from_numpy(np.stack([uniad_gt_mask(f) for f in frames])).to(self.device) \
            if frames else torch.zeros((0, 6, 2), device=self.device)

    def _cells(self, traj):
        traj = traj * self.flip
        q = (traj[..., [1, 0]] / self.dx).double().unsqueeze(2) + self.footprint
        r = q[..., 0].to(torch.int32).clamp_(0, self.bev_h - 1).long()
        c = q[..., 1].to(torch.int32).clamp_(0, self.bev_w - 1).long()
        return r, c


def uniad_gt_mask(frame_data):
    """(6, 2) float32 L2 mask of the expert trajectory, as the UniAD evaluation built it
    (the same shape handling as convert_uniad_to_vad_metrics.py; all ones if missing)."""
    m = frame_data.get('ego_fut_mask', frame_data.get('ego_fut_masks', None))
    if m is None:
        return np.ones((6, 2), dtype=np.float32)
    m = np.array(m, dtype=np.float32)
    if m.ndim == 3:
        m = m[0, :6, :2]
    elif m.ndim == 2:
        m = m[:6, :2] if m.shape[1] >= 2 else m[:6, None]
    elif m.ndim == 1:
        m = m[:6, None]
    else:
        return np.ones((6, 2), dtype=np.float32)
    if m.shape[0] < 6:
        padded = np.ones((6, 2), dtype=np.float32)
        if m.ndim == 1:
            padded[:m.shape[0], :] = m[:m.shape[0], None]
        else:
            k = min(m.shape[1] if m.ndim > 1 else 1, 2)
            padded[:m.shape[0], :k] = m[:m.shape[0], :k]
        m = padded
    elif m.ndim == 1:
        m = np.column_stack([m[:6], m[:6]])
    elif m.ndim == 2:
        m = np.column_stack([m[:6, 0], m[:6, 0]]) if m.shape[1] < 2 else m[:6, :2]
    if m.shape == (6, 2):
        return m.astype(np.float32)
    # Any other shape: as the batch loop of the old evaluation filled its mask tensor
    out = np.zeros((6, 2), dtype=np.float32)
    k = min(m.shape[0], 6)
    d = min(m.shape[1], 2) if m.ndim > 1 else 2
    if m.ndim == 1:
        out[:k, :] = np.column_stack([m[:6], m[:6]])[:k, :]
    else:
        out[:k, :d] = m[:k, :d]
    if k < 6:
        out[k:, :] = 1.0
    if d < 2:
        out[:, d:] = 1.0
    return out
