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
import json
import os
import time

import numpy as np
import torch
from skimage.draw import polygon

TIMESTEPS = {1: 2, 2: 4, 3: 6}   # time horizon (s) -> number of 0.5 s planning steps


def occ_grid(occ):
    """(6, 200, 200) bool grid of an occupancy array as stored per frame by the test scripts."""
    if occ.ndim == 4:
        occ = occ[0]
    return occ[:6] != 0


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def occ_pack_path(json_path):
    """The occupancy pack of a baseline JSON: <name>.json -> <name>.occ.npz, next to it."""
    json_path = str(json_path)
    return (json_path[:-5] if json_path.endswith('.json') else json_path) + '.occ.npz'


def occ_frame_key(occ_path):
    """Key of a frame in a pack: <sub>/<clip>/<frame> from <cache>/<sub>/<clip>/<frame>.npz."""
    parts = os.path.normpath(str(occ_path)).split(os.sep)[-3:]
    parts[-1] = parts[-1][:-4] if parts[-1].endswith('.npz') else parts[-1]
    return '/'.join(parts)


class OccupancyReader:
    """Occupancy grids of single frames, addressed by the occ_path stored in the baseline JSON.
    A frame is read from the pack of its JSON (occ_pack_path) when one is given and holds it,
    otherwise from its per-frame file. Returns None if it has neither."""

    def __init__(self, key):
        self.key = key
        self.packs = {}      # pack file -> ({frame key: index}, bit-packed grids, width) or None

    def _pack(self, pack_file):
        if pack_file not in self.packs:
            self.packs[pack_file] = None
            if os.path.exists(pack_file):
                with np.load(pack_file) as z:
                    if str(z['key']) == self.key:
                        self.packs[pack_file] = ({str(n): i for i, n in enumerate(z['names'])},
                                                 z['bits'], int(z['width']))
        return self.packs[pack_file]

    def get(self, occ_path, pack_file=None):
        pack = self._pack(pack_file) if pack_file else None
        if pack is not None:
            i = pack[0].get(occ_frame_key(occ_path))
            if i is not None:
                return np.unpackbits(pack[1][i], axis=-1, count=pack[2]).astype(bool)
        if not os.path.exists(occ_path):
            return None
        return occ_grid(np.load(occ_path)[self.key])


def pack_occ(json_path, key, delete=False):
    """Write the occupancy grids of all frames of a baseline JSON into its pack
    (occ_pack_path: frame keys and bit-packed grids, in JSON order), read it back and compare
    with the per-frame files; only then, with delete=True, remove those files (and folders
    left empty). Other JSONs may share per-frame files: delete only once they have packs too."""
    frames = json.load(open(json_path))
    paths = []
    for f in frames:
        p = f.get('occ_path')
        if p:
            paths.append(p if os.path.isabs(p) else os.path.join(REPO_ROOT, p))
    if not paths:
        print(f'{json_path}: no occ_path, no pack written', flush=True)
        return None
    reader = OccupancyReader(key)
    grids = []
    for p in paths:
        g = reader.get(p)
        if g is None:
            raise FileNotFoundError(f'{json_path}: occupancy file {p} missing; no pack written')
        grids.append(g)
    grids = np.stack(grids)
    names = np.array([occ_frame_key(p) for p in paths])
    if len(set(names)) != len(names):
        raise ValueError(f'{json_path}: two frames share an occupancy key; no pack written')
    out = occ_pack_path(json_path)
    tmp = out + '.tmp'
    with open(tmp, 'wb') as fh:      # written next to the target, then renamed: never half a pack
        np.savez_compressed(fh, key=key, width=grids.shape[-1], names=names, bits=np.packbits(grids, axis=-1))
    os.replace(tmp, out)
    check = OccupancyReader(key)
    for p, g in zip(paths, grids):
        if not np.array_equal(check.get('/missing/' + occ_frame_key(p) + '.npz', out), g):
            raise RuntimeError(f'{out}: frame {p} differs from its per-frame file; nothing deleted')
    print(f'{out}: {len(paths)} frames, verified ({os.path.getsize(out) / 1e6:.1f} MB)', flush=True)
    if delete:
        dirs = set()
        for p in paths:
            if os.path.exists(p):
                os.remove(p)
                dirs.add(os.path.dirname(p))
        for d in sorted(dirs, key=len, reverse=True):
            while d and os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
                d = os.path.dirname(d)
        print(f'removed its {len(paths)} per-frame files', flush=True)
    return out


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
        grids = OccupancyReader(self.occ_key)
        for i, f in enumerate(frames):
            occ_path = resolve_occ_path(f.get('occ_path', None))
            grid = grids.get(occ_path, f.get('occ_pack')) if occ_path else None
            if grid is None:
                raise ValueError(f"Missing occ_path for collision computation: {occ_path}. "
                                 "Please regenerate JSON with --occ-output-dir.")
            self.occ[i] = torch.from_numpy(grid)
        del grids

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


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Pack the occupancy grids of baseline JSONs into <name>.occ.npz.')
    ap.add_argument('jsons', nargs='+', help='baseline JSONs, e.g. baseline/VAD/b2d_val/vad_base_baseline_b2d_infos_val.json')
    ap.add_argument('--key', default='occ', help="array name in the per-frame files: 'occ' (VAD) or 'seg' (UniAD)")
    ap.add_argument('--delete', action='store_true', help='remove the per-frame files once the pack is verified')
    args = ap.parse_args()
    for j in args.jsons:
        pack_occ(j, args.key, args.delete)
