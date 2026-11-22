"""Lossless checkpoint patches: store a repaired model as (base checkpoint + changed values).

A repair changes only a few weights, so saving the whole checkpoint per run wastes
hundreds of MB. A patch stores, for every tensor that differs from the base, either
the changed positions and their FINAL values (sparse) or the whole tensor (dense),
in the original dtype. Applying it writes the values back without arithmetic, so the
rebuilt checkpoint is bit-identical to the repaired one.

    from repair_common.ckpt_patch import make_patch, save_patch, load_patched, materialize
    patch = make_patch(base_ckpt_path, repaired_ckpt_or_state_dict)
    save_patch(patch, 'repair_patch.pt')
    ckpt = load_patched('repair_patch.pt')                 # dict, in memory
    path = materialize('repair_patch.pt', '/tmp/x.pth')     # full .pth for test.py / agents
    with materialized(path_or_patch) as ckpt_path: ...       # temporary full .pth, removed afterwards

The base checkpoint is referenced by absolute path and checked by size and SHA-256;
it must not be moved or modified (use base_path=... to point at a moved copy).
"""
import hashlib
import os
import tempfile
from contextlib import contextmanager

import numpy as np
import torch

FORMAT = 'e2erep-ckpt-patch-v1'
_SPARSE_MAX_FRAC = 0.5          # store densely when more than half of a tensor changed
_sha_cache = {}


def file_sha256(path, chunk=1 << 24):
    path = os.path.abspath(path)
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _sha_cache:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for b in iter(lambda: f.read(chunk), b''):
                h.update(b)
        _sha_cache[key] = h.hexdigest()
    return _sha_cache[key]


def _load(path_or_obj):
    if isinstance(path_or_obj, (str, os.PathLike)):
        return torch.load(path_or_obj, map_location='cpu')
    return path_or_obj


def _split(ckpt):
    """Return (state_dict, top-level dict without state_dict, has_wrapper)."""
    if isinstance(ckpt, dict) and 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
        return ckpt['state_dict'], {k: v for k, v in ckpt.items() if k != 'state_dict'}, True
    return ckpt, {}, False


def deep_equal(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and a.dtype == b.dtype and np.array_equal(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:
        return False


def make_patch(base, repaired, base_path=None):
    """base: path (recommended, recorded in the patch) or loaded checkpoint (then give base_path)."""
    if isinstance(base, (str, os.PathLike)):
        base_path = base
    if base_path is None:
        raise ValueError('base_path is required to record which checkpoint the patch applies to')
    base_ckpt, rep_ckpt = _load(base), _load(repaired)
    b_sd, b_top, _ = _split(base_ckpt)
    r_sd, r_top, r_wrapped = _split(rep_ckpt)

    tensors, n_changed = {}, 0
    for k, v in r_sd.items():
        if k not in b_sd:
            continue
        o = b_sd[k]
        if (isinstance(v, torch.Tensor) and isinstance(o, torch.Tensor)
                and v.dtype == o.dtype and v.shape == o.shape):
            if torch.equal(v, o):
                continue
            # bitwise comparison, so NaN payloads and -0.0 are handled exactly
            if v.is_floating_point():
                iv = v.contiguous().view(-1).view(torch.int32 if v.element_size() == 4 else
                                                  torch.int16 if v.element_size() == 2 else torch.int64)
                io = o.contiguous().view(-1).view(iv.dtype)
                idx = (iv != io).nonzero().squeeze(1)
            else:
                idx = (v.contiguous().view(-1) != o.contiguous().view(-1)).nonzero().squeeze(1)
            if idx.numel() <= _SPARSE_MAX_FRAC * v.numel():
                tensors[k] = {'idx': idx.to(torch.int64), 'values': v.contiguous().view(-1)[idx].clone()}
            else:
                tensors[k] = {'idx': None, 'values': v.clone()}
            n_changed += idx.numel()
        elif not deep_equal(v, o):
            tensors[k] = {'idx': None, 'values': v}

    return {
        'format': FORMAT,
        'base_path': os.path.abspath(base_path),
        'base_size': os.path.getsize(base_path),
        'base_sha256': file_sha256(base_path),
        'wrapped': r_wrapped,
        'key_order': list(r_sd.keys()),
        'tensors': tensors,
        'added': {k: v for k, v in r_sd.items() if k not in b_sd},
        'top_level': {k: v for k, v in r_top.items() if not (k in b_top and deep_equal(v, b_top[k]))},
        'top_level_removed': [k for k in b_top if k not in r_top],
        'top_level_order': list(r_top.keys()),
        'n_changed_values': int(n_changed),
    }


def save_patch(patch, path):
    torch.save(patch, path)
    return path


def load_patched(patch, base_path=None, check_base=True):
    """Rebuild the full repaired checkpoint (same structure as the original .pth) in memory."""
    p = _load(patch)
    if p.get('format') != FORMAT:
        raise ValueError(f'not a {FORMAT} file')
    bp = base_path or p['base_path']
    if check_base:
        if os.path.getsize(bp) != p['base_size'] or file_sha256(bp) != p['base_sha256']:
            raise RuntimeError(f'base checkpoint {bp} does not match the one this patch was made from')
    base_ckpt = torch.load(bp, map_location='cpu')
    b_sd, b_top, _ = _split(base_ckpt)

    sd = {}
    for k in p['key_order']:
        if k in p['added']:
            sd[k] = p['added'][k]
        elif k in p['tensors']:
            t = p['tensors'][k]
            if t['idx'] is None:
                sd[k] = t['values']
            else:
                v = b_sd[k].clone()
                v.view(-1)[t['idx']] = t['values']
                sd[k] = v
        else:
            sd[k] = b_sd[k]
    if not p['wrapped']:
        return sd
    top = {}
    for k in p['top_level_order']:
        top[k] = p['top_level'][k] if k in p['top_level'] else b_top[k]
    top['state_dict'] = sd
    # keep the usual key order (meta, state_dict, optimizer, ...)
    order = list(base_ckpt.keys()) if isinstance(base_ckpt, dict) else []
    return {k: top[k] for k in order if k in top} | {k: v for k, v in top.items() if k not in order}


def materialize(patch, out_path, base_path=None):
    """Write the full checkpoint to out_path (e.g. on local disk) for code that needs a .pth path."""
    torch.save(load_patched(patch, base_path=base_path), out_path)
    return out_path


def is_patch(path):
    return str(path).endswith('.patch.pt')


@contextmanager
def materialized(path, tmp_dir=None):
    """Yield a path to a full checkpoint: `path` itself, or for a patch a temporary
    full .pth on local disk (E2EREP_CKPT_TMP or the system temp dir), removed on exit."""
    path = str(path)
    if not is_patch(path):
        yield path
        return
    d = tmp_dir or os.environ.get('E2EREP_CKPT_TMP') or tempfile.gettempdir()
    fd, out = tempfile.mkstemp(prefix='repaired_', suffix='.pth', dir=d)
    os.close(fd)
    try:
        materialize(path, out)
        yield out
    finally:
        if os.path.exists(out):
            os.remove(out)


def save_repaired(base_path, repaired_ckpt, out_dir, stem, save_full=False):
    """Save a repaired checkpoint as <stem>.patch.pt (verified lossless) and, if
    save_full, also as <stem>.pth. Returns the patch path."""
    def cpu(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        if isinstance(x, dict):
            return {k: cpu(v) for k, v in x.items()}
        return x
    repaired_ckpt = cpu(repaired_ckpt)
    patch_path = os.path.join(str(out_dir), stem + '.patch.pt')
    save_patch(make_patch(base_path, repaired_ckpt), patch_path)
    if not deep_equal(_split(load_patched(patch_path))[0], _split(repaired_ckpt)[0]):
        raise RuntimeError(f'patch {patch_path} does not rebuild the repaired model')
    if save_full:
        torch.save(repaired_ckpt, os.path.join(str(out_dir), stem + '.pth'))
    return patch_path


if __name__ == '__main__':
    # Rebuild a full checkpoint from a patch, e.g. for test.py or a CARLA agent:
    #   python repair_common/ckpt_patch.py <stem>.patch.pt /local/disk/out.pth
    import sys
    if len(sys.argv) != 3:
        sys.exit('usage: python repair_common/ckpt_patch.py <patch.pt> <out.pth>')
    print(materialize(sys.argv[1], sys.argv[2]))
