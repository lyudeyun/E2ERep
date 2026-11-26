"""Convert saved full repaired checkpoints into lossless patches, then optionally
delete the full files whose patch was verified.

Convert (never deletes) -- for each full .pth listed in --list:
  1. pick its base checkpoint: the one named in the run's repair.log if present,
     otherwise the candidate with the fewest differing tensors;
  2. write <stem>.patch.pt next to it (repair_common.ckpt_patch format);
  3. reload the patch from disk, rebuild the checkpoint, and compare it with the
     full file (every state_dict tensor bitwise, plus meta/optimizer);
  4. append one JSON line to --manifest (ok, sizes, base, n changed values).

  python mytools/ckpt/convert_full_ckpts.py --list full_ckpts.txt --manifest manifest.jsonl

Delete -- remove full files with ok=true in the manifest, after re-checking that
the patch and base still exist and the full file is unchanged. Dry run unless --yes:

  python mytools/ckpt/convert_full_ckpts.py --manifest manifest.jsonl --delete-verified [--yes]
"""
import argparse
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from repair_common.ckpt_patch import (_split, deep_equal, file_sha256, load_patched,  # noqa: E402
                                      make_patch, save_patch)

HOME_GIT = os.path.expanduser('~/git')
CANDIDATES = [
    f'{HOME_GIT}/E2ERep/Bench2DriveZoo/ckpts/vad_b2d_base.pth',
    f'{HOME_GIT}/E2ERep/Bench2DriveZoo/ckpts/uniad_base_b2d.pth',
    f'{HOME_GIT}/E2ERep/Bench2DriveZoo/ckpts/uniad_tiny_b2d.pth',
    f'{HOME_GIT}/VADRepair/models/VAD_base.pth',
    f'{HOME_GIT}/VADRepair/models/VAD_tiny.pth',
]
_base_cache = {}


def base_sd(path):
    if path not in _base_cache:
        if len(_base_cache) >= 5:
            _base_cache.pop(next(iter(_base_cache)))
        _base_cache[path] = _split(torch.load(path, map_location='cpu'))[0]
    return _base_cache[path]


def base_from_log(full_path):
    d = os.path.dirname(full_path)
    for _ in range(3):
        log = os.path.join(d, 'repair.log')
        if os.path.isfile(log):
            with open(log, errors='ignore') as f:
                for i, line in enumerate(f):
                    m = re.search(r'load checkpoint from (?:local )?path:\s*(\S+\.pth)', line)
                    if m:
                        return m.group(1)
                    if i > 200:
                        break
        d = os.path.dirname(d)
    return None


def n_diff(sd, bsd):
    if sd.keys() != bsd.keys():
        return 10 ** 9
    n = 0
    for k, v in sd.items():
        o = bsd[k]
        if not (isinstance(v, torch.Tensor) and isinstance(o, torch.Tensor)
                and v.shape == o.shape and v.dtype == o.dtype):
            return 10 ** 9
        n += not torch.equal(v, o)
    return n


def delete_verified(manifest, yes):
    recs = {}
    for line in open(manifest):
        r = json.loads(line)
        recs[r['full']] = r                      # last record per file wins
    todo, skipped, freed = [], [], 0
    for full, r in recs.items():
        if not r.get('ok'):
            skipped.append((full, 'not verified'))
        elif not os.path.isfile(full):
            skipped.append((full, 'already gone'))
        elif not (os.path.isfile(r['patch']) and os.path.getsize(r['patch']) == r['patch_bytes']):
            skipped.append((full, 'patch missing or changed'))
        elif os.path.getsize(full) != r['full_bytes']:
            skipped.append((full, 'full file changed since verification'))
        elif not os.path.isfile(r['base']):
            skipped.append((full, 'base checkpoint missing'))
        else:
            todo.append(full)
            freed += r['full_bytes']

    for full, why in skipped:
        print(f'skip ({why}): {full}')
    print(f'{len(todo)} files to delete, {freed / 1e9:.1f} GB; {len(skipped)} skipped')
    if not yes:
        print('dry run; pass --yes to delete')
        return
    for full in todo:
        os.remove(full)
    print(f'deleted {len(todo)} files')



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', help='text file, one full .pth path per line')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--delete-verified', action='store_true',
                    help='delete full files verified in the manifest instead of converting')
    ap.add_argument('--yes', action='store_true', help='really delete (default: dry run)')
    args = ap.parse_args()
    if args.delete_verified:
        return delete_verified(args.manifest, args.yes)
    if not args.list:
        ap.error('--list is required for conversion')

    done = set()
    if os.path.isfile(args.manifest):
        for line in open(args.manifest):
            r = json.loads(line)
            if r.get('ok'):
                done.add(r['full'])

    paths = [p.strip() for p in open(args.list) if p.strip()]
    for i, full in enumerate(paths, 1):
        if full in done:
            continue
        t0 = time.time()
        rec = {'full': full, 'ok': False}
        try:
            ckpt = torch.load(full, map_location='cpu')
            sd = _split(ckpt)[0]
            logged = base_from_log(full)
            cands = ([logged] if logged and os.path.isfile(logged) else []) + \
                [c for c in CANDIDATES if c != logged and os.path.isfile(c)]
            best, best_n = None, None
            for c in cands:
                n = n_diff(sd, base_sd(c))
                if best_n is None or n < best_n:
                    best, best_n = c, n
                if n <= 16:          # a near-identical base: no need to load the others
                    break
            if best is None or best_n >= 10 ** 9:
                raise RuntimeError('no candidate base with the same keys and shapes')
            patch = make_patch(best, ckpt)
            out = re.sub(r'\.pth$', '', full) + '.patch.pt'
            save_patch(patch, out)
            rebuilt = load_patched(out)                     # from disk, base hash checked
            ok = deep_equal(rebuilt, ckpt)
            rec.update(ok=bool(ok), patch=out, base=best, base_from_log=logged,
                       base_sha256=file_sha256(best), changed_tensors=len(patch['tensors']),
                       changed_values=patch['n_changed_values'],
                       top_level_changed=sorted(patch['top_level']),
                       full_bytes=os.path.getsize(full), patch_bytes=os.path.getsize(out),
                       seconds=round(time.time() - t0, 1))
        except Exception as e:  # keep going; the manifest records the failure
            rec['error'] = repr(e)
        with open(args.manifest, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(f"[{i}/{len(paths)}] {'OK ' if rec['ok'] else 'FAIL'} "
              f"{rec.get('changed_tensors', '-')}t/{rec.get('changed_values', '-')}v "
              f"{rec.get('patch_bytes', 0) / 1e3:.0f}kB {full} {rec.get('error', '')}", flush=True)


if __name__ == '__main__':
    main()
