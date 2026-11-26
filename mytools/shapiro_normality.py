#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shapiro–Wilk normality test for the 6 metrics in UniAD / VAD embedded datasets.

Usage
-----
python3 mytools/shapiro_normality.py

Notes
-----
- H0: samples come from a normal distribution.
- Small n (e.g., 10) means the test has low power; interpret cautiously.
"""

from __future__ import annotations

from typing import Dict, List


def _require_scipy():
    try:
        from scipy.stats import shapiro  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit(
            "scipy is required for the Shapiro-Wilk test.\n"
            "Install it in your conda environment: conda install -y scipy\n"
            "or: python -m pip install -U scipy\n"
        ) from e
    return shapiro


def main() -> None:
    # Import embedded data from corr_heatmap.py
    try:
        from mytools.corr_heatmap import EMBEDDED_DATASETS, EMBEDDED_METRICS  # type: ignore
    except Exception as e:
        raise SystemExit(f"cannot import the embedded data of mytools/corr_heatmap.py: {e}") from e

    shapiro = _require_scipy()

    alpha = 0.05
    datasets: Dict[str, Dict[str, List[float]]] = {}

    for ds_name in ["uniad", "vad"]:
        if ds_name not in EMBEDDED_DATASETS:
            continue
        ds = EMBEDDED_DATASETS[ds_name]
        datasets[ds_name] = {}
        for m in EMBEDDED_METRICS:
            vals = [v for v in ds[m] if v is not None]
            datasets[ds_name][m] = [float(v) for v in vals]

    for ds_name, ds in datasets.items():
        print("=" * 80)
        print(f"Dataset: {ds_name}")
        print(f"Shapiro–Wilk normality test (alpha={alpha})")
        print("-" * 80)
        for m in EMBEDDED_METRICS:
            x = ds[m]
            if len(x) < 3:
                print(f"{m:>15s} | n={len(x):2d} | skip (n<3)")
                continue
            W, p = shapiro(x)
            verdict = "fail to reject normality" if p >= alpha else "reject normality"
            print(f"{m:>15s} | n={len(x):2d} | W={W:.4f} | p={p:.4g} | {verdict}")


if __name__ == "__main__":
    main()

