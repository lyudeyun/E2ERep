#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute a correlation matrix (Pearson / Spearman) and plot a paper-style figure.

Input
-----
- **Embedded data (default)**: edit `ACTIVE_DATASET` and `EMBEDDED_DATASETS`.
- **External CSV/TSV**: provide `--input` and `--metrics`.

Typical usage (6 metrics -> 6x6 matrix)
--------------------------------------
# 1) Embedded data -> heatmap
python3 mytools/corr_heatmap.py --out-fig corr_heatmap.pdf

# 2) Embedded data -> heatmap + upper-triangle scatter/fit (pairgrid)
python3 mytools/corr_heatmap.py --pairgrid --out-fig corr_pairgrid.pdf

# 3) External table
python3 mytools/corr_heatmap.py \
  --input metrics.tsv --sep $'\\t' \
  --metrics "Avg. L2 Error,Collision,Driving Score,Success Rate,Efficiency,Comfortness" \
  --out-csv corr_matrix.csv \
  --out-fig corr_heatmap.pdf

Notes
-----
- `--method`: Pearson r or Spearman rho (rank-based), computed pairwise with deletion of missing values.
- `--pairgrid`: lower triangle shows correlations, upper triangle shows scatter + fitted line.
- p-values (`--annotate-p`): Student-t approximation on r/rho (requires SciPy for t CDF).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import matplotlib as mpl
except ModuleNotFoundError as e:
    raise SystemExit(
        "缺少绘图依赖。请先安装：\n"
        "  python3 -m pip install -U matplotlib\n"
        "或使用系统包：python3-matplotlib\n"
    ) from e


# ============================================================================
# Embedded config (optional)
# - 如果你不想用外部文件，可以把数据直接写在这里，然后用 `--use-embedded` 运行。
# - 推荐优先用 `EMBEDDED_VECTORS`（每个 metric 一个向量），更符合“6个metric各自一串数”的用法。
# ============================================================================
EMBEDDED_METRICS: List[str] = [
    "Avg. L2 Error",
    "Collision",
    "Driving Score",
    "Success Rate",
    "Efficiency",
    "Comfortness",
]

# Pretty display labels for plots only (mathtext supported by matplotlib).
DISPLAY_LABELS: Dict[str, str] = {
    # Make subscript upright (roman), not italic.
    "Avg. L2 Error": r"$\mu_{\mathrm{L2\_Err}}$",
    # Use plain text to avoid mathtext parsing issues in PDF backend.
    "Collision": "#colls",
}


def _display_metric(name: str) -> str:
    return DISPLAY_LABELS.get(name, name)

# Embedded datasets.
# Pick one by editing ACTIVE_DATASET (no need to pass extra CLI args).
ACTIVE_DATASET = "uniad"  # one of: "uniad", "vad"

EMBEDDED_DATASETS: Dict[str, Dict[str, List[Optional[float]]]] = {
    # UniAD repaired (10 runs) you provided earlier.
    "uniad": {
        "Avg. L2 Error": [
            1.2519,
            1.2603,
            1.2498,
            1.2518,
            1.2510,
            1.2504,
            1.2486,
            1.2504,
            1.2490,
            1.2491,
        ],
        "Collision": [
            592,
            509,
            569,
            584,
            555,
            574,
            516,
            545,
            580,
            565,
        ],
        "Driving Score": [
            39.38,
            49.14,
            41.52,
            39.75,
            40.37,
            50.35,
            40.04,
            49.26,
            52.38,
            41.51,
        ],
        "Success Rate": [
            10.0,
            30.0,
            10.0,
            10.0,
            20.0,
            30.0,
            20.0,
            20.0,
            30.0,
            20.0,
        ],
        "Efficiency": [
            95.04,
            99.35,
            96.43,
            95.46,
            91.55,
            93.70,
            93.24,
            99.71,
            95.02,
            88.05,
        ],
        "Comfortness": [
            34.92,
            36.65,
            37.88,
            39.40,
            47.94,
            36.00,
            41.26,
            35.94,
            35.53,
            42.64,
        ],
    },
    # VAD repaired (10 runs)
    "vad": {
        "Avg. L2 Error": [
            1.4066,
            1.4052,
            1.3890,
            1.3974,
            1.3888,
            1.3871,
            1.3953,
            1.4077,
            1.3870,
            1.3912,
        ],
        "Collision": [
            80,
            71,
            66,
            76,
            71,
            77,
            77,
            68,
            66,
            70,
        ],
        "Driving Score": [
            48.31,
            27.70,
            39.45,
            38.33,
            43.19,
            39.14,
            36.27,
            43.97,
            26.42,
            33.99,
        ],
        "Success Rate": [
            20.0,
            0.0,
            20.0,
            20.0,
            10.0,
            20.0,
            10.0,
            20.0,
            0.0,
            0.0,
        ],
        "Efficiency": [
            105.82,
            95.01,
            95.29,
            104.70,
            90.65,
            93.86,
            105.71,
            86.99,
            98.75,
            96.92,
        ],
        "Comfortness": [
            37.04,
            40.53,
            43.70,
            39.89,
            41.15,
            37.97,
            40.89,
            42.91,
            45.79,
            44.90,
        ],
    },
}

# Option B: row-wise dicts (kept for compatibility; unused by default).
EMBEDDED_ROWS: List[Dict[str, Optional[float]]] = []


def _parse_metrics_arg(s: str) -> List[str]:
    # allow comma-separated or repeated --metrics
    parts = [x.strip() for x in s.split(",")]
    return [p for p in parts if p]


def _try_float(x: str) -> Optional[float]:
    x = x.strip()
    if x == "" or x.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def load_table(path: Path, sep: str, metrics: Sequence[str]) -> Dict[str, List[Optional[float]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=sep)
        if reader.fieldnames is None:
            raise ValueError("输入文件没有 header 行，无法按列名读取。")

        missing = [m for m in metrics if m not in reader.fieldnames]
        if missing:
            raise ValueError(f"输入文件缺少这些列：{missing}\n可用列名：{reader.fieldnames}")

        cols: Dict[str, List[Optional[float]]] = {m: [] for m in metrics}
        for row in reader:
            for m in metrics:
                cols[m].append(_try_float(row.get(m, "")))
    return cols


def load_embedded(metrics: Sequence[str]) -> Dict[str, List[Optional[float]]]:
    # Prefer embedded dataset vectors if provided
    if EMBEDDED_DATASETS:
        if ACTIVE_DATASET not in EMBEDDED_DATASETS:
            raise ValueError(f"ACTIVE_DATASET={ACTIVE_DATASET} 不存在，可选：{list(EMBEDDED_DATASETS.keys())}")
        vectors = EMBEDDED_DATASETS[ACTIVE_DATASET]
        missing = [m for m in metrics if m not in vectors]
        if missing:
            raise ValueError(f"EMBEDDED_DATASETS[{ACTIVE_DATASET}] 缺少这些 metric：{missing}")
        lengths = {m: len(vectors[m]) for m in metrics}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"EMBEDDED_DATASETS[{ACTIVE_DATASET}] 各向量长度不一致：{lengths}")
        cols: Dict[str, List[Optional[float]]] = {}
        for m in metrics:
            cols[m] = [None if v is None else float(v) for v in vectors[m]]
        return cols

    # Fallback to rows
    if EMBEDDED_ROWS:
        cols2: Dict[str, List[Optional[float]]] = {m: [] for m in metrics}
        for row in EMBEDDED_ROWS:
            for m in metrics:
                v = row.get(m, None)
                cols2[m].append(None if v is None else float(v))
        return cols2

    raise ValueError(
        "你启用了 --use-embedded，但脚本里的 EMBEDDED_DATASETS/EMBEDDED_ROWS 都是空的。\n"
        "请把数据填进 EMBEDDED_DATASETS（推荐），或填 EMBEDDED_ROWS。"
    )


def pearson_r(x: Sequence[Optional[float]], y: Sequence[Optional[float]]) -> Tuple[float, int]:
    # Pairwise deletion
    xs: List[float] = []
    ys: List[float] = []
    for a, b in zip(x, y):
        if a is None or b is None:
            continue
        xs.append(float(a))
        ys.append(float(b))

    n = len(xs)
    if n < 2:
        return float("nan"), n

    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    denx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    deny = math.sqrt(sum((b - my) ** 2 for b in ys))
    if denx == 0.0 or deny == 0.0:
        return float("nan"), n
    return num / (denx * deny), n


def _rankdata_average(values: Sequence[float]) -> List[float]:
    """Average ranks for ties; ranks are 1..n (same convention as SciPy rankdata average)."""
    n = len(values)
    if n == 0:
        return []
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        v = values[indexed[i]]
        while j < n and values[indexed[j]] == v:
            j += 1
        # positions i..j-1 share average rank (1-based)
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j
    return ranks


def spearman_r(x: Sequence[Optional[float]], y: Sequence[Optional[float]]) -> Tuple[float, int]:
    """Spearman rho: Pearson correlation of average ranks (pairwise deletion)."""
    xs: List[float] = []
    ys: List[float] = []
    for a, b in zip(x, y):
        if a is None or b is None:
            continue
        xs.append(float(a))
        ys.append(float(b))
    n = len(xs)
    if n < 2:
        return float("nan"), n
    rx = _rankdata_average(xs)
    ry = _rankdata_average(ys)
    return pearson_r(rx, ry)  # type: ignore[arg-type]


def pearson_p_value(r: float, n: int) -> Optional[float]:
    """
    Two-sided p-value for Pearson correlation coefficient.
    Requires SciPy for accurate Student-t CDF.
    """
    if n < 3 or not math.isfinite(r):
        return None
    try:
        from scipy.stats import t as student_t  # type: ignore
    except ModuleNotFoundError:
        return None

    df = n - 2
    # Guard numerical issues when r is extremely close to ±1
    r = max(min(r, 0.999999999), -0.999999999)
    t = abs(r) * math.sqrt(df / (1.0 - r * r))
    p = 2.0 * (1.0 - student_t.cdf(t, df))
    return float(p)


def spearman_p_value(rho: float, n: int) -> Optional[float]:
    """
    p-value for Spearman rho. Without a full SciPy paired call here, we use the same
    Student-t approximation as Pearson applied to rho (common in small samples; exact
    tables differ slightly).
    """
    return pearson_p_value(rho, n)


def corr_matrix(
    cols: Dict[str, Sequence[Optional[float]]],
    metrics: Sequence[str],
    method: str = "pearson",
) -> Tuple[List[List[float]], List[List[int]], List[List[Optional[float]]]]:
    m = len(metrics)
    rmat: List[List[float]] = [[float("nan")] * m for _ in range(m)]
    nmat: List[List[int]] = [[0] * m for _ in range(m)]
    pmat: List[List[Optional[float]]] = [[None] * m for _ in range(m)]

    if method not in {"pearson", "spearman"}:
        raise ValueError("method 必须是 pearson 或 spearman")

    for i in range(m):
        for j in range(m):
            if method == "pearson":
                r, n = pearson_r(cols[metrics[i]], cols[metrics[j]])
                pmat[i][j] = pearson_p_value(r, n)
            else:
                r, n = spearman_r(cols[metrics[i]], cols[metrics[j]])
                pmat[i][j] = spearman_p_value(r, n)
            rmat[i][j] = r
            nmat[i][j] = n
    # force diagonal to 1.0 for readability (self-correlation)
    for k in range(m):
        rmat[k][k] = 1.0
        nmat[k][k] = len([v for v in cols[metrics[k]] if v is not None])
        pmat[k][k] = None
    return rmat, nmat, pmat


def save_corr_csv(metrics: Sequence[str], rmat: Sequence[Sequence[float]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + list(metrics))
        for name, row in zip(metrics, rmat):
            w.writerow([name] + ["" if not math.isfinite(v) else f"{v:.6f}" for v in row])


def plot_heatmap(
    cols: Optional[Dict[str, Sequence[Optional[float]]]],
    metrics: Sequence[str],
    rmat: Sequence[Sequence[float]],
    pmat: Sequence[Sequence[Optional[float]]],
    out_fig: Path,
    title: str,
    annotate: bool,
    annotate_p: bool,
    triangle: str,
    cell_fontsize: int,
    tick_fontsize: int,
    show_colorbar: bool,
    upper_scatter: bool,
) -> None:
    m = len(metrics)
    data = [[(float("nan") if not math.isfinite(v) else float(v)) for v in row] for row in rmat]

    # mask to triangle like the example figure
    if triangle not in {"full", "lower", "upper"}:
        raise ValueError("triangle 必须是 full/lower/upper")
    if triangle != "full":
        for i in range(m):
            for j in range(m):
                if triangle == "lower" and j > i:
                    data[i][j] = float("nan")
                if triangle == "upper" and i > j:
                    data[i][j] = float("nan")

    # Colormap: fixed to RdBu_r (paper-style; +1 red, -1 blue, white center)
    cmap = mpl.colormaps.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#ffffff")  # masked/NaN as white, like seaborn heatmap masks

    fig, ax = plt.subplots(figsize=(1.1 * m + 3.0, 1.0 * m + 2.6))
    # show NaNs as white (masked)
    masked = [[v for v in row] for row in data]
    # Match seaborn: center=0, vmin=-1, vmax=1 (symmetric diverging)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(masked, norm=norm, cmap=cmap, interpolation="nearest")
    ax.set_facecolor("white")
    # Make each matrix cell square in the exported PDF.
    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks(range(m))
    ax.set_yticks(range(m))
    xt = [_display_metric(mn) for mn in metrics]
    yt = [_display_metric(mn) for mn in metrics]
    ax.set_xticklabels(xt, rotation=45, ha="right", fontsize=tick_fontsize)
    ax.set_yticklabels(yt, fontsize=tick_fontsize)
    # Labels only on left and bottom (like common paper figures)
    ax.tick_params(
        axis="both",
        which="both",
        top=False,
        labeltop=False,
        right=False,
        labelright=False,
        bottom=True,
        labelbottom=True,
        left=True,
        labelleft=True,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Minor gridlines for a cleaner scientific look
    ax.set_xticks([x - 0.5 for x in range(1, m)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, m)], minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.6, alpha=0.10)
    ax.tick_params(which="minor", bottom=False, left=False)

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Default: no label text (paper-friendly minimal style)
        cbar.ax.tick_params(labelsize=tick_fontsize)

    if annotate:
        for i in range(m):
            for j in range(m):
                if not math.isfinite(data[i][j]):
                    continue
                r = rmat[i][j]
                if not math.isfinite(r):
                    text = "NA"
                else:
                    text = f"{r:.2f}"
                if annotate_p and pmat[i][j] is not None:
                    p = pmat[i][j]
                    if p is not None:
                        if p < 0.001:
                            text += "\n***"
                        elif p < 0.01:
                            text += "\n**"
                        elif p < 0.05:
                            text += "\n*"
                ax.text(j, i, text, ha="center", va="center", fontsize=cell_fontsize, color="#111827")

    # Optional: overlay scatter+fit in the upper triangle, while keeping the same base heatmap style.
    # This makes toggling scatter on/off not change the overall figure layout.
    if upper_scatter:
        if cols is None:
            raise ValueError("upper_scatter=True 需要提供 cols（每个 metric 的向量）")
        # upper triangle only
        for i in range(m):
            for j in range(m):
                if j <= i:
                    continue
                # If triangle masking already hides upper, we still overlay scatter on top.
                # Create an inset axes occupying the (i,j) cell in axes-fraction coordinates.
                x0 = j / m
                y0 = 1.0 - (i + 1) / m
                w = 1.0 / m
                h = 1.0 / m
                iax = ax.inset_axes([x0, y0, w, h], transform=ax.transAxes)
                iax.set_facecolor("white")
                iax.set_xticks([])
                iax.set_yticks([])
                # scatter: x=metric[j], y=metric[i]
                x, y = _drop_none_pairs(cols[metrics[j]], cols[metrics[i]])
                if x and y:
                    iax.scatter(x, y, s=10, color="#9ca3af", alpha=0.7, linewidths=0)
                    fit = _linear_fit(x, y)
                    if fit is not None:
                        a, b = fit
                        xmin, xmax = min(x), max(x)
                        iax.plot([xmin, xmax], [a * xmin + b, a * xmax + b], color="#991b1b", linewidth=2.0, alpha=0.9)
                # Keep a thin frame like the cell boundary
                for spine in iax.spines.values():
                    spine.set_visible(False)

    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _drop_none_pairs(
    x: Sequence[Optional[float]], y: Sequence[Optional[float]]
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for a, b in zip(x, y):
        if a is None or b is None:
            continue
        xs.append(float(a))
        ys.append(float(b))
    return xs, ys


def _linear_fit(xs: Sequence[float], ys: Sequence[float]) -> Optional[Tuple[float, float]]:
    # y = a*x + b
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    b = my - a * mx
    return a, b


def _lowess(
    xs: Sequence[float],
    ys: Sequence[float],
    frac: float = 0.66,
) -> Optional[Tuple[List[float], List[float]]]:
    """
    Simple LOWESS/LOESS smoother (local linear regression with tricube weights).
    Returns (x_sorted, y_smooth) evaluated at the observed x locations.
    No external dependencies.
    """
    n = len(xs)
    if n < 3:
        return None
    x = [float(v) for v in xs]
    y = [float(v) for v in ys]
    order = sorted(range(n), key=lambda i: x[i])
    x_sorted = [x[i] for i in order]
    y_sorted = [y[i] for i in order]

    k = max(2, int(math.ceil(frac * n)))
    yhat: List[float] = []
    for i in range(n):
        xi = x_sorted[i]
        dists = [abs(xj - xi) for xj in x_sorted]
        # bandwidth: distance to k-th nearest point
        h = sorted(dists)[k - 1]
        if h == 0.0:
            # all points identical x: fallback to mean
            yhat.append(sum(y_sorted) / n)
            continue

        # tricube weights
        w = []
        for dj in dists:
            u = dj / h
            if u >= 1.0:
                w.append(0.0)
            else:
                w.append((1.0 - u**3) ** 3)

        sw = sum(w)
        if sw == 0.0:
            yhat.append(sum(y_sorted) / n)
            continue

        # weighted local linear regression around xi:
        # minimize sum wj * (aj + bj*xj - yj)^2
        xw = sum(wj * xj for wj, xj in zip(w, x_sorted))
        yw = sum(wj * yj for wj, yj in zip(w, y_sorted))
        xxw = sum(wj * xj * xj for wj, xj in zip(w, x_sorted))
        xyw = sum(wj * xj * yj for wj, xj, yj in zip(w, x_sorted, y_sorted))

        den = sw * xxw - xw * xw
        if den == 0.0:
            # fallback to weighted mean
            yhat.append(yw / sw)
            continue
        b = (sw * xyw - xw * yw) / den
        a = (yw - b * xw) / sw
        yhat.append(a + b * xi)

    return x_sorted, yhat


def plot_paircorr(
    cols: Dict[str, Sequence[Optional[float]]],
    metrics: Sequence[str],
    rmat: Sequence[Sequence[float]],
    pmat: Sequence[Sequence[Optional[float]]],
    out_fig: Path,
    title: str,
    annotate_p: bool,
    cell_fontsize: int,
    tick_fontsize: int,
    upper: str,
    show_colorbar: bool,
    method: str,
) -> None:
    """
    Paper-style pair plot:
    - diagonal: histogram
    - lower triangle: correlation colored cell with number
    - upper triangle: scatter + linear fit (optional)
    """
    m = len(metrics)

    # Colormap + norm (center=0, symmetric) - fixed to RdBu_r
    cmap = mpl.colormaps.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#ffffff")
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    # Manual axes placement to guarantee identical row/column gaps in figure coordinates.
    side = 2.0 * m + 1.6
    fig = plt.figure(figsize=(side, side))

    grid_left = 0.10
    grid_bottom = 0.10
    grid_side = 0.78  # square grid area
    gap = 0.007       # absolute gap between cells (same for x/y)
    cell = (grid_side - gap * (m - 1)) / m

    axes: List[List[plt.Axes]] = []
    for i in range(m):
        row: List[plt.Axes] = []
        for j in range(m):
            x0 = grid_left + j * (cell + gap)
            y0 = grid_bottom + (m - 1 - i) * (cell + gap)
            ax = fig.add_axes([x0, y0, cell, cell])
            row.append(ax)
        axes.append(row)

    for i in range(m):
        for j in range(m):
            ax = axes[i][j]
            ax.set_facecolor("white")
            is_bottom = i == m - 1
            is_left = j == 0

            # Outer labels only: left column + bottom row
            if i < m - 1:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            else:
                ax.tick_params(axis="x", bottom=True, labelbottom=True, labelsize=tick_fontsize, rotation=35)
            if j > 0:
                ax.tick_params(axis="y", left=False, labelleft=False)
            else:
                ax.tick_params(axis="y", left=True, labelleft=True, labelsize=tick_fontsize)

            if i == j:
                # Diagonal: keep consistent with correlation heatmap (r=1)
                ax.imshow([[1.0]], cmap=cmap, norm=norm, interpolation="nearest")
                ax.text(
                    0.5,
                    0.5,
                    "1.00",
                    ha="center",
                    va="center",
                    fontsize=cell_fontsize,
                    color="#111827",
                    transform=ax.transAxes,
                    fontweight="bold",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if is_bottom:
                    ax.set_xlabel(_display_metric(metrics[j]), fontsize=tick_fontsize, labelpad=10)
                if is_left:
                    ax.set_ylabel(_display_metric(metrics[i]), fontsize=tick_fontsize, labelpad=10)
                continue

            if i > j:
                # Lower triangle: correlation cell
                r = rmat[i][j]
                ax.imshow([[r]], cmap=cmap, norm=norm, interpolation="nearest")
                text = "NA" if not math.isfinite(r) else f"{r:.2f}"
                if annotate_p and pmat[i][j] is not None:
                    p = pmat[i][j]
                    if p is not None:
                        if p < 0.001:
                            text += "\n***"
                        elif p < 0.01:
                            text += "\n**"
                        elif p < 0.05:
                            text += "\n*"
                ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=cell_fontsize, color="#111827",
                        transform=ax.transAxes, fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])
                if is_bottom:
                    ax.set_xlabel(_display_metric(metrics[j]), fontsize=tick_fontsize, labelpad=10)
                if is_left:
                    ax.set_ylabel(_display_metric(metrics[i]), fontsize=tick_fontsize, labelpad=10)
                continue

            # Upper triangle: scatter + fit (optional)
            if upper == "blank":
                ax.axis("off")
                # keep outer labels if needed
                if is_bottom:
                        ax.set_xlabel(_display_metric(metrics[j]), fontsize=tick_fontsize, labelpad=10)
                if is_left:
                        ax.set_ylabel(_display_metric(metrics[i]), fontsize=tick_fontsize, labelpad=10)
                continue
            if upper != "scatter":
                raise ValueError("upper 必须是 scatter 或 blank")

            x, y = _drop_none_pairs(cols[metrics[j]], cols[metrics[i]])
            if x and y:
                ax.scatter(x, y, s=12, color="#9ca3af", alpha=0.7, linewidths=0)
                if method == "spearman":
                    smoothed = _lowess(x, y, frac=0.66)
                    if smoothed is not None:
                        xs_s, ys_s = smoothed
                        # Use correlation sign (rho) to color trend, consistent with colormap extremes.
                        rho = rmat[i][j]
                        line_color = cmap(norm(1.0)) if rho >= 0 else cmap(norm(-1.0))
                        ax.plot(xs_s, ys_s, color=line_color, linewidth=2.2, alpha=0.95)
                else:
                    fit = _linear_fit(x, y)
                    if fit is not None:
                        a, b = fit
                        xmin, xmax = min(x), max(x)
                        xs_line = [xmin, xmax]
                        ys_line = [a * xmin + b, a * xmax + b]
                        # Fit line color: positive slope -> +1 color (red), negative slope -> -1 color (blue)
                        line_color = cmap(norm(1.0)) if a >= 0 else cmap(norm(-1.0))
                        ax.plot(xs_line, ys_line, color=line_color, linewidth=2.2, alpha=0.95)
            ax.grid(False)
            if is_bottom:
                ax.set_xlabel(_display_metric(metrics[j]), fontsize=tick_fontsize, labelpad=10)
            if is_left:
                ax.set_ylabel(_display_metric(metrics[i]), fontsize=tick_fontsize, labelpad=10)

    if title:
        fig.suptitle(title, fontsize=tick_fontsize + 1, fontweight="bold", y=1.01)

    # One shared colorbar for correlation cells
    if show_colorbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        # Colorbar height matches the square grid height.
        cax = fig.add_axes([grid_left + grid_side + 0.025, grid_bottom, 0.025, grid_side])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize=tick_fontsize)

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Correlation matrix (Pearson or Spearman) + heatmap for selected metrics.")
    p.add_argument(
        "--use-embedded",
        action="store_true",
        default=True,
        help="Use embedded metrics + embedded data in this script (default: True).",
    )
    p.add_argument("--input", default=None, help="Input CSV/TSV file path (ignored if --use-embedded).")
    p.add_argument("--sep", default="\\t", help="Delimiter. Default is tab (\\\\t).")
    p.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metric column names. If omitted with --use-embedded, uses EMBEDDED_METRICS.",
    )
    p.add_argument("--out-csv", default="corr_matrix.csv", help="Output correlation matrix CSV.")
    p.add_argument("--out-fig", default="corr_heatmap.pdf", help="Output heatmap figure (pdf/png).")
    p.add_argument("--title", default="", help="Figure title.")
    p.add_argument(
        "--annotate",
        action="store_true",
        default=True,
        help="Annotate each cell with r (and stars if p available). (default: True)",
    )
    p.add_argument("--annotate-p", action="store_true", help="Add significance stars if SciPy is available.")
    p.add_argument("--cell-fontsize", type=int, default=23, help="Font size for numbers inside cells.")
    p.add_argument("--tick-fontsize", type=int, default=19, help="Font size for axis tick labels and colorbar.")
    p.add_argument(
        "--no-colorbar",
        action="store_true",
        help="Disable the right-side colorbar.",
    )
    p.add_argument(
        "--triangle",
        default="lower",
        choices=["lower", "upper", "full"],
        help="Heatmap shape: lower/upper triangle (like papers) or full matrix. Default: lower.",
    )
    p.add_argument(
        "--pairgrid",
        action="store_true",
        help="Render a pair-plot style figure: diag hist, lower corr cells, upper scatter+fit.",
    )
    p.add_argument(
        "--upper",
        default="scatter",
        choices=["scatter", "blank"],
        help="(pairgrid only) Upper triangle content: scatter+fit or blank.",
    )
    p.add_argument(
        "--method",
        default="pearson",
        choices=["pearson", "spearman"],
        help="Correlation coefficient: Pearson r or Spearman rho (rank-based).",
    )
    args = p.parse_args()

    sep = args.sep.encode("utf-8").decode("unicode_escape")  # allow '\\t'
    # If user provides --input explicitly, prefer file mode.
    use_embedded = args.use_embedded and not args.input
    if use_embedded:
        metrics = EMBEDDED_METRICS if not args.metrics else _parse_metrics_arg(args.metrics)
        if len(metrics) < 2:
            raise SystemExit("metrics 至少需要 2 个列名。")
        cols = load_embedded(metrics)
    else:
        if not args.input:
            raise SystemExit("请提供 --input，或使用默认内置数据（不传 --input 即可）。")
        if not args.metrics:
            raise SystemExit("未启用 --use-embedded 时，必须提供 --metrics。")
        metrics = _parse_metrics_arg(args.metrics)
        if len(metrics) < 2:
            raise SystemExit("--metrics 至少需要 2 个列名。")
        cols = load_table(Path(args.input), sep=sep, metrics=metrics)

    method = str(args.method)
    rmat, _nmat, pmat = corr_matrix(cols, metrics, method=method)

    out_csv = Path(args.out_csv)
    if method == "spearman" and out_csv.name == "corr_matrix.csv":
        out_csv = out_csv.with_name("corr_matrix_spearman.csv")
    save_corr_csv(metrics, rmat, out_csv)
    # Keep old pairgrid mode, but the recommended way is to overlay upper scatter on the same heatmap layout.
    if args.pairgrid:
        plot_paircorr(
            cols=cols,
            metrics=metrics,
            rmat=rmat,
            pmat=pmat,
            out_fig=Path(args.out_fig),
            title=args.title,
            annotate_p=bool(args.annotate_p),
            cell_fontsize=int(args.cell_fontsize),
            tick_fontsize=int(args.tick_fontsize),
            upper=str(args.upper),
            show_colorbar=not bool(args.no_colorbar),
            method=method,
        )
    else:
        plot_heatmap(
            cols=cols,
            metrics=metrics,
            rmat=rmat,
            pmat=pmat,
            out_fig=Path(args.out_fig),
            title=args.title,
            annotate=args.annotate,
            annotate_p=args.annotate_p,
            triangle=args.triangle,
            cell_fontsize=int(args.cell_fontsize),
            tick_fontsize=int(args.tick_fontsize),
            show_colorbar=not bool(args.no_colorbar),
            upper_scatter=bool(args.upper == "scatter") and (str(args.triangle) == "lower"),
        )

    print(f"Saved: {out_csv}")
    print(f"Saved: {args.out_fig}")


if __name__ == "__main__":
    main()

