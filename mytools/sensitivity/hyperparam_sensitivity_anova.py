#!/usr/bin/env python3
"""
Hyperparameter sensitivity: ART-ANOVA (Type III) of the 2x3x3 full factorial design.

Non-parametric variance decomposition over method, fitness and weight and their interactions.
The implementation follows the local ARTool sources: mytools/sensitivity/ARTool/R/ (art.R, internal.R, anova.art.R).

By default it writes 4 separate ANOVA tables (not one combined table):
  - vad_l2_3s_anova_art.csv
  - vad_collision_3s_anova_art.csv
  - uniad_l2_3s_anova_art.csv
  - uniad_collision_3s_anova_art.csv

--combined also writes hyperparam_sensitivity_anova_all.csv (the 4 tables stacked, for Excel).
--pooled also runs a 2x2x3x3 (model x method x fitness x weight_tier) ART-ANOVA on VAD + UniAD.

Usage:
  conda activate b2d_zoo
  python mytools/sensitivity/hyperparam_sensitivity_anova.py [--time-horizon 3s] [--out-dir DIR] [--pooled] [--data-root DIR]

The data are read as in calculate_cohensd.py; the collision metric is #colls (collision frames) for both models.
--validate-r cross-checks against the official ARTool R package in mytools/sensitivity/ARTool if R is installed.
--parametric uses a Type III OLS ANOVA of the raw responses (not ART; for comparison only).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Reuse the parsing and metric extraction of calculate_cohensd (mytools/stats)
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "stats")]
from art_transform import anova_art
from calculate_cohensd import (
    DATA_ROOT,
    count_collision_frames_vad_rule,
    count_vad_collision_from_json,
    extract_metrics_from_log,
    parse_experiment_name,
)

FACTORS = ["method", "fitness", "weight"]
POOLED_FACTORS = ["model", "method", "fitness", "weight_tier"]
METRIC_L2 = "plan_L2_3s"
METRIC_COLLISION = "collision_frame_count"

WEIGHT_TIER_BY_MODEL: dict[str, dict[int, str]] = {
    "VAD": {26: "small", 52: "medium", 105: "large"},
    "UniAD": {7: "small", 13: "medium", 26: "large"},
}


def get_data_root(data_root: Path | None = None) -> Path:
    if data_root is not None:
        return data_root
    return Path(DATA_ROOT)


def get_model_config(model_type: str, data_root: Path | None = None):
    root = get_data_root(data_root)
    if model_type == "vad":
        return {
            "base_dirs": [
                str(root / "vad_base_Arachne_v2_DE_results"),
                str(root / "vad_base_semSegRep_DE_results"),
            ],
            "folder_patterns": ["VAD_base_REP_VAL_*"],
            "log_name": lambda _name: "vad_base_rep_val.log",
            "collision_col": "plan_obj_box_col_3s",
            "label": "VAD",
        }
    return {
        "base_dirs": [
            str(root / "uniad_base_Arachne_v2_DE_results"),
            str(root / "uniad_base_semSegRep_DE_results"),
        ],
        "folder_patterns": ["UniAD_base_REP_VAL_*", "UniAD_tiny_REP_VAL_*"],
        "log_name": lambda name: (
            "uniad_tiny_rep_val.log"
            if ("UniAD_tiny" in name or "UniAD_tiny_REP" in name)
            else "uniad_base_rep_val.log"
        ),
        "collision_col": "collision_frame_count",
        "label": "UniAD",
    }


def _folder_time_horizon(folder_name: str) -> str | None:
    match = re.search(r"_t([123])s_", folder_name)
    return f"{match.group(1)}s" if match else None


def assign_weight_tier(model_label: str, weight_count: int) -> str | None:
    return WEIGHT_TIER_BY_MODEL.get(model_label, {}).get(int(weight_count))


def collect_experiment_data(
    model_type: str,
    time_horizon: str | None,
    data_root: Path | None = None,
) -> pd.DataFrame:
    """Scan the experiment folders and return a long-format DataFrame."""
    cfg = get_model_config(model_type, data_root)
    converter_script = (
        Path(__file__).resolve().parents[2] / "baseline" / "convert_uniad_to_vad_metrics.py"
    )
    records = []

    for base_dir in cfg["base_dirs"]:
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"Warning: {base_dir} not found")
            continue

        exp_folders = []
        search_root = base_path
        for pat in cfg["folder_patterns"]:
            for exp_folder in search_root.rglob(pat):
                if exp_folder.is_dir():
                    exp_folders.append(exp_folder)
        exp_folders = sorted(set(exp_folders))
        print(f"  [{cfg['label']}] {base_dir}: {len(exp_folders)} folders")

        for exp_folder in exp_folders:
            log_name = cfg["log_name"](exp_folder.name)
            log_file = exp_folder / "open_loop_eval" / log_name
            json_file = exp_folder / "open_loop_eval" / (Path(log_name).stem + ".json")

            if model_type == "vad":
                if not log_file.exists():
                    continue
                metrics = extract_metrics_from_log(log_file)
                open_loop_dir = exp_folder / "open_loop_eval"
                vad_json = json_file
                if not vad_json.exists() and open_loop_dir.exists():
                    json_candidates = sorted(open_loop_dir.glob("*.json"))
                    if len(json_candidates) == 1:
                        vad_json = json_candidates[0]
                if vad_json.exists() and metrics is not None:
                    _valid, cframes, _events = count_vad_collision_from_json(vad_json)
                    if cframes is not None:
                        metrics["collision_frame_count"] = cframes
            else:
                if not json_file.exists():
                    open_loop_dir = exp_folder / "open_loop_eval"
                    rep_val_candidates = []
                    all_json = []
                    if open_loop_dir.exists():
                        all_json = sorted(open_loop_dir.glob("*.json"))
                        rep_val_candidates = [p for p in all_json if "rep_val" in p.name]
                    if len(rep_val_candidates) == 1:
                        json_file = rep_val_candidates[0]
                    elif len(all_json) == 1:
                        json_file = all_json[0]
                    else:
                        continue
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".json", delete=False, encoding="utf-8"
                    ) as f_out:
                        out_json = f_out.name
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".txt", delete=False, encoding="utf-8"
                    ) as f_sum:
                        summary_path = f_sum.name
                    ret = subprocess.run(
                        [
                            sys.executable,
                            str(converter_script),
                            str(json_file),
                            out_json,
                            "--summary",
                            summary_path,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if ret.returncode != 0:
                        continue
                    metrics = extract_metrics_from_log(summary_path)
                    collision_count = count_collision_frames_vad_rule(out_json)
                    if metrics is not None and collision_count is not None:
                        metrics["collision_frame_count"] = collision_count
                    for p in (out_json, summary_path):
                        try:
                            os.unlink(p)
                        except OSError:
                            pass
                except Exception:
                    continue

            folder_horizon = _folder_time_horizon(exp_folder.name)
            if time_horizon and folder_horizon and folder_horizon != time_horizon:
                continue
            if time_horizon and folder_horizon is None and f"_t{time_horizon[0]}s_" not in exp_folder.name:
                continue

            method, weight_count, fitness_type, repetition = parse_experiment_name(
                exp_folder.name
            )
            if not all([method, weight_count, fitness_type, repetition]):
                continue
            if metrics is None:
                continue

            records.append(
                {
                    "model": cfg["label"],
                    "method": method,
                    "fitness": fitness_type,
                    "weight": str(weight_count),
                    "weight_count": int(weight_count),
                    "weight_tier": assign_weight_tier(cfg["label"], int(weight_count)),
                    "repetition": int(repetition),
                    "folder": exp_folder.name,
                    **metrics,
                }
            )

    return pd.DataFrame(records)


def collect_pooled_data(time_horizon: str | None, data_root: Path | None = None) -> pd.DataFrame:
    """Pool the VAD and UniAD observations for the model x method x fitness x weight_tier analysis."""
    frames = []
    for model_type in ("vad", "uniad"):
        df = collect_experiment_data(model_type, time_horizon, data_root)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    pooled = pd.concat(frames, ignore_index=True)
    missing_tier = pooled[pooled["weight_tier"].isna()]
    if not missing_tier.empty:
        bad = sorted(missing_tier["weight_count"].unique())
        print(f"  Warning: dropping {len(missing_tier)} rows with unmapped weight counts: {bad}")
        pooled = pooled[pooled["weight_tier"].notna()].copy()
    return pooled


def _sum_contrast_matrix(values: pd.Series) -> tuple[np.ndarray, list]:
    """Sum-to-zero contrasts (R contr.sum): k levels -> k-1 columns."""
    levels = sorted(values.unique(), key=lambda x: str(x))
    k = len(levels)
    level_to_idx = {lv: i for i, lv in enumerate(levels)}
    n = len(values)
    if k == 1:
        return np.zeros((n, 0)), levels
    X = np.zeros((n, k - 1), dtype=float)
    idx = values.map(level_to_idx).to_numpy()
    for j in range(k - 1):
        X[:, j] = np.where(idx == j, 1.0, np.where(idx == k - 1, -1.0, 0.0))
    return X, levels


def _term_name(factor_names: list[str]) -> str:
    return ":".join(factor_names)


def build_factorial_design(
    df: pd.DataFrame, factors: list[str]
) -> tuple[np.ndarray, dict[str, list[int]], int]:
    """
    Design matrix of the full interaction model (sum contrasts, no intercept).
    Returns X, term_col_indices, n_obs.
    """
    n = len(df)
    term_cols: dict[str, list[int]] = {}
    blocks: list[np.ndarray] = []
    col_offset = 0

    factor_mats: dict[str, np.ndarray] = {}
    for f in factors:
        mat, _ = _sum_contrast_matrix(df[f])
        factor_mats[f] = mat

    def _interaction_block(combo: tuple[str, ...]) -> np.ndarray:
        mats = [factor_mats[c] for c in combo]
        if len(mats) == 1:
            return mats[0]
        cols = [mats[0][:, j] for j in range(mats[0].shape[1])]
        for m in mats[1:]:
            new_cols = []
            for base in cols:
                for j in range(m.shape[1]):
                    new_cols.append(base * m[:, j])
            cols = new_cols
        return np.column_stack(cols) if cols else np.zeros((n, 0))

    for r in range(1, len(factors) + 1):
        for combo in combinations(factors, r):
            block = _interaction_block(combo)
            name = _term_name(list(combo))
            ncols = block.shape[1]
            if ncols == 0:
                continue
            blocks.append(block)
            term_cols[name] = list(range(col_offset, col_offset + ncols))
            col_offset += ncols

    X = np.column_stack(blocks) if blocks else np.zeros((n, 0))
    return X, term_cols, n


def _ols_sse(y: np.ndarray, X: np.ndarray) -> tuple[float, int, int]:
    """Least-squares fit; returns SSE, df_model (rank), df_resid."""
    if X.size == 0:
        mean_y = float(np.mean(y))
        sse = float(np.sum((y - mean_y) ** 2))
        return sse, 0, len(y) - 1
    coef, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    sse = float(np.sum((y - fitted) ** 2))
    df_resid = max(len(y) - rank, 1)
    return sse, int(rank), df_resid


def anova_type3(
    df: pd.DataFrame,
    response: str,
    factors: list[str] | None = None,
) -> pd.DataFrame:
    """
    Type III ANOVA (main effects and interactions as R car::Anova(type=III) with contr.sum).
    Each effect: the increase of SSE when its columns are dropped from the full model, divided by its df.
    """
    factors = factors or FACTORS
    sub = df[factors + [response]].dropna().copy()
    if len(sub) < 2:
        raise ValueError(f"Not enough data for {response}")

    y = sub[response].to_numpy(dtype=float)
    X_full, term_cols, _ = build_factorial_design(sub, factors)
    sse_full, rank_full, df_resid = _ols_sse(y, X_full)
    ms_error = sse_full / df_resid

    rows = []
    # Order of output: main effects -> two-way -> three-way interactions
    for r in range(1, len(factors) + 1):
        for combo in combinations(factors, r):
            term = _term_name(list(combo))
            if term not in term_cols:
                continue
            drop_idx = term_cols[term]
            keep_idx = [i for i in range(X_full.shape[1]) if i not in drop_idx]
            X_reduced = X_full[:, keep_idx] if keep_idx else np.zeros((len(y), 0))
            sse_reduced, _, _ = _ols_sse(y, X_reduced)
            ss_effect = max(sse_reduced - sse_full, 0.0)
            df_effect = len(drop_idx)
            if df_effect == 0:
                continue
            ms_effect = ss_effect / df_effect
            f_val = ms_effect / ms_error if ms_error > 0 else np.nan
            p_val = float(stats.f.sf(f_val, df_effect, df_resid)) if np.isfinite(f_val) else np.nan
            partial_eta2 = ss_effect / (ss_effect + sse_full) if (ss_effect + sse_full) > 0 else np.nan
            rows.append(
                {
                    "term": term,
                    "df": df_effect,
                    "sum_sq": ss_effect,
                    "mean_sq": ms_effect,
                    "F": f_val,
                    "p_value": p_val,
                    "partial_eta_sq": partial_eta2,
                    "df_resid": df_resid,
                    "n_obs": len(sub),
                }
            )

    result = pd.DataFrame(rows)
    return result


def cell_count_table(df: pd.DataFrame, factors: list[str] | None = None) -> pd.DataFrame:
    """Repetitions per cell (diagnoses an unbalanced design)."""
    factors = factors or FACTORS
    ct = (
        df.groupby(factors, dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(factors)
    )
    return ct


def run_art_anova_r(
    df: pd.DataFrame,
    response: str,
    out_csv: Path,
    factors: list[str] | None = None,
) -> bool:
    """If R + ARTool are available, run the ART-ANOVA in R and write a CSV."""
    factors = factors or FACTORS
    rscript = _find_rscript()
    if rscript is None:
        return False
    artool_dir = Path(__file__).resolve().parent / "ARTool"
    r_helper = Path(__file__).resolve().parent / "run_art_anova.R"
    if not r_helper.exists():
        return False

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        csv_path = f.name
        df[factors + [response]].dropna().to_csv(f, index=False)

    try:
        cmd = [
            rscript,
            str(r_helper),
            csv_path,
            response,
            str(out_csv),
            str(artool_dir),
        ]
        ret = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if ret.returncode != 0:
            print(f"  ART-ANOVA failed: {ret.stderr.strip()}")
            return False
        return True
    finally:
        try:
            os.unlink(csv_path)
        except OSError:
            pass


def _find_rscript() -> str | None:
    for cand in ("Rscript", "/usr/bin/Rscript"):
        try:
            ret = subprocess.run([cand, "--version"], capture_output=True, timeout=10)
            if ret.returncode == 0:
                return cand
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def format_anova_table(result: pd.DataFrame, model_label: str, metric_label: str) -> pd.DataFrame:
    out = result.copy()
    out.insert(0, "model", model_label)
    out.insert(1, "metric", metric_label)
    for col in ("sum_sq", "mean_sq", "F", "p_value", "partial_eta_sq"):
        if col in out.columns:
            out[col] = out[col].astype(float).round(6)
    return out


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sensitivity analysis (ART-ANOVA, Type III)")
    parser.add_argument(
        "--time-horizon",
        type=str,
        default="3s",
        help="read only this time horizon (default 3s)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="output directory (default mytools/sensitivity/)",
    )
    parser.add_argument(
        "--parametric",
        action="store_true",
        help="Type III OLS ANOVA of the raw responses (default: ART-ANOVA, as ARTool)",
    )
    parser.add_argument(
        "--validate-r",
        action="store_true",
        help="if R is installed, also run the official mytools/sensitivity/ARTool implementation and write *_anova_r.csv for comparison",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="also write hyperparam_sensitivity_anova_all.csv (the 4 tables combined; default: 4 separate tables)",
    )
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="also run the model x method x fitness x weight_tier ART-ANOVA on VAD + UniAD pooled",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=f"root of the experiment results (default: {DATA_ROOT}, as calculate_cohensd.py)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root) if args.data_root else None

    analyses = [
        ("vad", METRIC_L2, "l2_3s", "L2 error 3s"),
        ("vad", METRIC_COLLISION, "collision_3s", "collision frames 3s"),
        ("uniad", METRIC_L2, "l2_3s", "L2 error 3s"),
        ("uniad", METRIC_COLLISION, "collision_3s", "collision frames 3s"),
    ]

    print("=" * 80)
    mode = "parametric Type III OLS" if args.parametric else "ART-ANOVA Type III (ARTool port)"
    print(f"Hyperparameter sensitivity analysis ({mode})")
    print(f"time_horizon={args.time_horizon}")
    print(f"data_root={get_data_root(data_root)}")
    print("=" * 80)

    all_tables = []

    for model_type, metric_col, short_name, metric_label in analyses:
        cfg = get_model_config(model_type, data_root)
        print(f"\n--- {cfg['label']} / {metric_label} ---")
        df = collect_experiment_data(model_type, args.time_horizon, data_root)
        if df.empty:
            print("  No data; skip.")
            continue

        if metric_col not in df.columns:
            print(f"  Column {metric_col} missing; skip.")
            continue

        counts_path = out_dir / f"{model_type}_{short_name}_cell_counts.csv"
        cell_count_table(df).to_csv(counts_path, index=False)
        print(f"  Cell counts: {counts_path}")

        sub = df[FACTORS + [metric_col]].dropna()
        print(f"  Observations: {len(sub)} (unique cells: {sub.groupby(FACTORS).ngroups})")

        if args.parametric:
            out_csv = out_dir / f"{model_type}_{short_name}_anova_type3.csv"
            try:
                table = anova_type3(sub, metric_col)
            except ValueError as e:
                print(f"  Skip: {e}")
                continue
            table.to_csv(out_csv, index=False)
            print(f"  Parametric Type III table: {out_csv}")
        else:
            out_csv = out_dir / f"{model_type}_{short_name}_anova_art.csv"
            try:
                table = anova_art(sub, metric_col, FACTORS, anova_type3_fn=anova_type3)
            except ValueError as e:
                print(f"  Skip: {e}")
                continue
            table.to_csv(out_csv, index=False)
            print(f"  ART-ANOVA table: {out_csv}")

        if args.validate_r:
            r_csv = out_dir / f"{model_type}_{short_name}_anova_r.csv"
            if run_art_anova_r(sub, metric_col, r_csv):
                print(f"  R ARTool validation table: {r_csv}")
            else:
                print("  R ARTool validation skipped (R/ARTool not available).")

        formatted = format_anova_table(table, cfg["label"], metric_label)
        all_tables.append(formatted)

        # Main effects sorted by partial eta^2
        main_effects = formatted[formatted["term"].isin(FACTORS)].copy()
        if not main_effects.empty:
            main_effects = main_effects.sort_values("partial_eta_sq", ascending=False)
            print("  Main effects (by partial eta²):")
            for _, row in main_effects.iterrows():
                sig = "***" if row["p_value"] < 0.001 else (
                    "**" if row["p_value"] < 0.01 else (
                        "*" if row["p_value"] < 0.05 else ""
                    )
                )
                print(
                    f"    {row['term']:12s}  F={row['F']:.4f}  "
                    f"p={row['p_value']:.4g}  eta²_p={row['partial_eta_sq']:.4f} {sig}"
                )

    if args.pooled:
        print("\n" + "=" * 80)
        print("Pooled analysis (VAD + UniAD): model × method × fitness × weight_tier")
        print("=" * 80)
        pooled_df = collect_pooled_data(args.time_horizon, data_root)
        if pooled_df.empty:
            print("  No pooled data; skip.")
        else:
            pooled_analyses = [
                (METRIC_L2, "l2_3s", "L2 error 3s"),
                (METRIC_COLLISION, "collision_3s", "collision frames 3s"),
            ]
            for metric_col, short_name, metric_label in pooled_analyses:
                print(f"\n--- Pooled / {metric_label} ---")
                if metric_col not in pooled_df.columns:
                    print(f"  Column {metric_col} missing; skip.")
                    continue

                counts_path = out_dir / f"pooled_{short_name}_cell_counts.csv"
                cell_count_table(pooled_df, POOLED_FACTORS).to_csv(counts_path, index=False)
                print(f"  Cell counts: {counts_path}")

                sub = pooled_df[POOLED_FACTORS + [metric_col]].dropna()
                print(
                    f"  Observations: {len(sub)} "
                    f"(unique cells: {sub.groupby(POOLED_FACTORS).ngroups})"
                )

                if args.parametric:
                    out_csv = out_dir / f"pooled_{short_name}_anova_type3.csv"
                    try:
                        table = anova_type3(sub, metric_col, POOLED_FACTORS)
                    except ValueError as e:
                        print(f"  Skip: {e}")
                        continue
                    table.to_csv(out_csv, index=False)
                    print(f"  Parametric Type III table: {out_csv}")
                else:
                    out_csv = out_dir / f"pooled_{short_name}_anova_art.csv"
                    try:
                        table = anova_art(
                            sub, metric_col, POOLED_FACTORS, anova_type3_fn=anova_type3
                        )
                    except ValueError as e:
                        print(f"  Skip: {e}")
                        continue
                    table.to_csv(out_csv, index=False)
                    print(f"  ART-ANOVA table: {out_csv}")

                formatted = format_anova_table(table, "VAD+UniAD", metric_label)
                all_tables.append(formatted)

                main_effects = formatted[
                    formatted["term"].isin(POOLED_FACTORS)
                ].copy()
                if not main_effects.empty:
                    main_effects = main_effects.sort_values("partial_eta_sq", ascending=False)
                    print("  Main effects (by partial eta²):")
                    for _, row in main_effects.iterrows():
                        sig = "***" if row["p_value"] < 0.001 else (
                            "**" if row["p_value"] < 0.01 else (
                                "*" if row["p_value"] < 0.05 else ""
                            )
                        )
                        print(
                            f"    {row['term']:20s}  F={row['F']:.4f}  "
                            f"p={row['p_value']:.4g}  eta²_p={row['partial_eta_sq']:.4f} {sig}"
                        )

    if all_tables and args.combined:
        combined = pd.concat(all_tables, ignore_index=True)
        combined_path = out_dir / "hyperparam_sensitivity_anova_all.csv"
        combined.to_csv(combined_path, index=False)
        print("\n" + "=" * 80)
        print(f"Optional combined table: {combined_path}")
        print("=" * 80)
    elif all_tables:
        print("\n" + "=" * 80)
        print(f"Done: {len(all_tables)} ANOVA table(s) written (use --combined for one merged CSV).")
        print("=" * 80)
    else:
        print("\nNo ANOVA tables generated. Check the data paths under --data-root.")
        sys.exit(1)


if __name__ == "__main__":
    main()
