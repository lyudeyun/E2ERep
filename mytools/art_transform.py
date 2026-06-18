"""
Python port of ARTool alignment / ranking (Wobbrock et al. 2011; Kay et al. ARTool).

Reference implementation:
  mytools/ARTool/R/art.R           — art(), residuals + aligned + rank
  mytools/ARTool/R/internal.R      — art.estimated.effects()
  mytools/ARTool/R/anova.art.R     — per-term Type III ANOVA on art(Y)
  mytools/ARTool/R/flat.anova.R    — Type III via car::Anova (default)

Procedure (mirrors art.R lines 216–226):
  1. cell means for each model term
  2. estimated effects (internal.R art.estimated.effects)
  3. residuals = y − cell_mean(highest-order interaction)
  4. aligned_term = residuals + estimated_effect_term
  5. art_term = rank(round(aligned_term, digits))

anova.art then fits Type III ANOVA on art_term for each term separately
and keeps the F-test row for that term only.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def _term_name(factors: list[str]) -> str:
    return ":".join(factors)


def all_model_terms(factors: list[str]) -> list[str]:
    """All fixed-effect terms in a full factorial model (main + interactions)."""
    terms: list[str] = []
    for r in range(1, len(factors) + 1):
        for combo in combinations(factors, r):
            terms.append(_term_name(list(combo)))
    return terms


def art_estimated_effects(
    df: pd.DataFrame,
    response: str,
    factors: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Port of ARTool::art.estimated.effects (internal.R).

    Returns
    -------
    cell_means : DataFrame, one column per term (+ grand mean column)
    estimated_effects : DataFrame, one column per term (no grand mean)
    residuals : ndarray, y − mean(highest-order cell)
    """
    factors = list(factors)
    terms = all_model_terms(factors)
    y = df[response].astype(float).to_numpy()
    n = len(y)

    n_f = len(factors)
    n_t = len(terms)
    factors_mat = np.zeros((n_f, n_t), dtype=int)
    for k, term in enumerate(terms):
        tf = set(term.split(":"))
        for i, f in enumerate(factors):
            if f in tf:
                factors_mat[i, k] = 1

    grand_col = np.zeros((n_f, 1), dtype=int)
    interaction_matrix = np.hstack([grand_col, factors_mat])
    interaction_names = [".grand"] + terms
    interaction_order = np.array([0] + [len(t.split(":")) for t in terms], dtype=int)

    cell_means = pd.DataFrame({interaction_names[0]: np.full(n, float(np.mean(y)))})
    for term in terms:
        group_cols = term.split(":")
        cell_means[term] = (
            df.groupby(group_cols, sort=False)[response].transform("mean").to_numpy()
        )

    est = pd.DataFrame(index=df.index)
    for j in range(1, interaction_matrix.shape[1]):
        term_j = interaction_names[j]
        involved = interaction_matrix[:, j] == 1
        col_sums = interaction_matrix[involved, :].sum(axis=0)
        use_cols = col_sums == interaction_order
        multipliers = np.where((interaction_order - interaction_order[j]) % 2, -1, 1)
        effect = np.zeros(n, dtype=float)
        for col_idx, use in enumerate(use_cols):
            if not use:
                continue
            effect += multipliers[col_idx] * cell_means[interaction_names[col_idx]].to_numpy()
        est[term_j] = effect

    ho_term = terms[-1]
    residuals = y - cell_means[ho_term].to_numpy()
    return cell_means, est, residuals


def art_aligned_ranks(
    df: pd.DataFrame,
    response: str,
    factors: list[str],
    rank_comparison_digits: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """
    Port of ARTool::art() alignment + ranking (art.R).

    Returns aligned DataFrame, art (ranked) DataFrame, column sums of aligned.
    """
    if rank_comparison_digits is None:
        rank_comparison_digits = int(-np.floor(np.log10(np.sqrt(np.finfo(float).eps))))

    cell_means, estimated_effects, residuals = art_estimated_effects(df, response, factors)
    terms = list(estimated_effects.columns)

    aligned = pd.DataFrame(
        {t: residuals + estimated_effects[t].to_numpy() for t in terms},
        index=df.index,
    )
    col_sums = {t: float(aligned[t].sum()) for t in terms}

    art_df = pd.DataFrame(
        {
            t: stats.rankdata(np.round(aligned[t].to_numpy(), rank_comparison_digits))
            for t in terms
        },
        index=df.index,
    )
    return aligned, art_df, col_sums


def art_diagnostics_ok(col_sums: dict[str, float], tol: float = 1e-6) -> bool:
    """summary.art: aligned column sums should all be ~0."""
    return all(abs(v) <= tol for v in col_sums.values())


def anova_art(
    df: pd.DataFrame,
    response: str,
    factors: list[str],
    anova_type3_fn,
) -> pd.DataFrame:
    """
    Port of ARTool::anova.art() with Type III tests (anova.art.R + flat.anova.R).

    For each term T, run Type III ANOVA on art(Y) aligned/ranked for T,
  keep the row for T only (default response='art' in anova.art).
    """
    sub = df[list(factors) + [response]].dropna().copy()
    _, art_df, col_sums = art_aligned_ranks(sub, response, factors)
    if not art_diagnostics_ok(col_sums):
        print(
            "  Warning (ARTool summary.art): aligned column sums not all ~0; "
            "ART assumptions may be violated."
        )
        print(f"    column sums: {col_sums}")

    rows = []
    for term in art_df.columns:
        model_df = sub[factors].copy()
        model_df["_art_response"] = art_df[term].to_numpy()
        table = anova_type3_fn(model_df, "_art_response", factors)
        row = table[table["term"] == term]
        if row.empty:
            continue
        rec = row.iloc[0].to_dict()
        rec["analysis"] = "ART_TypeIII"
        rec["art_aligned_col_sum"] = col_sums.get(term, np.nan)
        rows.append(rec)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["method_ref"] = "ARTool (Python port); see mytools/ARTool/R/"
    return out
