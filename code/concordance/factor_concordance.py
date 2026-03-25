"""
Experiment C: VCP Factor-Reduced Concordance.

Test whether reducing VCP from 10 correlated dimensions to 2-3 PCA factors
improves concordance with geometric features.

CPU-only — runs on existing data.
"""

import json
import os
import sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concordance.features import PRIMARY_FEATURES
from concordance.vcp_parser import VCP_V2_DIMENSIONS
from concordance.analysis import (
    load_phase_results, results_to_arrays, fwl_residualize
)

DIM_LETTERS = list(VCP_V2_DIMENSIONS.keys())
MODELS = {
    "qwen2.5-0.5b": "Qwen 0.5B",
    "qwen2.5-7b": "Qwen 7B",
    "meta-llama-3.1-8b": "Llama 8B",
    "mistral-7b-v0.3": "Mistral 7B",
}


def compute_vcp_factors(vcp_matrix, n_factors=3):
    """Compute PCA factor scores from VCP ratings.

    Returns factor scores and loadings.
    """
    valid = ~np.any(np.isnan(vcp_matrix), axis=1)
    V = vcp_matrix[valid]

    # Standardize
    V_mean = V.mean(axis=0)
    V_std = V.std(axis=0) + 1e-10
    V_std_matrix = (V - V_mean) / V_std

    # PCA via eigendecomposition
    corr = np.corrcoef(V_std_matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Factor loadings
    loadings = eigenvectors[:, :n_factors] * np.sqrt(np.maximum(eigenvalues[:n_factors], 0))

    # Factor scores for ALL rows (including those with NaN — they get NaN scores)
    factor_scores = np.full((vcp_matrix.shape[0], n_factors), np.nan)
    factor_scores[valid] = V_std_matrix @ eigenvectors[:, :n_factors]

    var_explained = eigenvalues / eigenvalues.sum()

    return factor_scores, loadings, eigenvalues, var_explained, V_mean, V_std


def factor_concordance_analysis(results_dir):
    """Compare dimension-level vs factor-level concordance."""
    print("=" * 70)
    print("EXPERIMENT C: VCP FACTOR-REDUCED CONCORDANCE")
    print("=" * 70)

    all_results = {}

    for model_short, model_name in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")

        results = load_phase_results(results_dir, model_short, phase="a")
        if not results:
            continue

        vcp, feat_matrix, tc, rl, pt = results_to_arrays(results, phase="generation")
        N = vcp.shape[0]
        if N < 20:
            print(f"  Too few ({N})")
            continue

        conf = np.column_stack([tc, rl])

        # Compute factors
        factor_scores, loadings, eigenvalues, var_exp, _, _ = compute_vcp_factors(vcp, n_factors=3)

        print(f"\n  N={N}")
        print(f"  Variance explained: PC1={var_exp[0]:.1%}, PC2={var_exp[1]:.1%}, PC3={var_exp[2]:.1%}")
        print(f"  Cumulative: {sum(var_exp[:3]):.1%}")

        # Factor loading interpretation
        print(f"\n  Factor Loadings:")
        for i, letter in enumerate(DIM_LETTERS):
            print(f"    {letter} ({VCP_V2_DIMENSIONS[letter]}): "
                  f"F1={loadings[i,0]:+.3f} F2={loadings[i,1]:+.3f} F3={loadings[i,2]:+.3f}")

        # Dimension-level concordance (max |rho_fwl| per feature)
        print(f"\n  --- Dimension-Level Concordance (best of 10 dims per feature) ---")
        dim_best = {}
        for j, feat_name in enumerate(PRIMARY_FEATURES):
            best_rho = 0
            best_dim = ""
            for i, dim in enumerate(DIM_LETTERS):
                valid = ~(np.isnan(vcp[:, i]) | np.isnan(feat_matrix[:, j]))
                if valid.sum() < 10:
                    continue
                v = vcp[valid, i]
                f = feat_matrix[valid, j]
                c = conf[valid]
                v_r = fwl_residualize(v, c)
                f_r = fwl_residualize(f, c)
                rho, p = stats.spearmanr(v_r, f_r)
                if abs(rho) > abs(best_rho):
                    best_rho = rho
                    best_dim = dim
            dim_best[feat_name] = {"best_dim": best_dim, "rho_fwl": round(best_rho, 4)}
            print(f"    {feat_name}: best dim = {best_dim}, rho_fwl = {best_rho:+.4f}")

        # Factor-level concordance
        print(f"\n  --- Factor-Level Concordance ---")
        factor_results = {}
        for fi in range(3):
            factor_results[f"F{fi+1}"] = {}
            for j, feat_name in enumerate(PRIMARY_FEATURES):
                valid = ~(np.isnan(factor_scores[:, fi]) | np.isnan(feat_matrix[:, j]))
                if valid.sum() < 10:
                    continue
                fs = factor_scores[valid, fi]
                fv = feat_matrix[valid, j]
                c = conf[valid]
                fs_r = fwl_residualize(fs, c)
                fv_r = fwl_residualize(fv, c)
                rho, p = stats.spearmanr(fs_r, fv_r)
                factor_results[f"F{fi+1}"][feat_name] = {
                    "rho_fwl": round(float(rho), 4),
                    "p_fwl": float(p),
                    "abs_rho": round(abs(float(rho)), 4),
                }

        for fi in range(3):
            best_feat = max(factor_results[f"F{fi+1}"].items(),
                          key=lambda x: abs(x[1]["rho_fwl"]))
            print(f"    F{fi+1} ({var_exp[fi]:.0%}): best = {best_feat[0]}, "
                  f"rho_fwl = {best_feat[1]['rho_fwl']:+.4f} "
                  f"(p={best_feat[1]['p_fwl']:.4f})")

        # Compare: does factor concordance exceed dimension concordance?
        print(f"\n  --- Comparison: Factor vs Dimension ---")
        comparison = {}
        for feat_name in PRIMARY_FEATURES:
            dim_rho = abs(dim_best[feat_name]["rho_fwl"])

            # Best factor rho for this feature
            best_factor_rho = 0
            best_factor = ""
            for fi in range(3):
                if feat_name in factor_results[f"F{fi+1}"]:
                    r = abs(factor_results[f"F{fi+1}"][feat_name]["rho_fwl"])
                    if r > abs(best_factor_rho):
                        best_factor_rho = r
                        best_factor = f"F{fi+1}"

            winner = "FACTOR" if best_factor_rho > dim_rho else "DIM"
            improvement = best_factor_rho - dim_rho

            comparison[feat_name] = {
                "dim_best_rho": round(dim_rho, 4),
                "dim_best": dim_best[feat_name]["best_dim"],
                "factor_best_rho": round(best_factor_rho, 4),
                "factor_best": best_factor,
                "winner": winner,
                "improvement": round(float(improvement), 4),
            }

            marker = "+" if winner == "FACTOR" else "-"
            print(f"    {feat_name}: dim={dim_rho:.3f} ({dim_best[feat_name]['best_dim']}), "
                  f"factor={best_factor_rho:.3f} ({best_factor}), "
                  f"winner={winner} [{marker}{abs(improvement):.3f}]")

        factor_wins = sum(1 for c in comparison.values() if c["winner"] == "FACTOR")
        print(f"\n    Factor wins: {factor_wins}/{len(comparison)}")

        all_results[model_name] = {
            "n": N,
            "eigenvalues": [round(float(e), 4) for e in eigenvalues],
            "var_explained": [round(float(v), 4) for v in var_exp[:3]],
            "loadings": {
                DIM_LETTERS[i]: {
                    "F1": round(float(loadings[i, 0]), 3),
                    "F2": round(float(loadings[i, 1]), 3),
                    "F3": round(float(loadings[i, 2]), 3),
                }
                for i in range(len(DIM_LETTERS))
            },
            "dim_concordance": dim_best,
            "factor_concordance": factor_results,
            "comparison": comparison,
            "factor_wins": factor_wins,
            "total_features": len(comparison),
        }

    # Save
    out_dir = os.path.join(results_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "factor_concordance_results.json")

    def json_default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=json_default, ensure_ascii=False)

    print(f"\n\nResults saved to {out_path}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/concordance")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        args.results_dir = "C:/Users/Thomas/Desktop/Watson-VCP-Concordance/results/concordance"

    factor_concordance_analysis(args.results_dir)
