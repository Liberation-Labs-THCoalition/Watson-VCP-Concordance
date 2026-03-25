"""
Deep analysis for Paper 2: VCP-Geometry Mode-Switching.

Extracts CCA loadings, VCP factor structure, encode-generation reversal
mechanism, and top_sv_ratio per-dimension breakdown.

CPU-only — runs on existing concordance trial data.
"""

import glob
import json
import os
import sys
import numpy as np
from scipy import stats
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concordance.features import PRIMARY_FEATURES, CONFOUND_COVARIATES
from concordance.vcp_parser import VCP_V2_DIMENSIONS
from concordance.analysis import (
    load_phase_results, results_to_arrays, fwl_residualize
)

DIM_LETTERS = list(VCP_V2_DIMENSIONS.keys())
DIM_NAMES = list(VCP_V2_DIMENSIONS.values())


def bootstrap_spearman_ci(x, y, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap 95% CI for Spearman rho."""
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rhos[b], _ = stats.spearmanr(x[idx], y[idx])
    alpha = (1 - ci) / 2
    lo = float(np.nanpercentile(rhos, 100 * alpha))
    hi = float(np.nanpercentile(rhos, 100 * (1 - alpha)))
    return lo, hi

MODELS = {
    "qwen2.5-0.5b": "Qwen 0.5B",
    "qwen2.5-7b": "Qwen 7B",
    "meta-llama-3.1-8b": "Llama 8B",
    "mistral-7b-v0.3": "Mistral 7B",
}


def safe_cca(X, Y, n_components=None):
    """CCA via SVD, returning loadings and correlations.

    Returns:
        cc: canonical correlations
        x_loadings: (n_vcp, n_components) — VCP dimension weights
        y_loadings: (n_feat, n_components) — feature weights
        x_scores, y_scores: canonical variates
    """
    from numpy.linalg import svd, inv, cholesky

    # Standardize
    X = (X - X.mean(0)) / (X.std(0) + 1e-10)
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-10)

    n = X.shape[0]
    p = X.shape[1]
    q = Y.shape[1]

    if n_components is None:
        n_components = min(p, q)

    # Cross-covariance
    Sxx = (X.T @ X) / (n - 1) + np.eye(p) * 1e-8
    Syy = (Y.T @ Y) / (n - 1) + np.eye(q) * 1e-8
    Sxy = (X.T @ Y) / (n - 1)

    # Cholesky decomposition
    try:
        Lx = cholesky(Sxx)
        Ly = cholesky(Syy)
    except np.linalg.LinAlgError:
        # Fall back to eigendecomposition for non-positive-definite
        Sxx += np.eye(p) * 0.01
        Syy += np.eye(q) * 0.01
        Lx = cholesky(Sxx)
        Ly = cholesky(Syy)

    # SVD of whitened cross-covariance
    Lx_inv = inv(Lx)
    Ly_inv = inv(Ly)
    M = Lx_inv @ Sxy @ Ly_inv.T
    U, s, Vt = svd(M, full_matrices=False)

    cc = s[:n_components]

    # Loadings (transform back to original space)
    x_weights = Lx_inv.T @ U[:, :n_components]
    y_weights = Ly_inv.T @ Vt[:n_components, :].T

    # Canonical variates
    x_scores = X @ x_weights
    y_scores = Y @ y_weights

    # Compute structure correlations (correlation of original vars with canonical variates)
    x_loadings = np.zeros((p, n_components))
    y_loadings = np.zeros((q, n_components))
    for k in range(n_components):
        for i in range(p):
            x_loadings[i, k] = np.corrcoef(X[:, i], x_scores[:, k])[0, 1]
        for j in range(q):
            y_loadings[j, k] = np.corrcoef(Y[:, j], y_scores[:, k])[0, 1]

    return cc, x_loadings, y_loadings, x_scores, y_scores


def analyze_vcp_factor_structure(vcp_matrix, model_name):
    """PCA on VCP ratings to identify factor structure."""
    # Remove NaN rows
    valid = ~np.any(np.isnan(vcp_matrix), axis=1)
    V = vcp_matrix[valid]

    if V.shape[0] < 10:
        return {"error": f"Too few valid rows ({V.shape[0]})"}

    # Standardize
    V_std = (V - V.mean(0)) / (V.std(0) + 1e-10)

    # Eigendecomposition of correlation matrix
    corr_matrix = np.corrcoef(V_std.T)
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Variance explained
    total_var = np.sum(eigenvalues)
    var_explained = eigenvalues / total_var
    cum_var = np.cumsum(var_explained)

    # Number of factors by Kaiser criterion (eigenvalue > 1)
    n_kaiser = int(np.sum(eigenvalues > 1.0))

    # Loadings (eigenvectors * sqrt(eigenvalue))
    loadings = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))

    result = {
        "model": model_name,
        "n_valid": int(V.shape[0]),
        "eigenvalues": [round(float(e), 4) for e in eigenvalues],
        "variance_explained": [round(float(v), 4) for v in var_explained],
        "cumulative_variance": [round(float(c), 4) for c in cum_var],
        "n_kaiser_factors": n_kaiser,
        "pc1_pct": round(float(var_explained[0]) * 100, 1),
        "pc2_pct": round(float(var_explained[1]) * 100, 1),
        "factor_loadings": {},
        "inter_dim_correlations": {},
    }

    # PC1 and PC2 loadings per dimension
    for i, letter in enumerate(DIM_LETTERS):
        result["factor_loadings"][letter] = {
            "name": DIM_NAMES[i],
            "PC1": round(float(loadings[i, 0]), 3),
            "PC2": round(float(loadings[i, 1]), 3),
            "PC3": round(float(loadings[i, 2]), 3) if loadings.shape[1] > 2 else 0,
        }

    # Inter-dimension correlation matrix
    for i, li in enumerate(DIM_LETTERS):
        for j, lj in enumerate(DIM_LETTERS):
            if i < j:
                r, p = stats.spearmanr(V[:, i], V[:, j])
                result["inter_dim_correlations"][f"{li}_{lj}"] = {
                    "rho": round(float(r), 3),
                    "p": float(p),
                }

    return result


def analyze_cca_loadings(vcp_matrix, feature_matrix, token_counts,
                          response_lengths, model_name):
    """CCA with full loading extraction."""
    # FWL residualize everything first
    valid = ~(np.any(np.isnan(vcp_matrix), axis=1) |
              np.any(np.isnan(feature_matrix), axis=1))

    V = vcp_matrix[valid]
    F = feature_matrix[valid]
    tc = token_counts[valid]
    rl = response_lengths[valid]

    if V.shape[0] < 20:
        return {"error": f"Too few valid rows ({V.shape[0]})"}

    # FWL residualize
    conf = np.column_stack([tc, rl])
    V_resid = np.zeros_like(V)
    F_resid = np.zeros_like(F)
    for i in range(V.shape[1]):
        V_resid[:, i] = fwl_residualize(V[:, i], conf)
    for i in range(F.shape[1]):
        F_resid[:, i] = fwl_residualize(F[:, i], conf)

    # Run CCA
    cc, x_load, y_load, x_scores, y_scores = safe_cca(V_resid, F_resid)

    result = {
        "model": model_name,
        "n_valid": int(V.shape[0]),
        "canonical_correlations": [round(float(c), 4) for c in cc],
        "vcp_loadings": {},
        "feature_loadings": {},
        "fwl_applied": True,
    }

    # VCP loadings on CC1 and CC2
    for i, letter in enumerate(DIM_LETTERS):
        result["vcp_loadings"][letter] = {
            "name": DIM_NAMES[i],
            "CC1": round(float(x_load[i, 0]), 3),
            "CC2": round(float(x_load[i, 1]), 3) if x_load.shape[1] > 1 else 0,
        }

    # Feature loadings on CC1 and CC2
    for i, feat in enumerate(PRIMARY_FEATURES):
        result["feature_loadings"][feat] = {
            "CC1": round(float(y_load[i, 0]), 3),
            "CC2": round(float(y_load[i, 1]), 3) if y_load.shape[1] > 1 else 0,
        }

    # Identify dominant VCP dimensions and features for CC1
    vcp_cc1 = [(letter, float(x_load[i, 0])) for i, letter in enumerate(DIM_LETTERS)]
    feat_cc1 = [(feat, float(y_load[i, 0])) for i, feat in enumerate(PRIMARY_FEATURES)]

    vcp_cc1.sort(key=lambda x: abs(x[1]), reverse=True)
    feat_cc1.sort(key=lambda x: abs(x[1]), reverse=True)

    result["cc1_dominant_vcp"] = [
        {"dim": v[0], "loading": round(v[1], 3)} for v in vcp_cc1[:3]
    ]
    result["cc1_dominant_features"] = [
        {"feat": f[0], "loading": round(f[1], 3)} for f in feat_cc1[:3]
    ]

    return result


def analyze_encode_generation_reversal(results_dir, model_short, model_name):
    """Deep dive into encode vs generation sign reversal."""
    results = load_phase_results(results_dir, model_short, phase="a")

    if not results:
        return {"error": "No results found"}

    # Get both encode and generation arrays
    vcp_enc, feat_enc, tc_enc, rl_enc, pt_enc = results_to_arrays(results, phase="encode")
    vcp_gen, feat_gen, tc_gen, rl_gen, pt_gen = results_to_arrays(results, phase="generation")

    N = min(vcp_enc.shape[0], vcp_gen.shape[0])
    if N < 10:
        return {"error": f"Too few valid ({N})"}

    # Truncate to same length (they should be same)
    vcp_enc = vcp_enc[:N]
    feat_enc = feat_enc[:N]
    vcp_gen = vcp_gen[:N]
    feat_gen = feat_gen[:N]
    # M1 FIX: Use phase-specific confounds, not generation for both
    tc_enc = tc_enc[:N]
    rl_enc = rl_enc[:N]
    tc_gen = tc_gen[:N]
    rl_gen = rl_gen[:N]

    conf_enc = np.column_stack([tc_enc, rl_enc])
    conf_gen = np.column_stack([tc_gen, rl_gen])

    result = {
        "model": model_name,
        "n_valid": N,
        "per_pair_analysis": {},
        "sign_flip_dims": [],
        "sign_consistent_dims": [],
        "by_prompt_type": {},
    }

    # For each VCP dim × feature pair, compute encode and generation FWL correlations
    flip_count = 0
    all_rho_pairs = []
    consistent_count = 0

    for i, dim in enumerate(DIM_LETTERS):
        for j, feat in enumerate(PRIMARY_FEATURES):
            vcp_col = vcp_gen[:, i]  # VCP is same for both phases
            feat_enc_col = feat_enc[:, j]
            feat_gen_col = feat_gen[:, j]

            valid = ~(np.isnan(vcp_col) | np.isnan(feat_enc_col) | np.isnan(feat_gen_col))
            if valid.sum() < 10:
                continue

            v = vcp_col[valid]
            fe = feat_enc_col[valid]
            fg = feat_gen_col[valid]
            c_enc = conf_enc[valid]
            c_gen = conf_gen[valid]

            # M1 FIX: Phase-specific FWL residualization
            # VCP comes from generation phase, so use gen confounds
            v_r = fwl_residualize(v, c_gen)
            # Encode features use encode confounds
            fe_r = fwl_residualize(fe, c_enc)
            # Generation features use generation confounds
            fg_r = fwl_residualize(fg, c_gen)

            rho_enc, p_enc = stats.spearmanr(v_r, fe_r)
            rho_gen, p_gen = stats.spearmanr(v_r, fg_r)

            # MC2 FIX: Bootstrap 95% CIs for both correlations
            ci_enc = bootstrap_spearman_ci(v_r, fe_r)
            ci_gen = bootstrap_spearman_ci(v_r, fg_r)

            key = f"{dim}_{feat}"
            # STRICT criterion (C1 fix): both must exceed threshold
            is_flip_strict = (rho_enc * rho_gen < 0) and (abs(rho_enc) > 0.15 and abs(rho_gen) > 0.15)
            # LENIENT criterion (original, for reference)
            is_flip_lenient = (rho_enc * rho_gen < 0) and (abs(rho_enc) > 0.1 or abs(rho_gen) > 0.1)
            # P-VALUE criterion: at least one must be p < 0.05
            is_flip_sig = (rho_enc * rho_gen < 0) and (p_enc < 0.05 or p_gen < 0.05)

            result["per_pair_analysis"][key] = {
                "dim": dim,
                "feature": feat,
                "rho_encode": round(float(rho_enc), 4),
                "p_encode": float(p_enc),
                "rho_encode_ci95": [round(ci_enc[0], 4), round(ci_enc[1], 4)],
                "rho_generation": round(float(rho_gen), 4),
                "p_generation": float(p_gen),
                "rho_generation_ci95": [round(ci_gen[0], 4), round(ci_gen[1], 4)],
                "sign_flip_strict": is_flip_strict,
                "sign_flip_lenient": is_flip_lenient,
                "sign_flip_sig": is_flip_sig,
                "delta": round(float(rho_gen - rho_enc), 4),
            }

            if is_flip_strict:
                flip_count += 1
                result["sign_flip_dims"].append(key)
            else:
                consistent_count += 1
                result["sign_consistent_dims"].append(key)

            all_rho_pairs.append((rho_enc, rho_gen))

    result["n_sign_flips_strict"] = flip_count
    result["n_sign_flips_lenient"] = sum(
        1 for v in result["per_pair_analysis"].values() if v.get("sign_flip_lenient"))
    result["n_sign_flips_sig"] = sum(
        1 for v in result["per_pair_analysis"].values() if v.get("sign_flip_sig"))
    result["n_consistent"] = consistent_count
    result["flip_rate_strict"] = round(flip_count / max(flip_count + consistent_count, 1), 3)

    # C2 FIX: Permutation null model for flip count
    n_total_pairs = flip_count + consistent_count
    n_perms = 1000
    null_flips = []
    rng = np.random.default_rng(42)
    for _ in range(n_perms):
        null_count = 0
        for rho_e, rho_g in all_rho_pairs:
            # Randomly flip sign of one correlation to simulate null
            if rng.random() < 0.5:
                rho_e_perm = -rho_e
            else:
                rho_e_perm = rho_e
            if (rho_e_perm * rho_g < 0) and (abs(rho_e_perm) > 0.15 and abs(rho_g) > 0.15):
                null_count += 1
        null_flips.append(null_count)

    null_mean = np.mean(null_flips)
    null_std = np.std(null_flips)
    p_perm = np.mean([n >= flip_count for n in null_flips])

    result["permutation_null"] = {
        "n_perms": n_perms,
        "null_mean_flips": round(float(null_mean), 2),
        "null_std_flips": round(float(null_std), 2),
        "observed_flips": flip_count,
        "p_value": round(float(p_perm), 4),
        "excess_over_null": round(float(flip_count - null_mean), 2),
    }

    # Analyze by prompt type — which types show most reversal?
    prompt_types_unique = list(set(pt_gen))
    for ptype in prompt_types_unique:
        type_mask = np.array([p == ptype for p in pt_gen[:N]])
        n_type = type_mask.sum()
        if n_type < 5:
            continue

        # Compute mean feature difference (gen - enc) for this type
        feat_delta = feat_gen[type_mask] - feat_enc[type_mask]
        mean_delta = np.nanmean(feat_delta, axis=0)

        result["by_prompt_type"][ptype] = {
            "n": int(n_type),
            "mean_feature_delta": {
                feat: round(float(mean_delta[j]), 4)
                for j, feat in enumerate(PRIMARY_FEATURES)
            },
        }

    return result


def analyze_top_sv_ratio_universality(results_dir):
    """Detailed analysis of WHY top_sv_ratio is the universal VCP indicator."""
    result = {
        "per_model": {},
        "cross_model_consistency": {},
    }

    for model_short, model_name in MODELS.items():
        results = load_phase_results(results_dir, model_short, phase="a")
        if not results:
            continue

        vcp, feat, tc, rl, pt = results_to_arrays(results, phase="generation")

        N = vcp.shape[0]
        if N < 10:
            continue

        conf = np.column_stack([tc, rl])

        # top_sv_ratio is index 4 in PRIMARY_FEATURES
        tsv_idx = PRIMARY_FEATURES.index("top_sv_ratio")
        tsv_col = feat[:, tsv_idx]

        model_result = {
            "n": N,
            "tsv_mean": round(float(np.nanmean(tsv_col)), 4),
            "tsv_std": round(float(np.nanstd(tsv_col)), 4),
            "per_dim_fwl_rho": {},
            "per_type_mean": {},
            "per_type_d_vs_others": {},
        }

        # FWL correlation with each VCP dimension
        for i, dim in enumerate(DIM_LETTERS):
            valid = ~(np.isnan(vcp[:, i]) | np.isnan(tsv_col))
            if valid.sum() < 10:
                continue
            v = vcp[valid, i]
            t = tsv_col[valid]
            c = conf[valid]

            v_r = fwl_residualize(v, c)
            t_r = fwl_residualize(t, c)
            rho, p = stats.spearmanr(v_r, t_r)

            model_result["per_dim_fwl_rho"][dim] = {
                "rho": round(float(rho), 4),
                "p": float(p),
                "abs_rho": round(abs(float(rho)), 4),
            }

        # Mean top_sv_ratio per prompt type
        for ptype in set(pt):
            mask = np.array([p == ptype for p in pt])
            vals = tsv_col[mask]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 2:
                model_result["per_type_mean"][ptype] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "n": len(vals),
                }

        # Cohen's d for each type vs all others
        for ptype in set(pt):
            mask = np.array([p == ptype for p in pt])
            in_vals = tsv_col[mask & ~np.isnan(tsv_col)]
            out_vals = tsv_col[~mask & ~np.isnan(tsv_col)]
            if len(in_vals) > 2 and len(out_vals) > 2:
                n1, n2 = len(in_vals), len(out_vals)
                v1 = np.var(in_vals, ddof=1)
                v2 = np.var(out_vals, ddof=1)
                pooled = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
                d = (np.mean(in_vals) - np.mean(out_vals)) / pooled if pooled > 0 else 0
                model_result["per_type_d_vs_others"][ptype] = round(float(d), 3)

        result["per_model"][model_name] = model_result

    # Cross-model consistency of top_sv_ratio VCP profile
    # For each model pair, correlate the 10-dim rho vectors
    model_names = list(result["per_model"].keys())
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i >= j:
                continue
            rhos1 = [result["per_model"][m1]["per_dim_fwl_rho"].get(d, {}).get("rho", np.nan)
                     for d in DIM_LETTERS]
            rhos2 = [result["per_model"][m2]["per_dim_fwl_rho"].get(d, {}).get("rho", np.nan)
                     for d in DIM_LETTERS]

            valid = ~(np.isnan(rhos1) | np.isnan(rhos2))
            if np.sum(valid) > 3:
                r1 = np.array(rhos1)[valid]
                r2 = np.array(rhos2)[valid]
                rho, p = stats.spearmanr(r1, r2)
                result["cross_model_consistency"][f"{m1}_vs_{m2}"] = {
                    "profile_rho": round(float(rho), 3),
                    "p": float(p),
                    "n_dims": int(np.sum(valid)),
                }

    return result


def run_deep_analysis(results_dir):
    """Run all deep analyses and save results."""
    print("=" * 70)
    print("DEEP ANALYSIS: VCP-Geometry Mode-Switching")
    print("=" * 70)

    all_results = {}

    for model_short, model_name in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")

        results = load_phase_results(results_dir, model_short, phase="a")
        if not results:
            print(f"  No results for {model_short}")
            continue

        vcp, feat, tc, rl, pt = results_to_arrays(results, phase="generation")
        print(f"  N valid: {vcp.shape[0]}")

        # 1. VCP Factor Structure
        print(f"\n  --- VCP Factor Structure ---")
        factors = analyze_vcp_factor_structure(vcp, model_name)
        if "error" not in factors:
            print(f"  PC1: {factors['pc1_pct']}%, PC2: {factors['pc2_pct']}%")
            print(f"  Kaiser factors: {factors['n_kaiser_factors']}")
            print(f"  Top PC1 loadings:")
            sorted_loadings = sorted(
                factors["factor_loadings"].items(),
                key=lambda x: abs(x[1]["PC1"]), reverse=True
            )
            for dim, load in sorted_loadings[:5]:
                print(f"    {dim} ({load['name']}): PC1={load['PC1']:.3f}, PC2={load['PC2']:.3f}")

        # 2. CCA Loadings (FWL-residualized)
        print(f"\n  --- CCA Loadings (FWL) ---")
        cca = analyze_cca_loadings(vcp, feat, tc, rl, model_name)
        if "error" not in cca:
            print(f"  CC1={cca['canonical_correlations'][0]:.3f}")
            print(f"  CC1 dominant VCP: {cca['cc1_dominant_vcp']}")
            print(f"  CC1 dominant features: {cca['cc1_dominant_features']}")

        # 3. Encode-Generation Reversal
        print(f"\n  --- Encode-Generation Reversal ---")
        reversal = analyze_encode_generation_reversal(results_dir, model_short, model_name)
        if "error" not in reversal:
            print(f"  Sign flips: {reversal['n_sign_flips_strict']}/{reversal['n_sign_flips_strict'] + reversal['n_consistent']}")
            print(f"  Flip rate: {reversal['flip_rate_strict']:.1%}")
            if reversal["sign_flip_dims"]:
                print(f"  Flipped pairs: {reversal['sign_flip_dims'][:10]}")

        all_results[model_name] = {
            "vcp_factors": factors,
            "cca_loadings": cca,
            "encode_gen_reversal": reversal,
        }

    # 4. top_sv_ratio universality analysis
    print(f"\n{'='*50}")
    print("top_sv_ratio Universality Analysis")
    print(f"{'='*50}")

    tsv_analysis = analyze_top_sv_ratio_universality(results_dir)
    all_results["top_sv_ratio_analysis"] = tsv_analysis

    for model_name, data in tsv_analysis["per_model"].items():
        print(f"\n  {model_name}:")
        print(f"    Mean: {data['tsv_mean']:.4f} +/- {data['tsv_std']:.4f}")
        print(f"    Per-type d (meta-cognitive vs others): {data['per_type_d_vs_others'].get('meta', 'N/A')}")
        print(f"    Top VCP correlates (FWL):")
        sorted_dims = sorted(
            data["per_dim_fwl_rho"].items(),
            key=lambda x: abs(x[1]["rho"]), reverse=True
        )
        for dim, info in sorted_dims[:5]:
            sig = "*" if info["p"] < 0.05 else ""
            print(f"      {dim}: rho={info['rho']:.3f} (p={info['p']:.4f}){sig}")

    print(f"\n  Cross-model profile consistency:")
    for pair, info in tsv_analysis["cross_model_consistency"].items():
        sig = "***" if info["p"] < 0.001 else "**" if info["p"] < 0.01 else "*" if info["p"] < 0.05 else ""
        print(f"    {pair}: rho={info['profile_rho']:.3f} (p={info['p']:.4f}){sig}")

    # Save
    out_dir = os.path.join(results_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "deep_analysis_results.json")

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

    run_deep_analysis(args.results_dir)
