"""
Concordance study analysis — 60 Spearman correlations with FWL + CCA + ICC.

Usage:
    python -m concordance.analysis --results-dir results/concordance
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concordance.features import PRIMARY_FEATURES, CONFOUND_COVARIATES
from concordance.vcp_parser import VCP_V2_DIMENSIONS, VCP_V5_DIMENSIONS


# ================================================================
# DATA LOADING
# ================================================================

def load_phase_results(results_dir, model_short, phase="a"):
    """Load all per-prompt JSONs from a phase directory."""
    phase_dir = os.path.join(results_dir, model_short, f"phase_{phase}")
    if not os.path.isdir(phase_dir):
        print(f"  Warning: {phase_dir} not found")
        return []

    results = []
    for f in sorted(glob.glob(os.path.join(phase_dir, "*.json"))):
        if os.path.basename(f).startswith("_") or f.endswith("summary.json"):
            continue
        with open(f, encoding="utf-8") as fh:
            results.append(json.load(fh))
    return results


def results_to_arrays(results, phase="generation"):
    """Convert results list to parallel numpy arrays for analysis.

    Args:
        results: list of per-prompt result dicts
        phase: "encode", "generation", or "delta"

    Returns:
        (vcp_matrix, feature_matrix, token_counts, response_lengths, prompt_types)
        vcp_matrix: (N, 10) array of VCP ratings
        feature_matrix: (N, 6) array of protocol features
        token_counts: (N,) array
        response_lengths: (N,) array
        prompt_types: list of N type strings
    """
    dim_letters = list(VCP_V2_DIMENSIONS.keys())

    valid_results = [r for r in results if r.get("vcp_parse_quality") == "clean"]
    if not valid_results:
        valid_results = [r for r in results if r.get("vcp_parse_quality") in ("clean", "partial")]

    N = len(valid_results)
    vcp_matrix = np.zeros((N, len(dim_letters)))
    feature_matrix = np.zeros((N, len(PRIMARY_FEATURES)))
    token_counts = np.zeros(N)
    response_lengths = np.zeros(N)
    prompt_types = []

    feat_key = f"{phase}_features"

    for i, r in enumerate(valid_results):
        # VCP ratings
        vcp = r.get("vcp_ratings", {})
        for j, letter in enumerate(dim_letters):
            vcp_matrix[i, j] = vcp.get(letter, np.nan)

        # Geometric features
        feat = r.get(feat_key, r.get("generation_features", {}))
        for j, fname in enumerate(PRIMARY_FEATURES):
            feature_matrix[i, j] = feat.get(fname, np.nan)

        # Confound covariates
        token_counts[i] = r.get("n_tokens", feat.get("n_tokens", 0))
        response_lengths[i] = r.get("response_length", 0)
        prompt_types.append(r.get("prompt_type", "unknown"))

    return vcp_matrix, feature_matrix, token_counts, response_lengths, prompt_types


# ================================================================
# FWL RESIDUALIZATION
# ================================================================

def fwl_residualize(y, X_confound):
    """Frisch-Waugh-Lovell residualization.

    Regress y on X_confound, return residuals.
    This removes the linear effect of confounds from y.
    """
    X = np.column_stack([np.ones(len(y)), X_confound])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
    except np.linalg.LinAlgError:
        residuals = y
    return residuals


# ================================================================
# PRIMARY ANALYSIS: 60 SPEARMAN CORRELATIONS
# ================================================================

def compute_correlation_matrix(vcp_matrix, feature_matrix, token_counts,
                                response_lengths, alpha=0.05):
    """Compute 60 Spearman correlations with FWL confound control.

    Returns:
        dict with raw correlations, FWL-corrected correlations,
        and confound flags.
    """
    dim_letters = list(VCP_V2_DIMENSIONS.keys())
    n_tests = len(dim_letters) * len(PRIMARY_FEATURES)
    bonferroni_alpha = alpha / n_tests

    confound_matrix = np.column_stack([token_counts, response_lengths])

    results = {}
    flagged = []

    for i, dim in enumerate(dim_letters):
        for j, feat in enumerate(PRIMARY_FEATURES):
            vcp_col = vcp_matrix[:, i]
            feat_col = feature_matrix[:, j]

            # Skip if too many NaNs
            valid = ~(np.isnan(vcp_col) | np.isnan(feat_col))
            if valid.sum() < 10:
                continue

            v = vcp_col[valid]
            f = feat_col[valid]
            tc = token_counts[valid]
            rl = response_lengths[valid]

            # Raw Spearman
            rho_raw, p_raw = stats.spearmanr(v, f)

            # FWL residualized Spearman
            conf = np.column_stack([tc, rl])
            v_resid = fwl_residualize(v, conf)
            f_resid = fwl_residualize(f, conf)
            rho_fwl, p_fwl = stats.spearmanr(v_resid, f_resid)

            # Confound flag
            confound_delta = abs(rho_raw - rho_fwl)
            is_confounded = confound_delta > 0.15

            key = f"{dim}_{feat}"
            results[key] = {
                "vcp_dim": dim,
                "vcp_name": VCP_V2_DIMENSIONS[dim],
                "feature": feat,
                "rho_raw": round(rho_raw, 4),
                "p_raw": p_raw,
                "rho_fwl": round(rho_fwl, 4),
                "p_fwl": p_fwl,
                "significant_raw": p_raw < bonferroni_alpha,
                "significant_fwl": p_fwl < bonferroni_alpha,
                "confound_delta": round(confound_delta, 4),
                "confound_flag": is_confounded,
                "n": int(valid.sum()),
            }

            if is_confounded:
                flagged.append(key)

    return {
        "correlations": results,
        "n_tests": n_tests,
        "bonferroni_alpha": bonferroni_alpha,
        "n_confound_flagged": len(flagged),
        "confound_flagged": flagged,
    }


# ================================================================
# HYPOTHESIS TESTS
# ================================================================

def _cohens_d(group1, group2):
    """Compute Cohen's d with pooled sample SD (Bessel-corrected)."""
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_sd


def _permutation_test(values, group_mask, n_perms=10000, rng_seed=42):
    """Two-sided permutation test for group mean difference."""
    valid = ~np.isnan(values)
    vals = values[valid]
    mask = group_mask[valid]
    observed = abs(np.mean(vals[mask]) - np.mean(vals[~mask]))
    rng = np.random.default_rng(rng_seed)
    count = 0
    for _ in range(n_perms):
        perm = rng.permutation(mask)
        perm_diff = abs(np.mean(vals[perm]) - np.mean(vals[~perm]))
        if perm_diff >= observed:
            count += 1
    return count / n_perms


def test_hypotheses(vcp_matrix, feature_matrix, prompt_types,
                    token_counts=None, response_lengths=None):
    """Test the 6 protocol hypotheses.

    H1: Analytical tasks -> higher eff_rank
    H2: Affective tasks -> distinct spectral_entropy
    H3: Meta-cognitive tasks -> highest layer_variance (using index 4=top_sv_ratio as proxy if no layer_var)
    H4: VCP-A correlates with eff_rank (r > 0.3) — FWL-corrected
    H5: VCP-D correlates with norm_per_token — FWL-corrected
    H6: Cross-scale consistency (tested separately per model)
    """
    dim_letters = list(VCP_V2_DIMENSIONS.keys())
    types = np.array(prompt_types)

    results = {}

    # H1: cognitive prompts -> higher eff_rank
    cog_mask = types == "cognitive"
    other_mask = types != "cognitive"
    if cog_mask.sum() > 5 and other_mask.sum() > 5:
        eff_rank_idx = PRIMARY_FEATURES.index("eff_rank")
        cog_vals = feature_matrix[cog_mask, eff_rank_idx]
        other_vals = feature_matrix[other_mask, eff_rank_idx]
        cog_vals = cog_vals[~np.isnan(cog_vals)]
        other_vals = other_vals[~np.isnan(other_vals)]
        if len(cog_vals) > 2 and len(other_vals) > 2:
            t_stat, p_val = stats.mannwhitneyu(cog_vals, other_vals, alternative="greater")
            d = _cohens_d(cog_vals, other_vals)
            results["H1"] = {
                "description": "Cognitive tasks -> higher eff_rank",
                "cog_mean": round(float(np.mean(cog_vals)), 2),
                "other_mean": round(float(np.mean(other_vals)), 2),
                "cohen_d": round(float(d), 3),
                "p": float(p_val),
                "supported": p_val < 0.05 and d > 0,
            }

    # H2: affective tasks -> distinct spectral_entropy
    aff_mask = types == "affective"
    if aff_mask.sum() > 5 and (~aff_mask).sum() > 5:
        se_idx = PRIMARY_FEATURES.index("spectral_entropy")
        aff_vals = feature_matrix[aff_mask, se_idx]
        non_vals = feature_matrix[~aff_mask, se_idx]
        aff_vals = aff_vals[~np.isnan(aff_vals)]
        non_vals = non_vals[~np.isnan(non_vals)]
        if len(aff_vals) > 2 and len(non_vals) > 2:
            t_stat, p_val = stats.mannwhitneyu(aff_vals, non_vals)
            d = _cohens_d(aff_vals, non_vals)
            results["H2"] = {
                "description": "Affective tasks -> distinct spectral_entropy",
                "aff_mean": round(float(np.mean(aff_vals)), 4),
                "other_mean": round(float(np.mean(non_vals)), 4),
                "cohen_d": round(float(d), 3),
                "p": float(p_val),
                "supported": p_val < 0.05,
            }

    # H3: meta-cognitive -> highest layer_variance (proxy: top_sv_ratio)
    meta_mask = types == "metacognitive"
    if meta_mask.sum() > 5:
        # Use top_sv_ratio as proxy since layer_variance is auxiliary
        tsv_idx = PRIMARY_FEATURES.index("top_sv_ratio")
        meta_vals = feature_matrix[meta_mask, tsv_idx]
        other_vals = feature_matrix[~meta_mask, tsv_idx]
        meta_vals = meta_vals[~np.isnan(meta_vals)]
        other_vals = other_vals[~np.isnan(other_vals)]
        if len(meta_vals) > 2 and len(other_vals) > 2:
            t_stat, p_val = stats.mannwhitneyu(meta_vals, other_vals)
            d = _cohens_d(meta_vals, other_vals)

            # Permutation test for H3
            perm_p = _permutation_test(
                feature_matrix[:, tsv_idx], meta_mask, n_perms=10000
            )

            results["H3"] = {
                "description": "Meta-cognitive -> distinct top_sv_ratio",
                "meta_mean": round(float(np.mean(meta_vals)), 4),
                "other_mean": round(float(np.mean(other_vals)), 4),
                "cohen_d": round(float(d), 3),
                "p": float(p_val),
                "p_permutation": float(perm_p),
                "supported": p_val < 0.05,
            }

    # H4: VCP-A correlates with eff_rank — FWL-corrected
    a_idx = dim_letters.index("A")
    er_idx = PRIMARY_FEATURES.index("eff_rank")
    valid = ~(np.isnan(vcp_matrix[:, a_idx]) | np.isnan(feature_matrix[:, er_idx]))
    if valid.sum() > 10:
        rho_raw, p_raw = stats.spearmanr(
            vcp_matrix[valid, a_idx], feature_matrix[valid, er_idx]
        )
        # FWL-corrected
        if token_counts is not None and response_lengths is not None:
            conf = np.column_stack([token_counts[valid], response_lengths[valid]])
            v_r = fwl_residualize(vcp_matrix[valid, a_idx], conf)
            f_r = fwl_residualize(feature_matrix[valid, er_idx], conf)
            rho_fwl, p_fwl = stats.spearmanr(v_r, f_r)
        else:
            rho_fwl, p_fwl = rho_raw, p_raw
        results["H4"] = {
            "description": "VCP-A (Analytical) correlates with eff_rank",
            "rho_raw": round(float(rho_raw), 4),
            "rho_fwl": round(float(rho_fwl), 4),
            "p_raw": float(p_raw),
            "p_fwl": float(p_fwl),
            "supported": abs(rho_fwl) > 0.3 and p_fwl < 0.05,
            "threshold": 0.3,
        }

    # H5: VCP-D correlates with norm_per_token — FWL-corrected
    d_idx = dim_letters.index("D")
    npt_idx = PRIMARY_FEATURES.index("norm_per_token")
    valid = ~(np.isnan(vcp_matrix[:, d_idx]) | np.isnan(feature_matrix[:, npt_idx]))
    if valid.sum() > 10:
        rho_raw, p_raw = stats.spearmanr(
            vcp_matrix[valid, d_idx], feature_matrix[valid, npt_idx]
        )
        if token_counts is not None and response_lengths is not None:
            conf = np.column_stack([token_counts[valid], response_lengths[valid]])
            v_r = fwl_residualize(vcp_matrix[valid, d_idx], conf)
            f_r = fwl_residualize(feature_matrix[valid, npt_idx], conf)
            rho_fwl, p_fwl = stats.spearmanr(v_r, f_r)
        else:
            rho_fwl, p_fwl = rho_raw, p_raw
        results["H5"] = {
            "description": "VCP-D (Depth) correlates with norm_per_token",
            "rho_raw": round(float(rho_raw), 4),
            "rho_fwl": round(float(rho_fwl), 4),
            "p_raw": float(p_raw),
            "p_fwl": float(p_fwl),
            "supported": abs(rho_fwl) > 0.2 and p_fwl < 0.05,
        }

    return results


# ================================================================
# ICC (VCP RELIABILITY)
# ================================================================

def compute_icc(phase_b_results):
    """Compute ICC(2,1) for each VCP dimension from Phase B replications.

    Requires results with multiple reps per prompt.
    """
    dim_letters = list(VCP_V2_DIMENSIONS.keys())

    # Group by prompt_id
    by_prompt = defaultdict(list)
    for r in phase_b_results:
        pid = r["prompt_id"]
        vcp = r.get("vcp_ratings", {})
        if isinstance(vcp, dict):
            by_prompt[pid].append(vcp)

    # Need at least 5 prompts with 2+ reps
    valid_prompts = {pid: reps for pid, reps in by_prompt.items() if len(reps) >= 2}
    if len(valid_prompts) < 5:
        return {"error": f"Only {len(valid_prompts)} prompts with 2+ reps, need >= 5"}

    iccs = {}
    for dim in dim_letters:
        # Build matrix: rows=prompts, cols=reps
        ratings = []
        for pid, reps in valid_prompts.items():
            row = [rep.get(dim, np.nan) for rep in reps]
            ratings.append(row)

        # Pad to same length
        max_reps = max(len(row) for row in ratings)
        padded = np.full((len(ratings), max_reps), np.nan)
        for i, row in enumerate(ratings):
            padded[i, :len(row)] = row

        # Remove rows/cols with all NaN
        valid_rows = ~np.all(np.isnan(padded), axis=1)
        padded = padded[valid_rows]

        if padded.shape[0] < 5:
            iccs[dim] = {"icc": np.nan, "n_prompts": int(padded.shape[0]), "reliable": False}
            continue

        # ICC(2,1) computation
        n = padded.shape[0]  # number of subjects (prompts)
        k = padded.shape[1]  # number of raters (reps)

        # Handle NaN by using mean imputation for ANOVA
        col_means = np.nanmean(padded, axis=0)
        for j in range(k):
            nan_mask = np.isnan(padded[:, j])
            padded[nan_mask, j] = col_means[j]

        grand_mean = np.mean(padded)
        row_means = np.mean(padded, axis=1)
        col_means = np.mean(padded, axis=0)

        # Sum of squares
        ss_total = np.sum((padded - grand_mean) ** 2)
        ss_rows = k * np.sum((row_means - grand_mean) ** 2)
        ss_cols = n * np.sum((col_means - grand_mean) ** 2)
        ss_error = ss_total - ss_rows - ss_cols

        # Mean squares
        ms_rows = ss_rows / (n - 1) if n > 1 else 0
        ms_cols = ss_cols / (k - 1) if k > 1 else 0
        ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 1

        # ICC(2,1)
        icc_val = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n) if (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n) != 0 else 0

        iccs[dim] = {
            "icc": round(float(icc_val), 4),
            "n_prompts": int(n),
            "n_reps": int(k),
            "reliable": float(icc_val) > 0.7,
        }

    return iccs


# ================================================================
# CCA (CANONICAL CORRELATION)
# ================================================================

def compute_cca(vcp_matrix, feature_matrix):
    """Canonical Correlation Analysis between VCP and geometric features.

    Returns canonical correlations and their significance.
    """
    # Remove rows with NaN
    valid = ~(np.any(np.isnan(vcp_matrix), axis=1) | np.any(np.isnan(feature_matrix), axis=1))
    X = vcp_matrix[valid]
    Y = feature_matrix[valid]

    if X.shape[0] < 20:
        return {"error": f"Only {X.shape[0]} valid observations, need >= 20"}

    # Standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-10)

    # Cross-covariance
    n = X.shape[0]
    Cxx = X.T @ X / n
    Cyy = Y.T @ Y / n
    Cxy = X.T @ Y / n

    # Regularize for numerical stability
    reg = 1e-6
    Cxx += reg * np.eye(Cxx.shape[0])
    Cyy += reg * np.eye(Cyy.shape[0])

    try:
        Cxx_inv_sqrt = np.linalg.inv(np.linalg.cholesky(Cxx))
        Cyy_inv_sqrt = np.linalg.inv(np.linalg.cholesky(Cyy))
        T = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt.T
        U, s, Vt = np.linalg.svd(T, full_matrices=False)
        canonical_corrs = s[:min(X.shape[1], Y.shape[1])]
    except np.linalg.LinAlgError:
        return {"error": "SVD failed in CCA"}

    # Permutation significance test for first canonical correlation
    n_perms = 1000
    rng = np.random.default_rng(42)
    observed_cc1 = canonical_corrs[0]
    exceed_count = 0
    for _ in range(n_perms):
        Y_perm = Y[rng.permutation(n)]
        Cxy_perm = X.T @ Y_perm / n
        try:
            T_perm = Cxx_inv_sqrt @ Cxy_perm @ Cyy_inv_sqrt.T
            _, s_perm, _ = np.linalg.svd(T_perm, full_matrices=False)
            if s_perm[0] >= observed_cc1:
                exceed_count += 1
        except np.linalg.LinAlgError:
            pass
    p_perm = exceed_count / n_perms

    return {
        "canonical_correlations": [round(float(c), 4) for c in canonical_corrs],
        "cc1_p_permutation": float(p_perm),
        "n_permutations": n_perms,
        "n_observations": int(n),
        "n_vcp_dims": int(X.shape[1]),
        "n_features": int(Y.shape[1]),
    }


# ================================================================
# CROSS-SCALE ANALYSIS (H6)
# ================================================================

def test_cross_scale_consistency(results_dir, models):
    """Test H6: correlations consistent across model scales.

    Compare correlation matrices between models using Spearman rho
    of the flattened correlation vectors.
    """
    model_corrs = {}

    for model_short in models:
        results = load_phase_results(results_dir, model_short, "a")
        if not results:
            continue

        vcp, feat, tc, rl, pt = results_to_arrays(results)
        if vcp.shape[0] < 20:
            continue

        # Compute FWL-corrected correlations (flattened vector)
        dim_letters = list(VCP_V2_DIMENSIONS.keys())
        confound_matrix = np.column_stack([tc, rl])
        corr_vector = []
        for i in range(len(dim_letters)):
            for j in range(len(PRIMARY_FEATURES)):
                valid = ~(np.isnan(vcp[:, i]) | np.isnan(feat[:, j]))
                if valid.sum() > 5:
                    conf = confound_matrix[valid]
                    v_r = fwl_residualize(vcp[valid, i], conf)
                    f_r = fwl_residualize(feat[valid, j], conf)
                    rho, _ = stats.spearmanr(v_r, f_r)
                    corr_vector.append(rho)
                else:
                    corr_vector.append(0)
        model_corrs[model_short] = np.array(corr_vector)

    # Pairwise comparison
    comparisons = {}
    model_names = list(model_corrs.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            rho, p = stats.spearmanr(model_corrs[m1], model_corrs[m2])
            comparisons[f"{m1}_vs_{m2}"] = {
                "rho": round(float(rho), 4),
                "p": float(p),
                "consistent": float(rho) > 0.5 and p < 0.05,
            }

    return {
        "models": model_names,
        "n_features_compared": len(model_corrs.get(model_names[0], [])) if model_names else 0,
        "comparisons": comparisons,
    }


# ================================================================
# MAIN ANALYSIS PIPELINE
# ================================================================

def run_full_analysis(results_dir, output_dir=None):
    """Run all analysis steps on collected data."""
    if output_dir is None:
        output_dir = os.path.join(results_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("CONCORDANCE ANALYSIS")
    print("=" * 70)

    # Find available models
    model_dirs = [d for d in os.listdir(results_dir)
                  if os.path.isdir(os.path.join(results_dir, d))
                  and d != "analysis" and d != "pilot"]

    all_results = {}
    for model_short in sorted(model_dirs):
        print(f"\n--- Model: {model_short} ---")

        # Load Phase A results
        results = load_phase_results(results_dir, model_short, "a")
        if not results:
            print(f"  No Phase A results found")
            continue

        print(f"  Loaded {len(results)} trials")

        # Parse quality summary
        clean = sum(1 for r in results if r.get("vcp_parse_quality") == "clean")
        partial = sum(1 for r in results if r.get("vcp_parse_quality") == "partial")
        failed = sum(1 for r in results if r.get("vcp_parse_quality") == "failed")
        print(f"  VCP parse: {clean} clean, {partial} partial, {failed} failed")

        # Convert to arrays
        vcp, feat, tc, rl, pt = results_to_arrays(results)
        print(f"  Valid observations: {vcp.shape[0]}")

        # 60 Spearman correlations with FWL
        corr_results = compute_correlation_matrix(vcp, feat, tc, rl)
        n_sig_raw = sum(1 for v in corr_results["correlations"].values() if v["significant_raw"])
        n_sig_fwl = sum(1 for v in corr_results["correlations"].values() if v["significant_fwl"])
        n_flagged = corr_results["n_confound_flagged"]
        print(f"  Correlations: {n_sig_raw} significant (raw), {n_sig_fwl} (FWL)")
        print(f"  Confound flagged: {n_flagged}")

        # Hypothesis tests (with confound data for FWL-corrected H4/H5)
        hypotheses = test_hypotheses(vcp, feat, pt, tc, rl)
        for h_id, h_result in hypotheses.items():
            status = "SUPPORTED" if h_result.get("supported") else "NOT SUPPORTED"
            print(f"  {h_id}: {status} - {h_result['description']}")

        # CCA
        cca = compute_cca(vcp, feat)
        if "canonical_correlations" in cca:
            print(f"  CCA: {cca['canonical_correlations'][:3]}")

        # Phase B ICC (if available)
        phase_b_results = load_phase_results(results_dir, model_short, "b")
        iccs = {}
        if phase_b_results:
            iccs = compute_icc(phase_b_results)
            reliable = sum(1 for v in iccs.values() if isinstance(v, dict) and v.get("reliable"))
            print(f"  ICC: {reliable}/10 dimensions reliable (> 0.7)")

        all_results[model_short] = {
            "n_trials": len(results),
            "n_valid": int(vcp.shape[0]),
            "correlations": corr_results,
            "hypotheses": hypotheses,
            "cca": cca,
            "icc": iccs,
        }

    # Encode-phase circularity check
    print(f"\n--- Encode-Phase Circularity Check ---")
    for model_short in sorted(model_dirs):
        results_list = load_phase_results(results_dir, model_short, "a")
        if not results_list:
            continue
        vcp_gen, feat_gen, tc_gen, rl_gen, pt_gen = results_to_arrays(results_list, "generation")
        vcp_enc, feat_enc, tc_enc, rl_enc, pt_enc = results_to_arrays(results_list, "encode")
        if vcp_gen.shape[0] < 20:
            continue

        dim_letters = list(VCP_V2_DIMENSIONS.keys())
        gen_fwl_vec = []
        enc_fwl_vec = []
        for i in range(len(dim_letters)):
            for j in range(len(PRIMARY_FEATURES)):
                v = vcp_gen[:, i]
                fg = feat_gen[:, j]
                fe = feat_enc[:, j]
                valid_g = ~(np.isnan(v) | np.isnan(fg))
                valid_e = ~(np.isnan(v) | np.isnan(fe))
                rho_g = rho_e = 0.0
                if valid_g.sum() > 10:
                    conf = np.column_stack([tc_gen[valid_g], rl_gen[valid_g]])
                    v_r = fwl_residualize(v[valid_g], conf)
                    f_r = fwl_residualize(fg[valid_g], conf)
                    rho_g, _ = stats.spearmanr(v_r, f_r)
                if valid_e.sum() > 10:
                    conf = np.column_stack([tc_enc[valid_e], rl_enc[valid_e]])
                    v_r = fwl_residualize(v[valid_e], conf)
                    f_r = fwl_residualize(fe[valid_e], conf)
                    rho_e, _ = stats.spearmanr(v_r, f_r)
                gen_fwl_vec.append(rho_g)
                enc_fwl_vec.append(rho_e)

        gen_arr = np.array(gen_fwl_vec)
        enc_arr = np.array(enc_fwl_vec)
        sign_flips = np.sum(np.sign(gen_arr) != np.sign(enc_arr))
        agreement_rho, agreement_p = stats.spearmanr(gen_arr, enc_arr)
        print(f"  {model_short}: encode-gen FWL agreement rho={agreement_rho:.3f} "
              f"(p={agreement_p:.2e}), sign flips={sign_flips}/60")

        # H3 on encode features
        types_arr = np.array(pt_gen)
        meta_mask = types_arr == "metacognitive"
        if meta_mask.sum() > 5:
            tsv_idx = PRIMARY_FEATURES.index("top_sv_ratio")
            meta_vals = feat_enc[meta_mask, tsv_idx]
            other_vals = feat_enc[~meta_mask, tsv_idx]
            meta_vals = meta_vals[~np.isnan(meta_vals)]
            other_vals = other_vals[~np.isnan(other_vals)]
            if len(meta_vals) > 2 and len(other_vals) > 2:
                stat, p = stats.mannwhitneyu(meta_vals, other_vals)
                d = _cohens_d(meta_vals, other_vals)
                print(f"    H3 encode: d={d:.3f}, p={p:.2e} "
                      f"({'SURVIVES' if p < 0.05 else 'FAILS'})")

        all_results[f"{model_short}_circularity"] = {
            "encode_gen_agreement_rho": round(float(agreement_rho), 4),
            "encode_gen_agreement_p": float(agreement_p),
            "sign_flips": int(sign_flips),
        }

    # Cross-scale consistency (H6)
    if len(model_dirs) > 1:
        print(f"\n--- Cross-Scale Consistency (H6) ---")
        h6 = test_cross_scale_consistency(results_dir, model_dirs)
        for comp, result in h6.get("comparisons", {}).items():
            status = "CONSISTENT" if result["consistent"] else "DIVERGENT"
            print(f"  {comp}: rho={result['rho']:.3f} ({status})")
        all_results["cross_scale"] = h6

    # Save
    def _json_default(obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    out_path = os.path.join(output_dir, "concordance_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'=' * 70}")
    print("CORRELATION SUMMARY (top 10 by |rho_fwl|)")
    print(f"{'=' * 70}")

    for model_short, model_results in all_results.items():
        if model_short == "cross_scale":
            continue
        corrs = model_results.get("correlations", {}).get("correlations", {})
        if not corrs:
            continue

        sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]["rho_fwl"]), reverse=True)
        print(f"\n{model_short}:")
        print(f"  {'Pair':<25s} {'rho_raw':>8s} {'rho_fwl':>8s} {'p_fwl':>10s} {'flag':>5s}")
        for key, v in sorted_corrs[:10]:
            flag = "***" if v["confound_flag"] else ""
            sig = "*" if v["significant_fwl"] else ""
            print(f"  {key:<25s} {v['rho_raw']:>8.3f} {v['rho_fwl']:>8.3f} {v['p_fwl']:>10.2e} {flag}{sig}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concordance analysis")
    parser.add_argument("--results-dir", default="results/concordance")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_full_analysis(args.results_dir, args.output_dir)
