"""
Circularity audit: do concordance patterns hold on encode-phase features
(which cannot have VCP token circularity)?

Compares generation_features vs encode_features correlations with VCP.
Also runs H3 test on encode-phase features.
"""

import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concordance.features import PRIMARY_FEATURES
from concordance.vcp_parser import VCP_V2_DIMENSIONS


def fwl_residualize(y, X_confound):
    X = np.column_stack([np.ones(len(y)), X_confound])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return y - X @ beta
    except np.linalg.LinAlgError:
        return y


def load_trials(results_dir, model_short):
    phase_dir = os.path.join(results_dir, model_short, "phase_a")
    if not os.path.isdir(phase_dir):
        return []
    trials = []
    for f in sorted(glob.glob(os.path.join(phase_dir, "*.json"))):
        if os.path.basename(f).startswith("_") or f.endswith("summary.json"):
            continue
        with open(f) as fh:
            trials.append(json.load(fh))
    return trials


def extract_arrays(trials, phase_key):
    """Extract VCP + feature arrays for a given phase."""
    dim_letters = list(VCP_V2_DIMENSIONS.keys())
    valid = [t for t in trials if t.get("vcp_parse_quality") == "clean"]
    N = len(valid)
    vcp = np.zeros((N, len(dim_letters)))
    feat = np.zeros((N, len(PRIMARY_FEATURES)))
    tc = np.zeros(N)
    rl = np.zeros(N)
    types = []

    for i, t in enumerate(valid):
        ratings = t.get("vcp_ratings", {})
        for j, letter in enumerate(dim_letters):
            vcp[i, j] = ratings.get(letter, np.nan)

        f = t.get(phase_key, {})
        for j, fname in enumerate(PRIMARY_FEATURES):
            feat[i, j] = f.get(fname, np.nan)

        if phase_key == "encode_features":
            tc[i] = f.get("n_tokens", f.get("n_input_tokens", 0))
        else:
            tc[i] = t.get("n_tokens", f.get("n_tokens", 0))
        rl[i] = t.get("response_length", 0)
        types.append(t.get("prompt_type", "unknown"))

    return vcp, feat, tc, rl, types


def run_audit(results_dir):
    model_dirs = [d for d in os.listdir(results_dir)
                  if os.path.isdir(os.path.join(results_dir, d))
                  and d not in ("analysis", "pilot")]

    dim_letters = list(VCP_V2_DIMENSIONS.keys())
    SEP = "=" * 80

    print(SEP)
    print("CIRCULARITY AUDIT: ENCODE vs GENERATION FEATURES")
    print(SEP)

    for model_short in sorted(model_dirs):
        trials = load_trials(results_dir, model_short)
        if not trials:
            continue

        print(f"\n{'—' * 60}")
        print(f"Model: {model_short}")
        print(f"{'—' * 60}")

        # Get arrays for both phases
        vcp_gen, feat_gen, tc_gen, rl_gen, types_gen = extract_arrays(trials, "generation_features")
        vcp_enc, feat_enc, tc_enc, rl_enc, types_enc = extract_arrays(trials, "encode_features")

        N = vcp_gen.shape[0]
        print(f"Valid trials: {N}")

        # --------------------------------------------------------
        # 1. Raw and FWL correlations for BOTH phases
        # --------------------------------------------------------
        gen_rhos = np.zeros((len(dim_letters), len(PRIMARY_FEATURES)))
        enc_rhos = np.zeros((len(dim_letters), len(PRIMARY_FEATURES)))
        gen_fwl = np.zeros((len(dim_letters), len(PRIMARY_FEATURES)))
        enc_fwl = np.zeros((len(dim_letters), len(PRIMARY_FEATURES)))

        for i in range(len(dim_letters)):
            for j in range(len(PRIMARY_FEATURES)):
                v = vcp_gen[:, i]
                fg = feat_gen[:, j]
                fe = feat_enc[:, j]
                valid_g = ~(np.isnan(v) | np.isnan(fg))
                valid_e = ~(np.isnan(v) | np.isnan(fe))

                if valid_g.sum() > 10:
                    gen_rhos[i, j], _ = stats.spearmanr(v[valid_g], fg[valid_g])
                    # FWL
                    conf = np.column_stack([tc_gen[valid_g], rl_gen[valid_g]])
                    v_r = fwl_residualize(v[valid_g], conf)
                    f_r = fwl_residualize(fg[valid_g], conf)
                    gen_fwl[i, j], _ = stats.spearmanr(v_r, f_r)

                if valid_e.sum() > 10:
                    enc_rhos[i, j], _ = stats.spearmanr(v[valid_e], fe[valid_e])
                    # FWL for encode: use n_input_tokens only
                    enc_tc = tc_enc[valid_e]
                    # Encode features don't depend on response length, but VCP does
                    # So we still need to control for response_length on VCP side
                    conf_enc = np.column_stack([enc_tc, rl_enc[valid_e]])
                    v_r = fwl_residualize(v[valid_e], conf_enc)
                    f_r = fwl_residualize(fe[valid_e], conf_enc)
                    enc_fwl[i, j], _ = stats.spearmanr(v_r, f_r)

        # --------------------------------------------------------
        # 2. Compare: how many correlations change sign?
        # --------------------------------------------------------
        sign_flips_raw = np.sum(np.sign(gen_rhos) != np.sign(enc_rhos))
        sign_flips_fwl = np.sum(np.sign(gen_fwl) != np.sign(enc_fwl))
        mean_diff_raw = np.mean(np.abs(gen_rhos - enc_rhos))
        mean_diff_fwl = np.mean(np.abs(gen_fwl - enc_fwl))

        print(f"\n  RAW correlations:")
        print(f"    Sign flips (encode vs generation): {sign_flips_raw}/60")
        print(f"    Mean |rho_gen - rho_enc|: {mean_diff_raw:.4f}")

        print(f"\n  FWL correlations:")
        print(f"    Sign flips (encode vs generation): {sign_flips_fwl}/60")
        print(f"    Mean |rho_gen_fwl - rho_enc_fwl|: {mean_diff_fwl:.4f}")

        # Correlation between encode and generation rho vectors
        flat_gen = gen_fwl.flatten()
        flat_enc = enc_fwl.flatten()
        rho_agreement, p_agreement = stats.spearmanr(flat_gen, flat_enc)
        print(f"    Spearman(gen_fwl, enc_fwl) across 60 pairs: rho={rho_agreement:.4f}, p={p_agreement:.2e}")

        # --------------------------------------------------------
        # 3. Top 5 FWL correlations: do they survive in encode?
        # --------------------------------------------------------
        print(f"\n  Top 5 generation FWL correlations — encode comparison:")
        print(f"  {'Pair':<20s} {'gen_fwl':>8s} {'enc_fwl':>8s} {'delta':>8s} {'sign_match':>10s}")
        print(f"  {'-'*58}")

        pairs = []
        for i in range(len(dim_letters)):
            for j in range(len(PRIMARY_FEATURES)):
                pairs.append((dim_letters[i], PRIMARY_FEATURES[j],
                               gen_fwl[i, j], enc_fwl[i, j]))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        for dim, feat, g, e in pairs[:5]:
            delta = g - e
            sign_ok = "YES" if (g > 0) == (e > 0) else "NO"
            print(f"  {dim}_{feat:<15s} {g:>8.3f} {e:>8.3f} {delta:>+8.3f} {sign_ok:>10s}")

        # --------------------------------------------------------
        # 4. H3 test on encode-phase features
        # --------------------------------------------------------
        types_arr = np.array(types_gen)
        meta_mask = types_arr == "metacognitive"
        if meta_mask.sum() > 5:
            tsv_idx = PRIMARY_FEATURES.index("top_sv_ratio")
            meta_vals = feat_enc[meta_mask, tsv_idx]
            other_vals = feat_enc[~meta_mask, tsv_idx]
            meta_vals = meta_vals[~np.isnan(meta_vals)]
            other_vals = other_vals[~np.isnan(other_vals)]

            if len(meta_vals) > 2 and len(other_vals) > 2:
                stat, p = stats.mannwhitneyu(meta_vals, other_vals)
                d = (np.mean(meta_vals) - np.mean(other_vals)) / np.sqrt(
                    (np.var(meta_vals, ddof=1) + np.var(other_vals, ddof=1)) / 2
                ) if (np.var(meta_vals, ddof=1) + np.var(other_vals, ddof=1)) > 0 else 0

                print(f"\n  H3 on ENCODE features (no circularity possible):")
                print(f"    meta top_sv_ratio mean: {np.mean(meta_vals):.6f}")
                print(f"    other top_sv_ratio mean: {np.mean(other_vals):.6f}")
                print(f"    Cohen's d: {d:.3f}")
                print(f"    Mann-Whitney p: {p:.2e}")
                print(f"    H3 survives encode phase: {'YES' if p < 0.05 else 'NO'}")

            # Also do generation-phase H3 for comparison
            meta_vals_g = feat_gen[meta_mask, tsv_idx]
            other_vals_g = feat_gen[~meta_mask, tsv_idx]
            meta_vals_g = meta_vals_g[~np.isnan(meta_vals_g)]
            other_vals_g = other_vals_g[~np.isnan(other_vals_g)]

            if len(meta_vals_g) > 2 and len(other_vals_g) > 2:
                stat_g, p_g = stats.mannwhitneyu(meta_vals_g, other_vals_g)
                d_g = (np.mean(meta_vals_g) - np.mean(other_vals_g)) / np.sqrt(
                    (np.var(meta_vals_g, ddof=1) + np.var(other_vals_g, ddof=1)) / 2
                ) if (np.var(meta_vals_g, ddof=1) + np.var(other_vals_g, ddof=1)) > 0 else 0
                print(f"\n  H3 on GENERATION features (for comparison):")
                print(f"    Cohen's d: {d_g:.3f}")
                print(f"    Mann-Whitney p: {p_g:.2e}")

        # --------------------------------------------------------
        # 5. Encode-generation feature correlation (how similar?)
        # --------------------------------------------------------
        print(f"\n  Encode vs Generation feature similarity (per feature):")
        print(f"  {'Feature':<20s} {'Pearson r':>10s} {'Mean enc':>10s} {'Mean gen':>10s}")
        print(f"  {'-'*55}")
        for j, fname in enumerate(PRIMARY_FEATURES):
            e_col = feat_enc[:, j]
            g_col = feat_gen[:, j]
            valid = ~(np.isnan(e_col) | np.isnan(g_col))
            if valid.sum() > 5:
                r, _ = stats.pearsonr(e_col[valid], g_col[valid])
                print(f"  {fname:<20s} {r:>10.4f} {np.mean(e_col[valid]):>10.2f} {np.mean(g_col[valid]):>10.2f}")

    print(f"\n{SEP}")
    print("INTERPRETATION")
    print(SEP)
    print("""
If encode-phase and generation-phase correlations are highly concordant
(rho > 0.7, few sign flips), then VCP token circularity is NOT driving results.
The encode phase captures prompt processing BEFORE any generation occurs —
no VCP tokens, no response tokens, zero circularity.

If H3 survives on encode features, the meta-cognitive geometry effect is
a property of prompt encoding, not generation, making circularity irrelevant.
""")


if __name__ == "__main__":
    results_dir = "results/concordance"
    if not os.path.isdir(results_dir):
        results_dir = "C:/Users/Thomas/Desktop/Watson-VCP-Concordance/results/concordance"
    run_audit(results_dir)
