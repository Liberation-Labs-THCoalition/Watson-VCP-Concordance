"""
Encode-Generation Reversal: Detailed analysis for Paper 2.

Key question: WHY do Mistral and Qwen 7B show 57-63% sign reversal
while Llama shows only 12%? Is it architecture, VCP dimensionality,
or prompt-type dependent?

Also: Which VCP dimensions GAIN vs LOSE geometric coupling during generation?
"""

import json
import os
import sys
import numpy as np
from scipy import stats
from collections import defaultdict

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


def per_type_reversal(results_dir, model_short, model_name):
    """Analyze reversal separately by prompt type."""
    results = load_phase_results(results_dir, model_short, phase="a")
    if not results:
        return {}

    # Get encode and generation features
    vcp_gen, feat_gen, tc_gen, rl_gen, pt_gen = results_to_arrays(results, phase="generation")
    vcp_enc, feat_enc, tc_enc, rl_enc, pt_enc = results_to_arrays(results, phase="encode")

    N = min(vcp_gen.shape[0], vcp_enc.shape[0])
    vcp = vcp_gen[:N]
    feat_g = feat_gen[:N]
    feat_e = feat_enc[:N]
    # M1 FIX: Phase-specific confounds
    tc_g = tc_gen[:N]
    rl_g = rl_gen[:N]
    tc_e = tc_enc[:N]
    rl_e = rl_enc[:N]
    pt = pt_gen[:N]
    conf_gen = np.column_stack([tc_g, rl_g])
    conf_enc = np.column_stack([tc_e, rl_e])

    types = sorted(set(pt))
    result = {}

    for ptype in types:
        mask = np.array([p == ptype for p in pt])
        n_type = mask.sum()
        if n_type < 10:
            continue

        v_t = vcp[mask]
        fg_t = feat_g[mask]
        fe_t = feat_e[mask]
        c_gen_t = conf_gen[mask]
        c_enc_t = conf_enc[mask]

        flips = 0
        total = 0
        gains = []  # pairs where |rho| increases encode→gen
        losses = []  # pairs where |rho| decreases

        for i, dim in enumerate(DIM_LETTERS):
            for j, feat in enumerate(PRIMARY_FEATURES):
                valid = ~(np.isnan(v_t[:, i]) | np.isnan(fg_t[:, j]) | np.isnan(fe_t[:, j]))
                if valid.sum() < 5:
                    continue

                v = v_t[valid, i]
                fg = fg_t[valid, j]
                fe = fe_t[valid, j]
                c_g = c_gen_t[valid]
                c_e = c_enc_t[valid]

                # M1 FIX: Phase-specific FWL
                v_r = fwl_residualize(v, c_g)  # VCP from gen phase
                fg_r = fwl_residualize(fg, c_g)  # Gen features, gen confounds
                fe_r = fwl_residualize(fe, c_e)  # Encode features, encode confounds

                rho_e, _ = stats.spearmanr(v_r, fe_r)
                rho_g, _ = stats.spearmanr(v_r, fg_r)

                total += 1
                # STRICT criterion (C1 fix): both must exceed threshold
                if rho_e * rho_g < 0 and (abs(rho_e) > 0.15 and abs(rho_g) > 0.15):
                    flips += 1

                if abs(rho_g) > abs(rho_e):
                    gains.append((dim, feat, rho_e, rho_g))
                else:
                    losses.append((dim, feat, rho_e, rho_g))

        result[ptype] = {
            "n": n_type,
            "flips": flips,
            "total": total,
            "flip_rate": round(flips / max(total, 1), 3),
            "gains": len(gains),
            "losses": len(losses),
            "top_gains": sorted(gains, key=lambda x: abs(x[3]) - abs(x[2]), reverse=True)[:5],
            "top_losses": sorted(losses, key=lambda x: abs(x[2]) - abs(x[3]), reverse=True)[:5],
        }

    return result


def coupling_strength_by_phase(results_dir, model_short, model_name):
    """Compare total VCP-geometry coupling strength between encode and generation.

    Uses mean |rho_fwl| across all 60 pairs as coupling metric.
    """
    results = load_phase_results(results_dir, model_short, phase="a")
    if not results:
        return {}

    vcp_gen, feat_gen, tc_gen, rl_gen, pt_gen = results_to_arrays(results, phase="generation")
    vcp_enc, feat_enc, tc_enc, rl_enc, pt_enc = results_to_arrays(results, phase="encode")

    N = min(vcp_gen.shape[0], vcp_enc.shape[0])
    vcp = vcp_gen[:N]
    fg = feat_gen[:N]
    fe = feat_enc[:N]
    # M1 FIX: Phase-specific confounds
    conf_enc = np.column_stack([tc_enc[:N], rl_enc[:N]])
    conf_gen = np.column_stack([tc_gen[:N], rl_gen[:N]])

    encode_rhos = []
    gen_rhos = []
    per_dim = defaultdict(lambda: {"encode": [], "gen": []})

    for i, dim in enumerate(DIM_LETTERS):
        for j, feat in enumerate(PRIMARY_FEATURES):
            valid = ~(np.isnan(vcp[:, i]) | np.isnan(fg[:, j]) | np.isnan(fe[:, j]))
            if valid.sum() < 10:
                continue

            v = vcp[valid, i]
            fgv = fg[valid, j]
            fev = fe[valid, j]
            c_g = conf_gen[valid]
            c_e = conf_enc[valid]

            # M1 FIX: Phase-specific FWL
            v_r = fwl_residualize(v, c_g)  # VCP from gen phase
            fg_r = fwl_residualize(fgv, c_g)
            fe_r = fwl_residualize(fev, c_e)  # Encode features, encode confounds

            rho_e, _ = stats.spearmanr(v_r, fe_r)
            rho_g, _ = stats.spearmanr(v_r, fg_r)

            encode_rhos.append(abs(rho_e))
            gen_rhos.append(abs(rho_g))
            per_dim[dim]["encode"].append(abs(rho_e))
            per_dim[dim]["gen"].append(abs(rho_g))

    result = {
        "model": model_name,
        "mean_encode_coupling": round(float(np.mean(encode_rhos)), 4),
        "mean_gen_coupling": round(float(np.mean(gen_rhos)), 4),
        "coupling_ratio": round(float(np.mean(gen_rhos) / max(np.mean(encode_rhos), 0.001)), 3),
        "gen_stronger": float(np.mean(gen_rhos)) > float(np.mean(encode_rhos)),
        "per_dimension": {},
    }

    for dim in DIM_LETTERS:
        enc_vals = per_dim[dim]["encode"]
        gen_vals = per_dim[dim]["gen"]
        if enc_vals and gen_vals:
            result["per_dimension"][dim] = {
                "encode_mean_abs_rho": round(float(np.mean(enc_vals)), 4),
                "gen_mean_abs_rho": round(float(np.mean(gen_vals)), 4),
                "strengthens": float(np.mean(gen_vals)) > float(np.mean(enc_vals)),
            }

    return result


def analyze_vcp_variance_by_prompt_type(results_dir, model_short, model_name):
    """Check if VCP self-report variance differs by prompt type.

    If metacognitive prompts produce MORE VCP variance, that could explain
    why their geometry is distinct — they're genuinely harder to self-rate.
    """
    results = load_phase_results(results_dir, model_short, phase="a")
    if not results:
        return {}

    vcp, _, _, _, pt = results_to_arrays(results, phase="generation")
    types = sorted(set(pt))

    result = {"model": model_name, "per_type": {}}

    for ptype in types:
        mask = np.array([p == ptype for p in pt])
        V = vcp[mask]

        valid_rows = ~np.any(np.isnan(V), axis=1)
        V = V[valid_rows]
        if V.shape[0] < 5:
            continue

        # Mean rating, variance, and inter-dimension spread
        means = V.mean(axis=0)
        within_response_std = np.std(V, axis=1)  # How much dims vary per response

        result["per_type"][ptype] = {
            "n": int(V.shape[0]),
            "mean_rating": round(float(np.mean(means)), 2),
            "mean_within_std": round(float(np.mean(within_response_std)), 3),
            "between_trials_std": round(float(np.std(V.mean(axis=1))), 3),
            "per_dim_means": {
                DIM_LETTERS[i]: round(float(means[i]), 2) for i in range(len(DIM_LETTERS))
            },
        }

    return result


def run_reversal_analysis(results_dir):
    """Run all reversal-related analyses."""
    print("=" * 70)
    print("ENCODE-GENERATION REVERSAL ANALYSIS")
    print("=" * 70)

    all_results = {}

    for model_short, model_name in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")

        # 1. Per-type reversal
        print("\n  --- Per-Type Reversal ---")
        ptype_rev = per_type_reversal(results_dir, model_short, model_name)
        for ptype, data in sorted(ptype_rev.items()):
            print(f"  {ptype}: {data['flips']}/{data['total']} flips ({data['flip_rate']:.0%}), "
                  f"gains={data['gains']}, losses={data['losses']}")

        # 2. Coupling strength comparison
        print("\n  --- Coupling Strength ---")
        coupling = coupling_strength_by_phase(results_dir, model_short, model_name)
        if coupling:
            print(f"  Encode mean |rho|: {coupling['mean_encode_coupling']:.4f}")
            print(f"  Generation mean |rho|: {coupling['mean_gen_coupling']:.4f}")
            print(f"  Ratio (gen/enc): {coupling['coupling_ratio']:.2f}x")
            print(f"  Dims that strengthen during generation:")
            for dim, data in coupling["per_dimension"].items():
                arrow = "UP" if data["strengthens"] else "DOWN"
                print(f"    {dim}: {data['encode_mean_abs_rho']:.3f} -> {data['gen_mean_abs_rho']:.3f} {arrow}")

        # 3. VCP variance by type
        print("\n  --- VCP Variance by Prompt Type ---")
        vcp_var = analyze_vcp_variance_by_prompt_type(results_dir, model_short, model_name)
        if vcp_var.get("per_type"):
            for ptype, data in sorted(vcp_var["per_type"].items()):
                print(f"  {ptype} (n={data['n']}): mean={data['mean_rating']:.1f}, "
                      f"within_std={data['mean_within_std']:.2f}, "
                      f"between_std={data['between_trials_std']:.2f}")

        all_results[model_name] = {
            "per_type_reversal": ptype_rev,
            "coupling_strength": coupling,
            "vcp_variance": vcp_var,
        }

    # Save
    out_dir = os.path.join(results_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "reversal_analysis_results.json")

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

    run_reversal_analysis(args.results_dir)
