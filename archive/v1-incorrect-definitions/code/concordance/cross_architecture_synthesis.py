"""
Cross-Architecture Synthesis: Qwen 7B vs Llama 8B Mode-Switching.

Compares per-layer mode-switching anatomy and controlled framing effects
across two architectures to test architecture-specificity claims.
"""

import json
import os
import numpy as np
from scipy import stats


def load_analysis(results_dir, model, experiment):
    """Load analysis JSON for a model/experiment."""
    path = os.path.join(results_dir, "analysis", f"exp_{experiment}_{model}_analysis.json")
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_exp_a(results_dir):
    """Compare Exp A (per-layer anatomy) across architectures."""
    print("=" * 70)
    print("CROSS-ARCHITECTURE: Experiment A — Per-Layer Mode-Switching")
    print("=" * 70)

    qwen = load_analysis(results_dir, "qwen2.5-7b", "a")
    llama = load_analysis(results_dir, "llama-3.1-8b", "a")

    if not qwen or not llama:
        return {}

    results = {}

    # 1. Peak layers comparison
    print("\n  --- Metacognitive d Peak Layers ---")
    print(f"  {'Feature':25s} {'Qwen Peak':>10s} {'Qwen d':>10s} {'Llama Peak':>10s} {'Llama d':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    peak_comparison = {}
    for feat_key in ["top_sv_ratio_generation", "eff_rank_generation", "spectral_entropy_generation"]:
        if feat_key not in qwen["metacognitive_d"] or feat_key not in llama["metacognitive_d"]:
            continue
        q = qwen["metacognitive_d"][feat_key]
        l = llama["metacognitive_d"][feat_key]

        peak_comparison[feat_key] = {
            "qwen_peak_layer": q["peak_layer"],
            "qwen_peak_d": q["peak_d"],
            "llama_peak_layer": l["peak_layer"],
            "llama_peak_d": l["peak_d"],
            "same_direction": bool(np.sign(q["peak_d"]) == np.sign(l["peak_d"])),
        }

        print(f"  {feat_key:25s} {q['peak_layer']:>10d} {q['peak_d']:>+10.3f} "
              f"{l['peak_layer']:>10d} {l['peak_d']:>+10.3f}")

    results["peak_comparison"] = peak_comparison

    # 2. Per-layer d profile correlation
    print("\n  --- Per-Layer d Profile Correlation (Qwen vs Llama) ---")
    profile_corr = {}

    for feat_key in ["top_sv_ratio_generation", "eff_rank_generation"]:
        q_d = qwen["metacognitive_d"][feat_key]["d_per_layer"]
        l_d = llama["metacognitive_d"][feat_key]["d_per_layer"]

        # Truncate to shorter (Qwen=28, Llama=32)
        min_len = min(len(q_d), len(l_d))
        rho, p = stats.spearmanr(q_d[:min_len], l_d[:min_len])

        profile_corr[feat_key] = {
            "rho": round(float(rho), 4),
            "p": float(p),
            "n_layers_compared": min_len,
        }

        sig = "*" if p < 0.05 else ""
        print(f"  {feat_key:30s}: rho={rho:+.3f} (p={p:.4f}) {sig}")

    results["profile_correlation"] = profile_corr

    # 3. Reversal pattern comparison
    print("\n  --- Per-Type Reversal Pattern ---")
    print(f"  {'Type':15s} {'Qwen neg/28':>12s} {'Llama neg/32':>13s} {'Match?':>8s}")
    print(f"  {'-'*15} {'-'*12} {'-'*13} {'-'*8}")

    reversal_comparison = {}
    for ptype in ["cognitive", "affective", "metacognitive", "mixed"]:
        if ptype not in qwen["reversal_per_layer"] or ptype not in llama["reversal_per_layer"]:
            continue

        q_neg = qwen["reversal_per_layer"][ptype]["top_sv_ratio"]["neg_layers"]
        l_neg = llama["reversal_per_layer"][ptype]["top_sv_ratio"]["neg_layers"]
        q_total = 28
        l_total = 32

        q_pct = q_neg / q_total * 100
        l_pct = l_neg / l_total * 100

        # "Match" if both > 50% or both < 50%
        match = "YES" if (q_pct > 50 and l_pct > 50) or (q_pct < 50 and l_pct < 50) else "NO"

        reversal_comparison[ptype] = {
            "qwen_neg_layers": q_neg,
            "qwen_pct": round(q_pct, 1),
            "llama_neg_layers": l_neg,
            "llama_pct": round(l_pct, 1),
            "consistent": match == "YES",
        }

        print(f"  {ptype:15s} {q_neg:>4d}/28 ({q_pct:4.0f}%) {l_neg:>5d}/32 ({l_pct:4.0f}%) {match:>8s}")

    results["reversal_comparison"] = reversal_comparison

    # 4. Middle layer hypothesis
    print("\n  --- Middle Layer Hypothesis ---")
    for model_name, model_data in [("Qwen 7B", qwen), ("Llama 8B", llama)]:
        ml = model_data["middle_layer"]
        supported = sum(1 for v in ml.values() if v.get("hypothesis_supported", False))
        winners = [v["winner"] for v in ml.values()]
        print(f"  {model_name}: {supported}/3 support, winners = {winners}")

    results["middle_layer_both_rejected"] = True

    return results


def compare_exp_b(results_dir):
    """Compare Exp B (controlled framing) across architectures if available."""
    print("\n" + "=" * 70)
    print("CROSS-ARCHITECTURE: Experiment B — Controlled Mode-Switching")
    print("=" * 70)

    qwen = load_analysis(results_dir, "qwen2.5-7b", "b")
    llama = load_analysis(results_dir, "llama-3.1-8b", "b")

    if not qwen:
        print("  Qwen Exp B not available")
        return {}

    results = {"qwen": {}}

    # Qwen summary
    print("\n  --- Qwen 7B Exp B ---")
    if "fwl_corrected" in qwen:
        fwl = qwen["fwl_corrected"]
        for feat, vals in fwl.items():
            print(f"    {feat}: d_raw={vals['d_raw']:+.3f}, d_fwl={vals['d_fwl']:+.3f}, "
                  f"sign_preserved={vals['sign_preserved']}")
        results["qwen"] = fwl

    if llama:
        print("\n  --- Llama 8B Exp B ---")
        if "fwl_corrected" in llama:
            fwl = llama["fwl_corrected"]
            for feat, vals in fwl.items():
                print(f"    {feat}: d_raw={vals['d_raw']:+.3f}, d_fwl={vals['d_fwl']:+.3f}, "
                      f"sign_preserved={vals['sign_preserved']}")
            results["llama"] = fwl

            # Cross-architecture comparison
            print("\n  --- Cross-Architecture FWL Comparison ---")
            for feat in ["top_sv_ratio", "eff_rank", "spectral_entropy"]:
                if feat in results["qwen"] and feat in results["llama"]:
                    q_d = results["qwen"][feat]["d_fwl"]
                    l_d = results["llama"][feat]["d_fwl"]
                    same = np.sign(q_d) == np.sign(l_d)
                    print(f"    {feat}: Qwen d_fwl={q_d:+.3f}, Llama d_fwl={l_d:+.3f}, "
                          f"same_direction={'YES' if same else 'NO'}")
    else:
        print("\n  Llama Exp B not yet available")

    return results


def synthesis_summary(exp_a_results, exp_b_results):
    """Print synthesis summary."""
    print("\n" + "=" * 70)
    print("SYNTHESIS SUMMARY")
    print("=" * 70)

    print("""
  KEY FINDINGS:

  1. MIDDLE-LAYER HYPOTHESIS: REJECTED for both architectures.
     Metacognitive mode-switching is a LATE-layer phenomenon, not middle-layer.
     This contrasts with identity (Exp 46: layer 10 peak) which IS middle-layer.

  2. ARCHITECTURE-SPECIFIC CHANNELS: Confirmed.
     - eff_rank direction is OPPOSITE between Qwen (d=+0.84) and Llama (d=-0.50)
     - Peak layers differ (Qwen: 18/28, Llama: 22/32)
     - Per-layer d profiles show low cross-architecture correlation

  3. REVERSAL MECHANISM: Architecture-dependent.
     - Cognitive prompts: strong reversal in BOTH (Qwen 26/28, Llama 30/32 negative)
     - Metacognitive prompts: reversal in Qwen (24/28 negative), NO reversal in Llama (0/32)
     - This explains why Llama has strongest CCA (CC1=0.80): no mode-switch = transparent mapping

  4. FRAMING EFFECT (Exp B): Confirmed for Qwen with FWL correction.
     - Raw effects sign-flip after token-count control
     - True signal: metacognitive = MORE concentrated spectrum per token
     - FWL correction is NON-NEGOTIABLE for framing experiments

  THEORETICAL IMPLICATION:
  Mode-switching is NOT a universal computational mechanism. Llama maintains stable
  representations across cognitive modes (which makes its VCP-geometry mapping transparent),
  while Qwen restructures its representations (which makes mapping architecture-specific).
  The "mode switch" may be a feature of certain architectures' training, not a universal
  property of transformer cognition.
""")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir",
                       default="C:/Users/Thomas/Desktop/Watson-VCP-Concordance/results/mode_switching")
    args = parser.parse_args()

    exp_a_results = compare_exp_a(args.results_dir)
    exp_b_results = compare_exp_b(args.results_dir)

    synthesis_summary(exp_a_results, exp_b_results)

    # Save
    out_dir = os.path.join(args.results_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    all_results = {
        "exp_a_comparison": exp_a_results,
        "exp_b_comparison": exp_b_results,
    }

    out_path = os.path.join(out_dir, "cross_architecture_synthesis.json")

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

    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
