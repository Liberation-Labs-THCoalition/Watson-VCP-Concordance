"""
Analyze Experiment D (v2): Length-Matched Generation Trajectory.

Tests: Does metacognitive framing alter KV-cache geometry trajectory
even when input token counts are matched?

Key analyses:
1. Verify length matching (encode-phase d ~ 0)
2. Trajectory divergence over generation checkpoints
3. Per-feature divergence timecourse
4. Cross-model consistency (Qwen vs Llama)
"""

import json
import os
import glob
import numpy as np
from scipy import stats


FEATURES = ["key_norm", "norm_per_token", "eff_rank", "top_sv_ratio",
            "spectral_entropy", "layer_variance"]


def load_exp_d_results(results_dir, model, subdir="exp_d_matched"):
    """Load Exp D paired results for a model."""
    exp_dir = os.path.join(results_dir, model, subdir)
    pairs = []
    for f in sorted(glob.glob(os.path.join(exp_dir, "pair_*.json"))):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if "cognitive" in data and "metacognitive" in data:
            if "error" not in data["cognitive"] and "error" not in data["metacognitive"]:
                pairs.append(data)
    return pairs


def cohens_d(a, b):
    """Paired Cohen's d."""
    diff = np.array(a) - np.array(b)
    if np.std(diff, ddof=1) == 0:
        return 0.0
    return float(np.mean(diff) / np.std(diff, ddof=1))


def analyze_length_matching(pairs, model_name):
    """Verify that input token counts are matched between conditions."""
    print(f"\n{'='*70}")
    print(f"ANALYSIS 1: Length Matching Verification — {model_name}")
    print(f"{'='*70}")

    cog_input = []
    meta_input = []
    diffs = []

    for p in pairs:
        c_len = p["cognitive"]["input_len"]
        m_len = p["metacognitive"]["input_len"]
        cog_input.append(c_len)
        meta_input.append(m_len)
        diffs.append(c_len - m_len)

    diffs = np.array(diffs)
    print(f"\n  Pairs loaded: {len(pairs)}")
    print(f"  Cognitive input tokens:     mean={np.mean(cog_input):.1f}, "
          f"range=[{min(cog_input)},{max(cog_input)}]")
    print(f"  Metacognitive input tokens: mean={np.mean(meta_input):.1f}, "
          f"range=[{min(meta_input)},{max(meta_input)}]")
    print(f"  Token difference (cog - meta): mean={np.mean(diffs):.2f}, "
          f"max|diff|={max(abs(diffs))}")

    if max(abs(diffs)) <= 2:
        print("  PASS: Token counts matched within 2 tokens")
    else:
        print(f"  WARNING: Max token difference = {max(abs(diffs))}")

    # Encode-phase feature comparison
    print(f"\n  Encode-phase feature comparison (should show d ~ 0):")
    encode_results = {}
    for feat in FEATURES:
        cog_vals = [p["cognitive"]["encode_features"][feat] for p in pairs]
        meta_vals = [p["metacognitive"]["encode_features"][feat] for p in pairs]
        d = cohens_d(meta_vals, cog_vals)
        t, p_val = stats.ttest_rel(meta_vals, cog_vals)
        encode_results[feat] = {"d": d, "t": t, "p": p_val}
        flag = " ***CONFOUND***" if abs(d) > 0.5 else ""
        print(f"    {feat:20s}: d={d:+.3f}, t={t:+.3f}, p={p_val:.4f}{flag}")

    return encode_results


def analyze_trajectory_divergence(pairs, model_name):
    """Track how cognitive vs metacognitive trajectories diverge over generation."""
    print(f"\n{'='*70}")
    print(f"ANALYSIS 2: Trajectory Divergence — {model_name}")
    print(f"{'='*70}")

    # Collect checkpoints common to all pairs
    all_checkpoints = set()
    for p in pairs:
        for cp in p["cognitive"]["trajectory"]:
            all_checkpoints.add(cp["checkpoint"])
    all_checkpoints = sorted(all_checkpoints)

    # For each checkpoint, compute per-feature divergence
    results_by_checkpoint = {}

    for cp in all_checkpoints:
        feat_results = {}
        for feat in FEATURES:
            cog_vals = []
            meta_vals = []
            for p in pairs:
                cog_cp = [t for t in p["cognitive"]["trajectory"] if t["checkpoint"] == cp]
                meta_cp = [t for t in p["metacognitive"]["trajectory"] if t["checkpoint"] == cp]
                if cog_cp and meta_cp:
                    cog_vals.append(cog_cp[0][feat])
                    meta_vals.append(meta_cp[0][feat])

            if len(cog_vals) >= 5:
                d = cohens_d(meta_vals, cog_vals)
                t, p_val = stats.ttest_rel(meta_vals, cog_vals)
                try:
                    w, p_w = stats.wilcoxon(np.array(meta_vals) - np.array(cog_vals))
                except ValueError:
                    w, p_w = 0, 1.0
                feat_results[feat] = {
                    "d": d, "t": t, "p_t": p_val, "p_w": p_w,
                    "n": len(cog_vals),
                    "cog_mean": float(np.mean(cog_vals)),
                    "meta_mean": float(np.mean(meta_vals)),
                }

        results_by_checkpoint[cp] = feat_results

    # Print trajectory table
    print(f"\n  Cohen's d at each checkpoint (meta - cog):")
    header = f"  {'CP':>5s}"
    for feat in FEATURES:
        header += f" | {feat[:12]:>12s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cp in sorted(results_by_checkpoint.keys()):
        row = f"  {cp:5d}"
        for feat in FEATURES:
            if feat in results_by_checkpoint[cp]:
                d = results_by_checkpoint[cp][feat]["d"]
                sig = "*" if results_by_checkpoint[cp][feat]["p_t"] < 0.05 else " "
                row += f" | {d:+11.3f}{sig}"
            else:
                row += f" | {'N/A':>12s}"
        print(row)

    # Identify earliest divergence point
    print(f"\n  Earliest significant divergence (p < 0.05):")
    for feat in FEATURES:
        earliest = None
        for cp in sorted(results_by_checkpoint.keys()):
            if feat in results_by_checkpoint[cp]:
                if results_by_checkpoint[cp][feat]["p_t"] < 0.05:
                    earliest = cp
                    break
        if earliest:
            d = results_by_checkpoint[earliest][feat]["d"]
            print(f"    {feat:20s}: token {earliest} (d={d:+.3f})")
        else:
            print(f"    {feat:20s}: no significant divergence")

    return results_by_checkpoint


def analyze_delta_trajectory(pairs, model_name):
    """Analyze trajectory as delta from encode baseline (controls for starting point)."""
    print(f"\n{'='*70}")
    print(f"ANALYSIS 3: Delta-from-Encode Trajectory — {model_name}")
    print(f"{'='*70}")

    all_checkpoints = set()
    for p in pairs:
        for cp in p["cognitive"]["trajectory"]:
            all_checkpoints.add(cp["checkpoint"])
    all_checkpoints = sorted(all_checkpoints)

    delta_results = {}

    for cp in all_checkpoints:
        feat_results = {}
        for feat in FEATURES:
            cog_deltas = []
            meta_deltas = []
            for p in pairs:
                enc_cog = p["cognitive"]["encode_features"][feat]
                enc_meta = p["metacognitive"]["encode_features"][feat]
                cog_cp = [t for t in p["cognitive"]["trajectory"] if t["checkpoint"] == cp]
                meta_cp = [t for t in p["metacognitive"]["trajectory"] if t["checkpoint"] == cp]
                if cog_cp and meta_cp:
                    cog_deltas.append(cog_cp[0][feat] - enc_cog)
                    meta_deltas.append(meta_cp[0][feat] - enc_meta)

            if len(cog_deltas) >= 5:
                d = cohens_d(meta_deltas, cog_deltas)
                t, p_val = stats.ttest_rel(meta_deltas, cog_deltas)
                feat_results[feat] = {"d": d, "t": t, "p": p_val, "n": len(cog_deltas)}

        delta_results[cp] = feat_results

    # Print delta trajectory table
    print(f"\n  Delta Cohen's d at each checkpoint (meta_delta - cog_delta):")
    header = f"  {'CP':>5s}"
    for feat in FEATURES:
        header += f" | {feat[:12]:>12s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cp in sorted(delta_results.keys()):
        row = f"  {cp:5d}"
        for feat in FEATURES:
            if feat in delta_results[cp]:
                d = delta_results[cp][feat]["d"]
                sig = "*" if delta_results[cp][feat]["p"] < 0.05 else " "
                row += f" | {d:+11.3f}{sig}"
            else:
                row += f" | {'N/A':>12s}"
        print(row)

    return delta_results


def analyze_growth_rate(pairs, model_name):
    """Compare per-token feature growth rates between conditions."""
    print(f"\n{'='*70}")
    print(f"ANALYSIS 4: Per-Token Growth Rate — {model_name}")
    print(f"{'='*70}")

    # For each pair, compute linear slope of each feature vs gen_tokens
    cog_slopes = {f: [] for f in FEATURES}
    meta_slopes = {f: [] for f in FEATURES}

    for p in pairs:
        for framing, slope_dict in [("cognitive", cog_slopes), ("metacognitive", meta_slopes)]:
            traj = p[framing]["trajectory"]
            if len(traj) < 3:
                continue
            gen_tokens = [t["gen_tokens"] for t in traj]
            for feat in FEATURES:
                vals = [t[feat] for t in traj]
                if len(set(vals)) > 1:  # not all identical
                    slope, _, _, _, _ = stats.linregress(gen_tokens, vals)
                    slope_dict[feat].append(slope)

    print(f"\n  Feature growth rate (slope per gen token):")
    print(f"  {'Feature':20s} | {'Cog slope':>12s} | {'Meta slope':>12s} | {'d':>8s} | {'p':>8s}")
    print(f"  {'-'*68}")

    growth_results = {}
    for feat in FEATURES:
        c = np.array(cog_slopes[feat])
        m = np.array(meta_slopes[feat])
        n = min(len(c), len(m))
        if n >= 5:
            c, m = c[:n], m[:n]
            d = cohens_d(m, c)
            t, p_val = stats.ttest_rel(m, c)
            growth_results[feat] = {"d": d, "t": t, "p": p_val,
                                    "cog_mean": float(np.mean(c)),
                                    "meta_mean": float(np.mean(m))}
            print(f"  {feat:20s} | {np.mean(c):+12.4f} | {np.mean(m):+12.4f} | "
                  f"{d:+8.3f} | {p_val:.4f}")

    return growth_results


def cross_model_comparison(results_by_model):
    """Compare trajectory effects across models."""
    print(f"\n{'='*70}")
    print(f"ANALYSIS 5: Cross-Model Trajectory Consistency")
    print(f"{'='*70}")

    models = list(results_by_model.keys())
    if len(models) < 2:
        print("  Only one model — skipping cross-model comparison")
        return {}

    # For each shared checkpoint, compare d values across models
    shared_checkpoints = set.intersection(
        *[set(results_by_model[m].keys()) for m in models]
    )

    print(f"\n  Models: {', '.join(models)}")
    print(f"  Shared checkpoints: {sorted(shared_checkpoints)}")

    # Collect d values at each checkpoint for correlation
    d_vectors = {m: [] for m in models}
    labels = []

    for cp in sorted(shared_checkpoints):
        for feat in FEATURES:
            all_have = all(feat in results_by_model[m][cp] for m in models)
            if all_have:
                for m in models:
                    d_vectors[m].append(results_by_model[m][cp][feat]["d"])
                labels.append(f"cp{cp}_{feat}")

    # Correlate d-value profiles across models
    if len(labels) >= 5:
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                rho, p = stats.spearmanr(d_vectors[m1], d_vectors[m2])
                print(f"\n  d-profile correlation {m1} vs {m2}:")
                print(f"    Spearman rho = {rho:.3f}, p = {p:.4f}, n = {len(labels)} feature-checkpoints")

                if rho > 0.6:
                    print(f"    CONSISTENT: Both models show similar divergence patterns")
                elif rho > 0.3:
                    print(f"    MODERATE: Partial consistency in divergence patterns")
                else:
                    print(f"    INCONSISTENT: Models diverge differently")

    return d_vectors


def generate_summary(encode_results, traj_results, delta_results, growth_results,
                     model_name, n_pairs):
    """Generate summary JSON for paper integration."""
    summary = {
        "model": model_name,
        "n_pairs": n_pairs,
        "encode_phase": {},
        "trajectory": {},
        "delta_trajectory": {},
        "growth_rates": {},
    }

    # Encode phase
    for feat, r in encode_results.items():
        summary["encode_phase"][feat] = {
            "d": round(r["d"], 4),
            "p": round(r["p"], 6),
        }

    # Trajectory: d at each checkpoint
    for cp in sorted(traj_results.keys()):
        cp_key = f"cp_{cp}"
        summary["trajectory"][cp_key] = {}
        for feat, r in traj_results[cp].items():
            summary["trajectory"][cp_key][feat] = {
                "d": round(r["d"], 4),
                "p_t": round(r["p_t"], 6),
                "p_w": round(r["p_w"], 6),
                "cog_mean": round(r["cog_mean"], 4),
                "meta_mean": round(r["meta_mean"], 4),
                "n": r["n"],
            }

    # Delta trajectory
    for cp in sorted(delta_results.keys()):
        cp_key = f"cp_{cp}"
        summary["delta_trajectory"][cp_key] = {}
        for feat, r in delta_results[cp].items():
            summary["delta_trajectory"][cp_key][feat] = {
                "d": round(r["d"], 4),
                "p": round(r["p"], 6),
            }

    # Growth rates
    for feat, r in growth_results.items():
        summary["growth_rates"][feat] = {
            "d": round(r["d"], 4),
            "p": round(r["p"], 6),
            "cog_slope": round(r["cog_mean"], 6),
            "meta_slope": round(r["meta_mean"], 6),
        }

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/mode_switching")
    parser.add_argument("--models", nargs="+", default=["qwen2.5-7b", "llama-3.1-8b"])
    parser.add_argument("--subdir", default="exp_d_matched")
    parser.add_argument("--output-dir", default="results/mode_switching/analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_traj_results = {}
    all_summaries = {}

    for model in args.models:
        pairs = load_exp_d_results(args.results_dir, model, args.subdir)
        if not pairs:
            print(f"\nNo results for {model} in {args.subdir} — skipping")
            continue

        print(f"\n\n{'#'*70}")
        print(f"# MODEL: {model} ({len(pairs)} pairs)")
        print(f"{'#'*70}")

        # Run all analyses
        encode_results = analyze_length_matching(pairs, model)
        traj_results = analyze_trajectory_divergence(pairs, model)
        delta_results = analyze_delta_trajectory(pairs, model)
        growth_results = analyze_growth_rate(pairs, model)

        all_traj_results[model] = traj_results

        # Generate summary
        summary = generate_summary(encode_results, traj_results, delta_results,
                                   growth_results, model, len(pairs))
        all_summaries[model] = summary

        # Save per-model summary
        out_path = os.path.join(args.output_dir, f"exp_d_matched_{model.replace('.', '_')}.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Saved summary to {out_path}")

    # Cross-model comparison
    if len(all_traj_results) >= 2:
        cross_results = cross_model_comparison(all_traj_results)

        # Save cross-model summary
        cross_path = os.path.join(args.output_dir, "exp_d_matched_cross_model.json")
        cross_summary = {
            "models": list(all_traj_results.keys()),
            "per_model": all_summaries,
        }
        # Add d-value vectors for cross-model correlation
        if cross_results:
            for m, dvec in cross_results.items():
                cross_summary.setdefault("d_vectors", {})[m] = [round(v, 4) for v in dvec]

        with open(cross_path, "w") as f:
            json.dump(cross_summary, f, indent=2)
        print(f"\n  Saved cross-model summary to {cross_path}")

    print("\n\nExperiment D (matched) analysis complete!")


if __name__ == "__main__":
    main()
