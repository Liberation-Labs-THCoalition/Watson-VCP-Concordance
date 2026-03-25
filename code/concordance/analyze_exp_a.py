"""
Analyze Experiment A: Per-Layer Mode-Switching Anatomy.

Tests prediction: metacognitive reorganization peaks at middle (semantic) layers.

Compares per-layer feature profiles between prompt types, focusing on
encode-to-generation sign reversal rates at each layer.
"""

import json
import os
import glob
import numpy as np
from scipy import stats
from collections import defaultdict


PROMPT_TYPES = ["cognitive", "affective", "metacognitive", "mixed"]
LAYER_FEATURES = ["layer_norm", "eff_rank", "spectral_entropy", "top_sv_ratio", "rank_10", "norm_per_token"]


def load_exp_a_results(results_dir, model="qwen2.5-7b"):
    """Load all Exp A result files for a model."""
    exp_dir = os.path.join(results_dir, model, "exp_a")
    results = []
    for f in sorted(glob.glob(os.path.join(exp_dir, "*.json"))):
        with open(f, "r", encoding="utf-8") as fh:
            results.append(json.load(fh))
    return results


def extract_per_layer_profiles(results):
    """Extract per-layer feature arrays grouped by prompt type and phase."""
    n_layers = len(results[0]["encode_features"]["per_layer"])

    profiles = {}
    for ptype in PROMPT_TYPES:
        type_results = [r for r in results if r["prompt_type"] == ptype]
        if not type_results:
            continue

        profiles[ptype] = {
            "encode": {},
            "generation": {},
            "n": len(type_results),
        }

        for phase in ["encode", "generation"]:
            for feat in LAYER_FEATURES:
                # shape: (n_prompts, n_layers)
                matrix = np.zeros((len(type_results), n_layers))
                for i, r in enumerate(type_results):
                    layer_data = r[f"{phase}_features"]["per_layer"]
                    for layer_idx in range(n_layers):
                        matrix[i, layer_idx] = layer_data[str(layer_idx)][feat]
                profiles[ptype][phase][feat] = matrix

    return profiles, n_layers


def analyze_metacognitive_distinctiveness(profiles, n_layers):
    """Cohen's d between metacognitive and other types at each layer."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Metacognitive vs Others — Per-Layer Cohen's d")
    print("=" * 70)

    meta = profiles.get("metacognitive")
    if not meta:
        print("  No metacognitive data!")
        return {}

    # Pool all non-metacognitive
    other_types = [t for t in PROMPT_TYPES if t != "metacognitive" and t in profiles]

    results = {}

    for feat in ["top_sv_ratio", "eff_rank", "spectral_entropy"]:
        print(f"\n  Feature: {feat}")

        for phase in ["encode", "generation"]:
            meta_matrix = meta[phase][feat]  # (12, n_layers)

            # Pool others
            other_matrices = [profiles[t][phase][feat] for t in other_types]
            other_matrix = np.vstack(other_matrices)  # (36, n_layers)

            d_per_layer = []
            p_per_layer = []
            for layer in range(n_layers):
                m = meta_matrix[:, layer]
                o = other_matrix[:, layer]
                pooled_std = np.sqrt((np.var(m, ddof=1) * (len(m)-1) + np.var(o, ddof=1) * (len(o)-1)) / (len(m) + len(o) - 2))
                d = (np.mean(m) - np.mean(o)) / pooled_std if pooled_std > 0 else 0
                _, p = stats.mannwhitneyu(m, o, alternative='two-sided')
                d_per_layer.append(d)
                p_per_layer.append(p)

            # Find peak
            abs_d = [abs(x) for x in d_per_layer]
            peak_layer = np.argmax(abs_d)
            peak_d = d_per_layer[peak_layer]

            key = f"{feat}_{phase}"
            results[key] = {
                "d_per_layer": [round(x, 4) for x in d_per_layer],
                "p_per_layer": [round(x, 6) for x in p_per_layer],
                "peak_layer": int(peak_layer),
                "peak_d": round(peak_d, 4),
                "mean_d": round(np.mean(abs_d), 4),
                "sig_layers": sum(1 for p in p_per_layer if p < 0.05),
            }

            # Print compact summary
            sig_str = f"{results[key]['sig_layers']}/{n_layers} sig"
            print(f"    {phase:12s}: peak at layer {peak_layer:2d} (d={peak_d:+.3f}), "
                  f"mean |d|={np.mean(abs_d):.3f}, {sig_str}")

    return results


def analyze_encode_generation_reversal_per_layer(profiles, n_layers):
    """Sign reversal rate between encode and generation at each layer."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Encode-Generation Reversal Per Layer")
    print("=" * 70)

    results = {}

    for ptype in PROMPT_TYPES:
        if ptype not in profiles:
            continue

        print(f"\n  Prompt type: {ptype}")
        type_results = {}

        for feat in ["top_sv_ratio", "eff_rank"]:
            enc_matrix = profiles[ptype]["encode"][feat]    # (n, n_layers)
            gen_matrix = profiles[ptype]["generation"][feat]  # (n, n_layers)

            reversal_per_layer = []
            for layer in range(n_layers):
                enc_vals = enc_matrix[:, layer]
                gen_vals = gen_matrix[:, layer]

                # Correlation between encode and generation values across prompts
                if np.std(enc_vals) > 0 and np.std(gen_vals) > 0:
                    rho_enc, _ = stats.spearmanr(enc_vals, gen_vals)
                else:
                    rho_enc = 0

                # "Reversal" = negative correlation (encode high -> generation low)
                reversal_per_layer.append(round(float(rho_enc), 4))

            # Find layer of maximum reversal (most negative rho)
            min_layer = np.argmin(reversal_per_layer)

            type_results[feat] = {
                "rho_per_layer": reversal_per_layer,
                "min_layer": int(min_layer),
                "min_rho": round(float(reversal_per_layer[min_layer]), 4),
                "mean_rho": round(float(np.mean(reversal_per_layer)), 4),
                "neg_layers": sum(1 for r in reversal_per_layer if r < 0),
            }

            print(f"    {feat:20s}: max reversal at layer {min_layer:2d} "
                  f"(rho={reversal_per_layer[min_layer]:+.3f}), "
                  f"mean rho={np.mean(reversal_per_layer):+.3f}, "
                  f"{type_results[feat]['neg_layers']}/{n_layers} negative")

        results[ptype] = type_results

    return results


def analyze_layer_profile_shape(profiles, n_layers):
    """Compare the SHAPE of per-layer feature profiles between types."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Layer Profile Shape — Encode vs Generation")
    print("=" * 70)

    results = {}

    for feat in ["top_sv_ratio", "eff_rank", "spectral_entropy"]:
        print(f"\n  Feature: {feat}")
        results[feat] = {}

        for ptype in PROMPT_TYPES:
            if ptype not in profiles:
                continue

            # Mean profile across prompts: (n_layers,)
            enc_profile = profiles[ptype]["encode"][feat].mean(axis=0)
            gen_profile = profiles[ptype]["generation"][feat].mean(axis=0)

            # Correlation between encode and generation profiles
            rho, p = stats.spearmanr(enc_profile, gen_profile)

            # Difference profile
            diff = gen_profile - enc_profile

            results[feat][ptype] = {
                "enc_profile": [round(x, 4) for x in enc_profile],
                "gen_profile": [round(x, 4) for x in gen_profile],
                "diff_profile": [round(x, 4) for x in diff],
                "profile_rho": round(float(rho), 4),
                "profile_p": float(p),
                "max_diff_layer": int(np.argmax(np.abs(diff))),
                "max_diff_value": round(float(diff[np.argmax(np.abs(diff))]), 4),
            }

            print(f"    {ptype:15s}: enc-gen profile rho={rho:+.3f} (p={p:.4f}), "
                  f"max diff at layer {np.argmax(np.abs(diff)):2d} ({diff[np.argmax(np.abs(diff))]:+.4f})")

    return results


def analyze_middle_layer_hypothesis(profiles, n_layers):
    """Test prediction: metacognitive reorganization peaks at layers 8-14 of 28."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Middle Layer Hypothesis (layers 8-14)")
    print("=" * 70)

    meta = profiles.get("metacognitive")
    if not meta:
        return {}

    other_types = [t for t in PROMPT_TYPES if t != "metacognitive" and t in profiles]

    # Define layer regions
    early = list(range(0, 8))       # Layers 0-7
    middle = list(range(8, 15))     # Layers 8-14
    late = list(range(15, n_layers)) # Layers 15-27

    results = {}

    for feat in ["top_sv_ratio", "eff_rank", "spectral_entropy"]:
        meta_enc = meta["encode"][feat]      # (12, 28)
        meta_gen = meta["generation"][feat]   # (12, 28)

        other_enc = np.vstack([profiles[t]["encode"][feat] for t in other_types])
        other_gen = np.vstack([profiles[t]["generation"][feat] for t in other_types])

        # Compute encode-to-generation CHANGE for meta vs others at each layer
        meta_delta = meta_gen - meta_enc    # (12, 28)
        other_delta = other_gen - other_enc  # (36, 28)

        # Mean |d| for each region
        region_d = {}
        for region_name, region_layers in [("early", early), ("middle", middle), ("late", late)]:
            d_vals = []
            for layer in region_layers:
                m = meta_delta[:, layer]
                o = other_delta[:, layer]
                pooled_std = np.sqrt((np.var(m, ddof=1) * (len(m)-1) + np.var(o, ddof=1) * (len(o)-1)) / (len(m) + len(o) - 2))
                d = (np.mean(m) - np.mean(o)) / pooled_std if pooled_std > 0 else 0
                d_vals.append(abs(d))
            region_d[region_name] = round(float(np.mean(d_vals)), 4)

        winner = max(region_d, key=region_d.get)
        hypothesis_supported = (winner == "middle")

        results[feat] = {
            "early_mean_d": region_d["early"],
            "middle_mean_d": region_d["middle"],
            "late_mean_d": region_d["late"],
            "winner": winner,
            "hypothesis_supported": hypothesis_supported,
        }

        marker = "CONFIRMED" if hypothesis_supported else f"REJECTED (winner={winner})"
        print(f"  {feat:20s}: early={region_d['early']:.3f}, "
              f"middle={region_d['middle']:.3f}, late={region_d['late']:.3f} "
              f"-> {marker}")

    return results


def analyze_generation_delta_confound(profiles, n_layers):
    """Check if token count confound explains per-layer differences."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Token Count Confound Check")
    print("=" * 70)

    # Get token counts per type
    results = {}

    for ptype in PROMPT_TYPES:
        if ptype not in profiles:
            continue

        gen_norms = profiles[ptype]["generation"]["norm_per_token"]  # (n, 28)
        enc_norms = profiles[ptype]["encode"]["norm_per_token"]      # (n, 28)

        # Mean norm_per_token across all layers
        mean_gen = gen_norms.mean()
        mean_enc = enc_norms.mean()

        results[ptype] = {
            "mean_gen_norm_per_token": round(float(mean_gen), 4),
            "mean_enc_norm_per_token": round(float(mean_enc), 4),
            "ratio": round(float(mean_gen / mean_enc) if mean_enc > 0 else 0, 4),
        }

        print(f"  {ptype:15s}: enc norm/tok={mean_enc:.3f}, gen norm/tok={mean_gen:.3f}, "
              f"ratio={mean_gen/mean_enc:.3f}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir",
                       default="C:/Users/Thomas/Desktop/Watson-VCP-Concordance/results/mode_switching")
    parser.add_argument("--model", default="qwen2.5-7b")
    args = parser.parse_args()

    print("=" * 70)
    print(f"EXPERIMENT A ANALYSIS: Per-Layer Mode-Switching Anatomy")
    print(f"Model: {args.model}")
    print("=" * 70)

    results = load_exp_a_results(args.results_dir, args.model)
    print(f"\nLoaded {len(results)} result files")

    # Count by type
    type_counts = defaultdict(int)
    for r in results:
        type_counts[r["prompt_type"]] += 1
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    profiles, n_layers = extract_per_layer_profiles(results)
    print(f"  Layers: {n_layers}")

    all_analyses = {}

    # Analysis 1: Metacognitive distinctiveness per layer
    all_analyses["metacognitive_d"] = analyze_metacognitive_distinctiveness(profiles, n_layers)

    # Analysis 2: Encode-generation reversal per layer
    all_analyses["reversal_per_layer"] = analyze_encode_generation_reversal_per_layer(profiles, n_layers)

    # Analysis 3: Layer profile shape
    all_analyses["profile_shape"] = analyze_layer_profile_shape(profiles, n_layers)

    # Analysis 4: Middle layer hypothesis
    all_analyses["middle_layer"] = analyze_middle_layer_hypothesis(profiles, n_layers)

    # Analysis 5: Token confound
    all_analyses["token_confound"] = analyze_generation_delta_confound(profiles, n_layers)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Middle layer verdict
    ml = all_analyses["middle_layer"]
    supported = sum(1 for f in ml.values() if f.get("hypothesis_supported", False))
    total = len(ml)
    print(f"\n  Middle-layer hypothesis: {supported}/{total} features support")

    # Peak layers for metacognitive d
    md = all_analyses["metacognitive_d"]
    for key, val in md.items():
        if "generation" in key:
            print(f"  {key}: peak d at layer {val['peak_layer']} (d={val['peak_d']:+.3f})")

    # Save
    out_dir = os.path.join(args.results_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"exp_a_{args.model}_analysis.json")

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
        json.dump(all_analyses, f, indent=2, default=json_default, ensure_ascii=False)

    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
