"""
Analyze Experiment B: Controlled Mode-Switching Paradigm.

Tests prediction: adding metacognitive framing ("explain your reasoning process")
to identical content shifts top_sv_ratio with d > 0.5.

20 prompt pairs: same content, cognitive vs metacognitive framing.
"""

import json
import os
import glob
import numpy as np
from scipy import stats


LAYER_FEATURES = ["layer_norm", "eff_rank", "spectral_entropy", "top_sv_ratio", "rank_10", "norm_per_token"]


def load_exp_b_results(results_dir, model="qwen2.5-7b"):
    """Load Exp B paired results."""
    exp_dir = os.path.join(results_dir, model, "exp_b")
    pairs = {}
    for f in sorted(glob.glob(os.path.join(exp_dir, "*.json"))):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        pair_id = data.get("pair_id", os.path.basename(f).split("_")[1])
        framing = data.get("framing", "cognitive" if "cognitive" in os.path.basename(f) else "metacognitive")
        if pair_id not in pairs:
            pairs[pair_id] = {}
        pairs[pair_id][framing] = data
    return pairs


def analyze_paired_shift(pairs):
    """Paired analysis: does metacognitive framing shift geometry?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Paired Framing Effect on Aggregate Features")
    print("=" * 70)

    # Collect aggregate features for each framing
    cog_features = {}
    meta_features = {}

    for feat in ["key_norm", "norm_per_token", "mean_top_sv_ratio", "mean_eff_rank", "layer_variance"]:
        cog_features[feat] = []
        meta_features[feat] = []

    n_valid = 0
    for pair_id, pair in sorted(pairs.items()):
        if "cognitive" not in pair or "metacognitive" not in pair:
            continue
        n_valid += 1

        cog_agg = pair["cognitive"]["generation_features"]["aggregate"]
        meta_agg = pair["metacognitive"]["generation_features"]["aggregate"]

        for feat in cog_features:
            if feat in cog_agg and feat in meta_agg:
                cog_features[feat].append(cog_agg[feat])
                meta_features[feat].append(meta_agg[feat])

    print(f"\n  Valid pairs: {n_valid}")

    results = {}
    for feat in cog_features:
        c = np.array(cog_features[feat])
        m = np.array(meta_features[feat])

        if len(c) < 5:
            continue

        # Paired t-test
        t, p_t = stats.ttest_rel(m, c)
        # Wilcoxon signed-rank
        try:
            w, p_w = stats.wilcoxon(m - c)
        except ValueError:
            w, p_w = 0, 1.0

        # Paired Cohen's d
        diff = m - c
        d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

        results[feat] = {
            "cog_mean": round(float(np.mean(c)), 6),
            "meta_mean": round(float(np.mean(m)), 6),
            "diff_mean": round(float(np.mean(diff)), 6),
            "paired_d": round(float(d), 4),
            "t_stat": round(float(t), 4),
            "p_ttest": float(p_t),
            "p_wilcoxon": float(p_w),
        }

        sig = "*" if p_w < 0.05 else ""
        print(f"  {feat:25s}: cog={np.mean(c):.4f}, meta={np.mean(m):.4f}, "
              f"d={d:+.3f}, p_w={p_w:.4f} {sig}")

    return results


def analyze_per_layer_framing_effect(pairs):
    """Per-layer analysis of framing effect."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Per-Layer Framing Effect")
    print("=" * 70)

    # Determine n_layers from first pair
    first_pair = next(iter(pairs.values()))
    first_data = first_pair.get("cognitive", first_pair.get("metacognitive"))
    n_layers = len(first_data["generation_features"]["per_layer"])

    results = {}

    for feat in ["top_sv_ratio", "eff_rank", "spectral_entropy"]:
        print(f"\n  Feature: {feat}")

        cog_matrix = []
        meta_matrix = []

        for pair_id, pair in sorted(pairs.items()):
            if "cognitive" not in pair or "metacognitive" not in pair:
                continue

            cog_row = []
            meta_row = []
            for layer in range(n_layers):
                cog_row.append(pair["cognitive"]["generation_features"]["per_layer"][str(layer)][feat])
                meta_row.append(pair["metacognitive"]["generation_features"]["per_layer"][str(layer)][feat])
            cog_matrix.append(cog_row)
            meta_matrix.append(meta_row)

        cog_matrix = np.array(cog_matrix)
        meta_matrix = np.array(meta_matrix)

        d_per_layer = []
        p_per_layer = []

        for layer in range(n_layers):
            c = cog_matrix[:, layer]
            m = meta_matrix[:, layer]
            diff = m - c
            d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
            try:
                _, p = stats.wilcoxon(diff)
            except ValueError:
                p = 1.0
            d_per_layer.append(round(float(d), 4))
            p_per_layer.append(float(p))

        peak_layer = int(np.argmax([abs(x) for x in d_per_layer]))
        sig_layers = sum(1 for p in p_per_layer if p < 0.05)

        results[feat] = {
            "d_per_layer": d_per_layer,
            "p_per_layer": p_per_layer,
            "peak_layer": peak_layer,
            "peak_d": d_per_layer[peak_layer],
            "sig_layers": sig_layers,
            "mean_abs_d": round(float(np.mean([abs(x) for x in d_per_layer])), 4),
        }

        print(f"    Peak at layer {peak_layer} (d={d_per_layer[peak_layer]:+.3f}), "
              f"mean |d|={np.mean([abs(x) for x in d_per_layer]):.3f}, "
              f"{sig_layers}/{n_layers} sig")

    return results


def analyze_encode_vs_generation_framing(pairs):
    """Does framing effect appear in encode or only generation?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Framing Effect — Encode vs Generation")
    print("=" * 70)

    first_pair = next(iter(pairs.values()))
    first_data = first_pair.get("cognitive", first_pair.get("metacognitive"))
    n_layers = len(first_data["generation_features"]["per_layer"])

    results = {}

    for feat in ["top_sv_ratio", "eff_rank"]:
        results[feat] = {}

        for phase in ["encode", "generation"]:
            cog_vals = []
            meta_vals = []

            for pair_id, pair in sorted(pairs.items()):
                if "cognitive" not in pair or "metacognitive" not in pair:
                    continue

                # Mean across all layers
                cog_layers = [pair["cognitive"][f"{phase}_features"]["per_layer"][str(l)][feat]
                             for l in range(n_layers)]
                meta_layers = [pair["metacognitive"][f"{phase}_features"]["per_layer"][str(l)][feat]
                              for l in range(n_layers)]

                cog_vals.append(np.mean(cog_layers))
                meta_vals.append(np.mean(meta_layers))

            c = np.array(cog_vals)
            m = np.array(meta_vals)
            diff = m - c
            d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

            try:
                _, p = stats.wilcoxon(diff)
            except ValueError:
                p = 1.0

            results[feat][phase] = {
                "paired_d": round(float(d), 4),
                "p_wilcoxon": float(p),
            }

            sig = "*" if p < 0.05 else ""
            print(f"  {feat:20s} {phase:12s}: d={d:+.3f}, p={p:.4f} {sig}")

    return results


def analyze_token_length_confound(pairs):
    """Check if metacognitive framing simply produces longer responses."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Token Length Confound")
    print("=" * 70)

    cog_tokens = []
    meta_tokens = []

    for pair_id, pair in sorted(pairs.items()):
        if "cognitive" not in pair or "metacognitive" not in pair:
            continue
        cog_tokens.append(pair["cognitive"]["n_tokens"])
        meta_tokens.append(pair["metacognitive"]["n_tokens"])

    c = np.array(cog_tokens)
    m = np.array(meta_tokens)
    diff = m - c
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

    try:
        _, p = stats.wilcoxon(diff)
    except ValueError:
        p = 1.0

    result = {
        "cog_mean_tokens": round(float(np.mean(c)), 1),
        "meta_mean_tokens": round(float(np.mean(m)), 1),
        "diff_mean": round(float(np.mean(diff)), 1),
        "paired_d": round(float(d), 4),
        "p_wilcoxon": float(p),
        "meta_longer_pct": round(float(np.mean(m > c) * 100), 1),
    }

    print(f"  Cognitive mean tokens: {np.mean(c):.1f}")
    print(f"  Metacognitive mean tokens: {np.mean(m):.1f}")
    print(f"  Mean difference: {np.mean(diff):+.1f} tokens")
    print(f"  Paired d: {d:+.3f}")
    print(f"  Meta longer in {np.mean(m > c)*100:.0f}% of pairs")

    if abs(d) > 0.5:
        print("  WARNING: Substantial length difference — geometry effects may be confounded")
    else:
        print("  OK: Length difference is small")

    return result


def analyze_fwl_corrected(pairs):
    """FWL-corrected framing effect: partial out token count."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: FWL-Corrected Framing Effect (token count partialled out)")
    print("=" * 70)

    first_pair = next(iter(pairs.values()))
    first_data = first_pair.get("cognitive", first_pair.get("metacognitive"))
    n_layers = len(first_data["generation_features"]["per_layer"])

    # Collect all data in flat form
    framing_labels = []  # 0=cognitive, 1=metacognitive
    token_counts = []
    feature_vals = {feat: [] for feat in ["top_sv_ratio", "eff_rank", "spectral_entropy"]}

    for pair_id, pair in sorted(pairs.items()):
        if "cognitive" not in pair or "metacognitive" not in pair:
            continue
        for framing in ["cognitive", "metacognitive"]:
            framing_labels.append(0 if framing == "cognitive" else 1)
            token_counts.append(pair[framing]["n_tokens"])
            for feat in feature_vals:
                # Mean across layers for aggregate
                vals = [pair[framing]["generation_features"]["per_layer"][str(l)][feat]
                       for l in range(n_layers)]
                feature_vals[feat].append(np.mean(vals))

    framing_arr = np.array(framing_labels, dtype=float)
    tokens_arr = np.array(token_counts, dtype=float)

    results = {}

    for feat in feature_vals:
        feat_arr = np.array(feature_vals[feat])

        # FWL: regress both framing and feature on tokens, take residuals
        from numpy.linalg import lstsq
        X = np.column_stack([np.ones(len(tokens_arr)), tokens_arr])
        _, res_framing, _, _ = lstsq(X, framing_arr, rcond=None)
        framing_resid = framing_arr - X @ lstsq(X, framing_arr, rcond=None)[0]
        feat_resid = feat_arr - X @ lstsq(X, feat_arr, rcond=None)[0]

        # Point-biserial on residuals
        rho, p = stats.spearmanr(framing_resid, feat_resid)

        # Cohen's d on residuals split by original framing
        cog_resid = feat_resid[framing_arr == 0]
        meta_resid = feat_resid[framing_arr == 1]
        pooled_std = np.sqrt((np.var(cog_resid, ddof=1) * (len(cog_resid)-1) +
                             np.var(meta_resid, ddof=1) * (len(meta_resid)-1)) /
                            (len(cog_resid) + len(meta_resid) - 2))
        d_fwl = (np.mean(meta_resid) - np.mean(cog_resid)) / pooled_std if pooled_std > 0 else 0

        # Raw d for comparison
        cog_raw = feat_arr[framing_arr == 0]
        meta_raw = feat_arr[framing_arr == 1]
        pooled_std_raw = np.sqrt((np.var(cog_raw, ddof=1) * (len(cog_raw)-1) +
                                 np.var(meta_raw, ddof=1) * (len(meta_raw)-1)) /
                                (len(cog_raw) + len(meta_raw) - 2))
        d_raw = (np.mean(meta_raw) - np.mean(cog_raw)) / pooled_std_raw if pooled_std_raw > 0 else 0

        results[feat] = {
            "d_raw": round(float(d_raw), 4),
            "d_fwl": round(float(d_fwl), 4),
            "rho_fwl": round(float(rho), 4),
            "p_fwl": float(p),
            "d_change": round(float(abs(d_fwl) - abs(d_raw)), 4),
            "sign_preserved": bool(np.sign(d_raw) == np.sign(d_fwl)),
        }

        change_pct = ((abs(d_fwl) - abs(d_raw)) / abs(d_raw) * 100) if abs(d_raw) > 0 else 0
        sig = "*" if p < 0.05 else ""
        print(f"  {feat:20s}: d_raw={d_raw:+.3f}, d_fwl={d_fwl:+.3f} "
              f"({change_pct:+.0f}%), rho_fwl={rho:+.3f} (p={p:.4f}) {sig}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir",
                       default="C:/Users/Thomas/Desktop/Watson-VCP-Concordance/results/mode_switching")
    parser.add_argument("--model", default="qwen2.5-7b")
    args = parser.parse_args()

    print("=" * 70)
    print(f"EXPERIMENT B ANALYSIS: Controlled Mode-Switching Paradigm")
    print(f"Model: {args.model}")
    print("=" * 70)

    pairs = load_exp_b_results(args.results_dir, args.model)
    print(f"\nLoaded {len(pairs)} prompt pairs")

    all_analyses = {}

    # Analysis 1: Paired aggregate shift
    all_analyses["paired_shift"] = analyze_paired_shift(pairs)

    # Analysis 2: Per-layer framing effect
    all_analyses["per_layer_framing"] = analyze_per_layer_framing_effect(pairs)

    # Analysis 3: Encode vs generation
    all_analyses["encode_vs_generation"] = analyze_encode_vs_generation_framing(pairs)

    # Analysis 4: Token confound
    all_analyses["token_confound"] = analyze_token_length_confound(pairs)

    # Analysis 5: FWL-corrected
    all_analyses["fwl_corrected"] = analyze_fwl_corrected(pairs)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    ps = all_analyses["paired_shift"]
    sig_features = [f for f, v in ps.items() if v.get("p_wilcoxon", 1) < 0.05]
    print(f"\n  Significant aggregate features: {len(sig_features)}/{len(ps)}")
    for f in sig_features:
        print(f"    {f}: d={ps[f]['paired_d']:+.3f}")

    pl = all_analyses["per_layer_framing"]
    for feat in ["top_sv_ratio", "eff_rank"]:
        if feat in pl:
            print(f"\n  {feat} per-layer: peak at layer {pl[feat]['peak_layer']} "
                  f"(d={pl[feat]['peak_d']:+.3f}), "
                  f"{pl[feat]['sig_layers']}/{len(pl[feat]['d_per_layer'])} sig layers")

    tc = all_analyses["token_confound"]
    if abs(tc["paired_d"]) > 0.5:
        print(f"\n  TOKEN CONFOUND: d={tc['paired_d']:+.3f} — results need FWL correction")
    else:
        print(f"\n  Token confound: d={tc['paired_d']:+.3f} (acceptable)")

    # Save
    out_dir = os.path.join(args.results_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"exp_b_{args.model}_analysis.json")

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
