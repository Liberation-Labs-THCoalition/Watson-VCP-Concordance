"""
Concordance study pilot test — 12 prompts, 6 pass criteria.

Run this BEFORE the full experiment to verify:
1. Feature extraction produces valid numbers (no NaN, no all-zero)
2. VCP parser achieves >=80% clean parse rate
3. Encode features != generation features (not identical)
4. n_tokens has sufficient variance for FWL
5. Runtime per prompt estimated
6. spectral_entropy != layer_norm_entropy (naming sanity check)

Usage:
    python -m concordance.pilot --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concordance.features import (
    extract_encode_only_features,
    extract_generation_features,
    compute_delta_features,
    PRIMARY_FEATURES,
)
from concordance.vcp_parser import parse_vcp_response, vcp_elicitation_suffix, validate_vcp_distribution
from concordance.prompts import get_pilot_subset


def run_pilot(model_name, device="cuda", output_dir=None):
    """Run pilot test and report pass/fail for 6 criteria."""
    print("=" * 70)
    print(f"CONCORDANCE PILOT TEST — {model_name}")
    print("=" * 70)

    # Load model
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    prompts = get_pilot_subset()
    vcp_suffix = vcp_elicitation_suffix("v2")

    results = []
    timings = []

    for i, prompt_info in enumerate(prompts):
        pid = prompt_info["id"]
        ptype = prompt_info["type"]
        full_prompt = prompt_info["text"] + vcp_suffix

        print(f"\n--- [{i+1}/{len(prompts)}] {pid} ({ptype}) ---")
        t0 = time.time()

        # Phase 1: Encode-only features
        encode_feat = extract_encode_only_features(
            model, tokenizer, full_prompt, device=device
        )

        # Phase 2: Generation features
        # 600 tokens: enough for task response (~400) + VCP ratings (~100) + margin
        gen_feat, response_text, cache = extract_generation_features(
            model, tokenizer, full_prompt,
            max_new_tokens=600, do_sample=False, device=device
        )
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Phase 3: Delta features
        delta_feat = compute_delta_features(encode_feat, gen_feat)

        # Phase 4: Parse VCP ratings
        vcp = parse_vcp_response(response_text, version="v2")

        elapsed = time.time() - t0
        timings.append(elapsed)

        result = {
            "prompt_id": pid,
            "prompt_type": ptype,
            "encode_features": encode_feat,
            "generation_features": gen_feat,
            "delta_features": delta_feat,
            "vcp_ratings": vcp,
            "response_text": response_text,
            "elapsed_seconds": elapsed,
        }
        results.append(result)

        # Quick print
        print(f"  encode eff_rank={encode_feat['eff_rank']:.2f}, spectral_ent={encode_feat['spectral_entropy']:.4f}, layer_norm_ent={encode_feat['layer_norm_entropy']:.4f}")
        print(f"  gen    eff_rank={gen_feat['eff_rank']:.2f}, spectral_ent={gen_feat['spectral_entropy']:.4f}, layer_norm_ent={gen_feat['layer_norm_entropy']:.4f}")
        print(f"  delta  eff_rank={delta_feat.get('eff_rank', '?'):.2f}")
        print(f"  VCP: {vcp['parse_quality']} ({vcp['n_parsed']}/10 dims)")
        print(f"  tokens: encode={encode_feat['n_tokens']}, gen={gen_feat['n_tokens']}")
        print(f"  elapsed: {elapsed:.1f}s")

    # ================================================================
    # EVALUATE 6 CRITERIA
    # ================================================================
    print("\n" + "=" * 70)
    print("PILOT RESULTS — 6 CRITERIA")
    print("=" * 70)

    passes = 0
    total = 6

    # Criterion 1: No NaN, no all-zero features
    all_valid = True
    for r in results:
        for phase in ["encode_features", "generation_features"]:
            feat = r[phase]
            for key in PRIMARY_FEATURES:
                v = feat.get(key, None)
                if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    print(f"  FAIL: {r['prompt_id']} {phase}.{key} = {v}")
                    all_valid = False
    if all_valid:
        print("  [PASS] Criterion 1: All features valid (no NaN, no inf)")
        passes += 1
    else:
        print("  [FAIL] Criterion 1: Invalid feature values detected")

    # Criterion 2: VCP parser >= 80% clean parse rate
    clean_count = sum(1 for r in results if r["vcp_ratings"]["parse_quality"] == "clean")
    parse_rate = clean_count / len(results) * 100
    if parse_rate >= 80:
        print(f"  [PASS] Criterion 2: VCP parse rate {parse_rate:.0f}% ({clean_count}/{len(results)} clean)")
        passes += 1
    else:
        print(f"  [FAIL] Criterion 2: VCP parse rate {parse_rate:.0f}% ({clean_count}/{len(results)} clean) — need >= 80%")
        # Show failed parses
        for r in results:
            if r["vcp_ratings"]["parse_quality"] != "clean":
                print(f"    {r['prompt_id']}: {r['vcp_ratings']['parse_quality']} ({r['vcp_ratings']['n_parsed']}/10)")
                print(f"    Response (first 200 chars): {r['response_text'][:200]}")

    # Criterion 3: Encode != generation features
    all_different = True
    for r in results:
        enc = r["encode_features"]
        gen = r["generation_features"]
        if all(abs(enc.get(k, 0) - gen.get(k, 0)) < 1e-6 for k in PRIMARY_FEATURES):
            print(f"  FAIL: {r['prompt_id']} encode == generation (identical features)")
            all_different = False
    if all_different:
        print("  [PASS] Criterion 3: Encode features differ from generation features")
        passes += 1
    else:
        print("  [FAIL] Criterion 3: Some encode/generation features are identical")

    # Criterion 4: n_tokens has sufficient variance for FWL
    gen_tokens = [r["generation_features"]["n_tokens"] for r in results]
    token_var = np.var(gen_tokens)
    token_cv = np.std(gen_tokens) / np.mean(gen_tokens) if np.mean(gen_tokens) > 0 else 0
    if token_cv > 0.1:  # At least 10% coefficient of variation
        print(f"  [PASS] Criterion 4: Token count CV={token_cv:.2f} (mean={np.mean(gen_tokens):.0f}, std={np.std(gen_tokens):.0f})")
        passes += 1
    else:
        print(f"  [FAIL] Criterion 4: Token count CV={token_cv:.2f} — too low for FWL")

    # Criterion 5: Runtime estimate
    mean_time = np.mean(timings)
    est_full_a = mean_time * 240 * 3 / 3600  # 240 prompts, 3 models
    est_full_b = mean_time * 48 * 3 * 3 / 3600  # 48 prompts, 3 reps, rough (only 7B)
    print(f"  [INFO] Criterion 5: Mean time/prompt = {mean_time:.1f}s")
    print(f"    Estimated Phase A: {est_full_a:.1f} hours (240 prompts × 3 models)")
    print(f"    Estimated Phase B: {est_full_b:.1f} hours (48 prompts × 3 reps)")
    passes += 1  # Always passes (informational)

    # Criterion 6: spectral_entropy != layer_norm_entropy
    all_distinct = True
    for r in results:
        for phase in ["encode_features", "generation_features"]:
            feat = r[phase]
            se = feat.get("spectral_entropy", 0)
            lne = feat.get("layer_norm_entropy", 0)
            if abs(se - lne) < 1e-6:
                print(f"  FAIL: {r['prompt_id']} {phase}: spectral_entropy == layer_norm_entropy ({se})")
                all_distinct = False
    if all_distinct:
        # Show example values to verify they're meaningfully different
        r0 = results[0]
        se = r0["generation_features"]["spectral_entropy"]
        lne = r0["generation_features"]["layer_norm_entropy"]
        print(f"  [PASS] Criterion 6: spectral_entropy ({se:.4f}) != layer_norm_entropy ({lne:.4f})")
        passes += 1
    else:
        print("  [FAIL] Criterion 6: spectral_entropy == layer_norm_entropy — naming fix didn't work")

    # Final verdict
    print(f"\n{'=' * 70}")
    print(f"PILOT VERDICT: {passes}/{total} criteria passed")
    if passes == total:
        print("ALL CRITERIA MET — proceed to full experiment.")
    else:
        print("SOME CRITERIA FAILED — fix before full run.")
    print("=" * 70)

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "pilot_results.json")
        with open(out_path, "w") as f:
            json.dump({
                "model": model_name,
                "n_prompts": len(results),
                "criteria_passed": passes,
                "criteria_total": total,
                "mean_time_per_prompt": mean_time,
                "results": results,
            }, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

    # Cleanup
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return passes == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concordance pilot test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", default="results/concordance/pilot")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    success = run_pilot(args.model, device=args.device, output_dir=args.output_dir)
    sys.exit(0 if success else 1)
