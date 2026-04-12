"""
Concordance study full experiment runner.

Two-phase design:
  Phase A: temp=0, 1 rep per prompt per model = 720 trials (deterministic geometry)
  Phase B: temp=0.7, 3 reps on 20% subset = 432 trials (VCP reliability ICC)

Usage:
    # Phase A — all prompts, all models
    python -m concordance.experiment --phase A --model Qwen/Qwen2.5-7B-Instruct

    # Phase B — VCP reliability subset
    python -m concordance.experiment --phase B --model Qwen/Qwen2.5-7B-Instruct --n-reps 3

    # Run all models sequentially
    python -m concordance.experiment --phase A --all-models
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concordance.features import (
    extract_encode_only_features,
    extract_generation_features,
    compute_delta_features,
)
from concordance.vcp_parser import (
    parse_vcp_response,
    vcp_elicitation_suffix,
    validate_vcp_distribution,
    extract_vcp_ratings_only,
)
from concordance.prompts import get_all_prompts, get_pilot_subset


MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
]

EXTENDED_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-72B-Instruct",
]

RESULTS_BASE = "results/concordance"


def model_short_name(model_name):
    """Convert 'Qwen/Qwen2.5-7B-Instruct' -> 'qwen2.5-7b'."""
    name = model_name.split("/")[-1].lower()
    name = name.replace("-instruct", "")
    return name


def get_checkpoint_path(output_dir):
    return os.path.join(output_dir, "_checkpoint.json")


def load_checkpoint(output_dir):
    """Load checkpoint to resume from last completed prompt."""
    cp_path = get_checkpoint_path(output_dir)
    if os.path.exists(cp_path):
        with open(cp_path) as f:
            return json.load(f)
    return {"completed_ids": []}


def save_checkpoint(output_dir, completed_ids):
    cp_path = get_checkpoint_path(output_dir)
    with open(cp_path, "w") as f:
        json.dump({"completed_ids": completed_ids}, f)


def run_experiment(model_name, phase="A", n_reps=1, output_dir=None,
                   device="cuda", version="v2", quantize_4bit=False):
    """Run concordance experiment for one model.

    Args:
        model_name: HuggingFace model ID
        phase: "A" (temp=0, 1 rep) or "B" (temp=0.7, n_reps reps, subset)
        n_reps: number of repetitions (only used in phase B)
        output_dir: output directory (auto-generated if None)
        device: "cuda" or "cpu"
        version: "v2" (240 prompts) or "v5" (260 prompts)
        quantize_4bit: load model in 4-bit (for 32B models)
    """
    short_name = model_short_name(model_name)
    phase_lower = phase.lower()

    if output_dir is None:
        output_dir = os.path.join(RESULTS_BASE, short_name, f"phase_{phase_lower}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"CONCORDANCE EXPERIMENT — Phase {phase}")
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load checkpoint
    checkpoint = load_checkpoint(output_dir)
    completed_ids = set(checkpoint["completed_ids"])
    if completed_ids:
        print(f"Resuming from checkpoint: {len(completed_ids)} prompts already done")

    # Get prompts
    if phase == "A":
        prompts = get_all_prompts(version)
        do_sample = False
        temperature = 1.0
        reps = 1
    elif phase == "B":
        # 20% subset for VCP reliability
        all_prompts = get_all_prompts(version)
        # Take every 5th prompt for balanced subset
        prompts = [p for i, p in enumerate(all_prompts) if i % 5 == 0]
        do_sample = True
        temperature = 0.7
        reps = n_reps
    else:
        raise ValueError(f"Unknown phase: {phase}")

    print(f"Prompts: {len(prompts)} × {reps} reps = {len(prompts) * reps} trials")

    # Load model
    print(f"\nLoading {model_name}...")
    load_kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    print("Model loaded.")

    vcp_suffix = vcp_elicitation_suffix(version)
    summary = {
        "model": model_name,
        "phase": phase,
        "version": version,
        "n_prompts": len(prompts),
        "n_reps": reps,
        "temperature": temperature if do_sample else 0.0,
        "do_sample": do_sample,
        "started_at": datetime.now().isoformat(),
        "completed": [],
        "failed": [],
    }

    total_trials = len(prompts) * reps
    trial_num = 0
    timings = []

    for rep in range(reps):
        for prompt_info in prompts:
            pid = prompt_info["id"]
            trial_id = f"{pid}_rep{rep}" if reps > 1 else pid
            trial_num += 1

            if trial_id in completed_ids:
                continue

            full_prompt = prompt_info["text"] + vcp_suffix
            ptype = prompt_info["type"]

            print(f"\n[{trial_num}/{total_trials}] {trial_id} ({ptype})")
            t0 = time.time()

            try:
                # Encode-only features
                encode_feat = extract_encode_only_features(
                    model, tokenizer, full_prompt, device=device
                )

                # Generation features
                # 600 tokens: enough for task response (~400) + VCP ratings (~100) + margin
                gen_feat, response_text, cache = extract_generation_features(
                    model, tokenizer, full_prompt,
                    max_new_tokens=600,
                    do_sample=do_sample,
                    temperature=temperature,
                    device=device,
                )
                del cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Delta features
                delta_feat = compute_delta_features(encode_feat, gen_feat)

                # Parse VCP
                vcp = parse_vcp_response(response_text, version=version)
                vcp_valid = validate_vcp_distribution(vcp, version=version)

                elapsed = time.time() - t0
                timings.append(elapsed)

                # Build result
                result = {
                    "prompt_id": pid,
                    "trial_id": trial_id,
                    "prompt_type": ptype,
                    "prompt_text": prompt_info["text"],
                    "response_text": response_text,
                    "vcp_ratings": extract_vcp_ratings_only(vcp),
                    "vcp_parse_quality": vcp["parse_quality"],
                    "vcp_n_parsed": vcp["n_parsed"],
                    "vcp_validation": vcp_valid,
                    "encode_features": encode_feat,
                    "generation_features": gen_feat,
                    "delta_features": delta_feat,
                    "n_tokens": gen_feat["n_tokens"],
                    "response_length": len(response_text),
                    "model": model_name,
                    "temperature": temperature if do_sample else 0.0,
                    "rep": rep,
                    "elapsed_seconds": elapsed,
                    "timestamp": datetime.now().isoformat(),
                }

                # Save individual result
                out_path = os.path.join(output_dir, f"{trial_id}.json")
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2, default=str)

                summary["completed"].append(trial_id)
                completed_ids.add(trial_id)

                print(f"  eff_rank: enc={encode_feat['eff_rank']:.1f} gen={gen_feat['eff_rank']:.1f}")
                print(f"  VCP: {vcp['parse_quality']} ({vcp['n_parsed']}/10)")
                print(f"  {elapsed:.1f}s")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  FAILED: {e}")
                summary["failed"].append({"trial_id": trial_id, "error": str(e)})

            # Checkpoint every 10 prompts
            if trial_num % 10 == 0:
                save_checkpoint(output_dir, list(completed_ids))
                remaining = total_trials - trial_num
                if timings:
                    eta = remaining * (sum(timings) / len(timings)) / 60
                    print(f"  [Checkpoint] {len(completed_ids)} done, ~{eta:.0f} min remaining")

    # Final summary
    summary["completed_at"] = datetime.now().isoformat()
    summary["n_completed"] = len(summary["completed"])
    summary["n_failed"] = len(summary["failed"])
    if timings:
        summary["mean_time_per_trial"] = sum(timings) / len(timings)
        summary["total_time_minutes"] = sum(timings) / 60

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Clean up checkpoint
    cp_path = get_checkpoint_path(output_dir)
    if os.path.exists(cp_path):
        os.remove(cp_path)

    print(f"\n{'=' * 70}")
    print(f"COMPLETE: {summary['n_completed']}/{total_trials} trials")
    print(f"Failed: {summary['n_failed']}")
    if timings:
        print(f"Total time: {sum(timings)/60:.1f} minutes")
    print(f"Results: {output_dir}")
    print(f"{'=' * 70}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concordance experiment runner")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--phase", choices=["A", "B"], default="A")
    parser.add_argument("--n-reps", type=int, default=3, help="Reps for phase B")
    parser.add_argument("--version", choices=["v2", "v5"], default="v2")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quantize-4bit", action="store_true")
    parser.add_argument("--all-models", action="store_true",
                        help="Run all 3 Qwen models sequentially")
    parser.add_argument("--extended", action="store_true",
                        help="Run extended models (Llama, Mistral, 72B)")
    args = parser.parse_args()

    if args.all_models or args.extended:
        model_list = MODELS if args.all_models else []
        if args.extended:
            model_list = model_list + EXTENDED_MODELS
        for m in model_list:
            q4 = any(s in m for s in ["32B", "70B", "72B"])
            run_experiment(
                m, phase=args.phase, n_reps=args.n_reps,
                device=args.device, version=args.version,
                quantize_4bit=q4,
            )
    else:
        run_experiment(
            args.model, phase=args.phase, n_reps=args.n_reps,
            output_dir=args.output_dir, device=args.device,
            version=args.version, quantize_4bit=args.quantize_4bit,
        )
