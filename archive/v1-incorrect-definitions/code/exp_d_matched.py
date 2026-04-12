"""
Experiment D (v2): Length-Matched Generation Trajectory.

Fixes the input-length confound from v1: cognitive prompts are padded to
match metacognitive prompt token count, ensuring identical input lengths.

Usage:
    python exp_d_matched.py --model qwen
    python exp_d_matched.py --model llama
    python exp_d_matched.py --model all
"""

import argparse
import gc
import json
import os
import time
import numpy as np
import torch
from datetime import datetime

CHECKPOINTS = [10, 25, 50, 75, 100, 150, 200]

# Padding templates of increasing length. We pick the one that gets closest
# to the target, then fine-tune with trailing tokens.
PADDING_PREFIXES = [
    "Consider this carefully.",                              # ~6 tokens
    "Think about this question carefully.",                   # ~8 tokens
    "Please think about the following question carefully.",   # ~10 tokens
    "Please think carefully about the following question and provide your answer.",  # ~14 tokens
    "Take your time and think carefully about the following question. Please provide a complete answer.",  # ~18 tokens
    "Please take your time with this. Think carefully about the following question and provide a thorough, complete answer.",  # ~22 tokens
    "Please take your time with this question. Think carefully and methodically about what is being asked, and provide a thorough, complete answer.",  # ~28 tokens
]

# Original prompt pairs (same as exp_d_trajectory.py)
EXP_D_PAIRS = [
    {"id": 1, "content": "Calculate the area of a circle with radius 7.",
     "cognitive": "Calculate the area of a circle with radius 7.",
     "metacognitive": "Calculate the area of a circle with radius 7. As you work through this, describe your reasoning process at each step."},
    {"id": 2, "content": "Why do leaves change color in autumn?",
     "cognitive": "Why do leaves change color in autumn?",
     "metacognitive": "Why do leaves change color in autumn? As you explain, reflect on how confident you feel about each part of your answer."},
    {"id": 3, "content": "Compare democracy and authoritarianism as systems of governance.",
     "cognitive": "Compare democracy and authoritarianism as systems of governance.",
     "metacognitive": "Compare democracy and authoritarianism as systems of governance. Track how you balance different perspectives as you reason through this."},
    {"id": 4, "content": "Explain how a binary search algorithm works.",
     "cognitive": "Explain how a binary search algorithm works.",
     "metacognitive": "Explain how a binary search algorithm works. Describe what's happening in your reasoning as you formulate each part of the explanation."},
    {"id": 5, "content": "What causes ocean tides?",
     "cognitive": "What causes ocean tides?",
     "metacognitive": "What causes ocean tides? Rate your confidence (1-10) in each claim you make as you explain."},
    {"id": 6, "content": "Solve: If 3x + 7 = 22, what is x?",
     "cognitive": "Solve: If 3x + 7 = 22, what is x?",
     "metacognitive": "Solve: If 3x + 7 = 22, what is x? Walk me through your internal reasoning process as you solve this step by step."},
    {"id": 7, "content": "Explain the concept of supply and demand in economics.",
     "cognitive": "Explain the concept of supply and demand in economics.",
     "metacognitive": "Explain the concept of supply and demand in economics. As you explain, reflect on which parts you find easiest and hardest to articulate."},
    {"id": 8, "content": "What is photosynthesis and why is it important?",
     "cognitive": "What is photosynthesis and why is it important?",
     "metacognitive": "What is photosynthesis and why is it important? Monitor your own explanation process — note where you feel most and least certain."},
    {"id": 9, "content": "Describe the water cycle.",
     "cognitive": "Describe the water cycle.",
     "metacognitive": "Describe the water cycle. As you describe it, reflect on how you're organizing this information and what choices you're making about emphasis."},
    {"id": 10, "content": "What is the significance of the Pythagorean theorem?",
     "cognitive": "What is the significance of the Pythagorean theorem?",
     "metacognitive": "What is the significance of the Pythagorean theorem? Describe how your understanding unfolds as you think through this question."},
    {"id": 11, "content": "Explain how vaccines work.",
     "cognitive": "Explain how vaccines work.",
     "metacognitive": "Explain how vaccines work. As you explain, track which parts of your explanation you're most confident about and which you find harder to articulate precisely."},
    {"id": 12, "content": "What causes earthquakes?",
     "cognitive": "What causes earthquakes?",
     "metacognitive": "What causes earthquakes? Reflect on your reasoning process — how are you deciding what information to include and in what order?"},
    {"id": 13, "content": "Explain the difference between weather and climate.",
     "cognitive": "Explain the difference between weather and climate.",
     "metacognitive": "Explain the difference between weather and climate. As you formulate your response, describe how you're structuring your thinking about this distinction."},
    {"id": 14, "content": "How does natural selection drive evolution?",
     "cognitive": "How does natural selection drive evolution?",
     "metacognitive": "How does natural selection drive evolution? Walk through your reasoning process, noting where you're drawing on different types of knowledge."},
    {"id": 15, "content": "Explain the greenhouse effect.",
     "cognitive": "Explain the greenhouse effect.",
     "metacognitive": "Explain the greenhouse effect. As you explain, reflect on how you're balancing simplicity with accuracy in your explanation."},
    {"id": 16, "content": "What is the difference between an atom and a molecule?",
     "cognitive": "What is the difference between an atom and a molecule?",
     "metacognitive": "What is the difference between an atom and a molecule? Describe your internal process as you formulate this explanation — what analogies or frameworks are you considering?"},
    {"id": 17, "content": "Explain how compound interest works.",
     "cognitive": "Explain how compound interest works.",
     "metacognitive": "Explain how compound interest works. Track your reasoning — how are you deciding whether to use examples, formulas, or intuitive explanations?"},
    {"id": 18, "content": "What is the role of DNA in heredity?",
     "cognitive": "What is the role of DNA in heredity?",
     "metacognitive": "What is the role of DNA in heredity? As you explain, monitor which aspects of this topic you find clearest and which require more careful thought."},
    {"id": 19, "content": "Explain Newton's three laws of motion.",
     "cognitive": "Explain Newton's three laws of motion.",
     "metacognitive": "Explain Newton's three laws of motion. Reflect on how you're organizing your explanation — what choices are you making about order, emphasis, and examples?"},
    {"id": 20, "content": "What causes inflation in an economy?",
     "cognitive": "What causes inflation in an economy?",
     "metacognitive": "What causes inflation in an economy? As you explain, describe how confident you feel about each causal mechanism you identify."},
]


def match_prompt_lengths(pairs, tokenizer):
    """Pad cognitive prompts to match metacognitive token count.

    Strategy: prepend neutral preamble to cognitive prompt, choosing the
    preamble length that gets closest to the metacognitive token count.
    If still short, append period-separated filler words.
    """
    matched = []
    for pair in pairs:
        meta_tokens = len(tokenizer.encode(pair["metacognitive"]))

        best_diff = float("inf")
        best_padded = pair["cognitive"]

        for prefix in PADDING_PREFIXES:
            padded = f"{prefix} {pair['cognitive']}"
            padded_tokens = len(tokenizer.encode(padded))
            diff = abs(padded_tokens - meta_tokens)
            if diff < best_diff:
                best_diff = diff
                best_padded = padded

        # Fine-tune: if still short, add filler words one at a time
        fillers = ["Indeed,", "Now,", "So,", "Well,", "Right,", "OK,",
                   "Sure,", "Yes,", "Here,", "First,"]
        filler_idx = 0
        current = best_padded
        while len(tokenizer.encode(current)) < meta_tokens and filler_idx < len(fillers):
            current = current.replace(". ", f". {fillers[filler_idx]} ", 1)
            if len(tokenizer.encode(current)) > meta_tokens:
                current = best_padded  # revert if overshoot
                break
            best_padded = current
            filler_idx += 1

        cog_final_tokens = len(tokenizer.encode(best_padded))

        matched.append({
            "id": pair["id"],
            "content": pair["content"],
            "cognitive_original": pair["cognitive"],
            "cognitive_padded": best_padded,
            "metacognitive": pair["metacognitive"],
            "cog_tokens": cog_final_tokens,
            "meta_tokens": meta_tokens,
            "token_diff": cog_final_tokens - meta_tokens,
        })

    return matched


def extract_features_from_cache(cache, n_tokens):
    """Extract aggregate features from a KV cache state."""
    n_layers = len(cache)
    all_norms = []
    all_tsv = []
    all_er = []
    all_se = []

    for layer_idx in range(n_layers):
        key_states = cache[layer_idx][0]
        k = key_states[0]
        k_flat = k.permute(1, 0, 2).reshape(k.shape[1], -1).float()

        layer_norm = float(torch.norm(k_flat, p='fro').item())
        all_norms.append(layer_norm)

        try:
            U, S, Vh = torch.linalg.svd(k_flat, full_matrices=False)
            S_np = S.cpu().numpy()
            S_pos = S_np[S_np > 1e-10]
            S_norm = S_pos / S_pos.sum()
            spectral_entropy = float(-np.sum(S_norm * np.log(S_norm + 1e-15)))
            eff_rank = float(np.exp(spectral_entropy))
            top_sv_ratio = float(S_np[0] / S_np.sum()) if S_np.sum() > 0 else 0
        except Exception:
            spectral_entropy = 0
            eff_rank = 0
            top_sv_ratio = 0

        all_tsv.append(top_sv_ratio)
        all_er.append(eff_rank)
        all_se.append(spectral_entropy)

    norms = np.array(all_norms)
    total_norm = float(np.sqrt(np.sum(norms ** 2)))

    return {
        "key_norm": round(total_norm, 4),
        "norm_per_token": round(total_norm / max(n_tokens, 1), 4),
        "eff_rank": round(float(np.mean(all_er)), 4),
        "top_sv_ratio": round(float(np.mean(all_tsv)), 6),
        "spectral_entropy": round(float(np.mean(all_se)), 4),
        "layer_variance": round(float(np.var(norms)), 4),
        "n_tokens": n_tokens,
    }


def run_trajectory(model, tokenizer, prompt, checkpoints=CHECKPOINTS, max_tokens=200):
    """Generate with checkpointed feature extraction."""
    device = next(model.parameters()).device

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Encode only
    with torch.no_grad():
        enc_out = model(**inputs, use_cache=True)
        enc_cache = enc_out.past_key_values
    encode_features = extract_features_from_cache(enc_cache, input_len)

    # Full generation
    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
            return_dict_in_generate=True,
            use_cache=True,
        )

    output_ids = gen_out.sequences[0]
    total_gen_tokens = len(output_ids) - input_len
    response_text = tokenizer.decode(output_ids[input_len:], skip_special_tokens=True)

    # Checkpointed generation
    trajectory = []
    for cp in checkpoints:
        if cp > total_gen_tokens:
            break
        with torch.no_grad():
            cp_out = model.generate(
                **inputs,
                max_new_tokens=cp,
                do_sample=False,
                temperature=1.0,
                return_dict_in_generate=True,
                use_cache=True,
            )
        cp_cache = cp_out.past_key_values
        cp_n_tokens = input_len + cp
        cp_features = extract_features_from_cache(cp_cache, cp_n_tokens)
        cp_features["gen_tokens"] = cp
        cp_features["checkpoint"] = cp
        trajectory.append(cp_features)
        del cp_out, cp_cache
        torch.cuda.empty_cache()

    # Final
    final_features = extract_features_from_cache(gen_out.past_key_values, len(output_ids))
    final_features["gen_tokens"] = total_gen_tokens
    final_features["checkpoint"] = total_gen_tokens
    trajectory.append(final_features)

    del gen_out, enc_out
    torch.cuda.empty_cache()

    return {
        "encode_features": encode_features,
        "trajectory": trajectory,
        "total_gen_tokens": total_gen_tokens,
        "input_len": input_len,
        "response_text": response_text,
    }


MODELS = {
    "qwen": {"name": "qwen2.5-7b", "hf_id": "Qwen/Qwen2.5-7B-Instruct"},
    "llama": {"name": "llama-3.1-8b", "hf_id": "meta-llama/Llama-3.1-8B-Instruct"},
}


def load_model(model_key):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    info = MODELS[model_key]
    print(f"\nLoading {info['hf_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(info["hf_id"])
    model = AutoModelForCausalLM.from_pretrained(
        info["hf_id"], dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, info["name"]


def run_experiment_d_matched(model, tokenizer, model_name, output_dir):
    """Run Exp D with length-matched prompts."""
    exp_dir = os.path.join(output_dir, model_name, "exp_d_matched")
    os.makedirs(exp_dir, exist_ok=True)

    # Match lengths
    matched = match_prompt_lengths(EXP_D_PAIRS, tokenizer)

    print(f"\n{'='*60}")
    print(f"Experiment D (matched): Generation Trajectory — {model_name}")
    print(f"  {len(matched)} prompt pairs, length-matched")
    print(f"{'='*60}")

    # Report matching quality
    diffs = [m["token_diff"] for m in matched]
    print(f"  Token diffs after matching: mean={np.mean(diffs):.1f}, "
          f"max={max(abs(d) for d in diffs)}, range=[{min(diffs)},{max(diffs)}]")

    for idx, pair in enumerate(matched):
        out_path = os.path.join(exp_dir, f"pair_{pair['id']:03d}.json")
        if os.path.exists(out_path):
            print(f"  [{idx+1}/{len(matched)}] Pair {pair['id']} — already done, skipping")
            continue

        print(f"  [{idx+1}/{len(matched)}] Pair {pair['id']}: {pair['content'][:50]}...")
        result = {
            "pair_id": pair["id"],
            "content": pair["content"],
            "cog_tokens_input": pair["cog_tokens"],
            "meta_tokens_input": pair["meta_tokens"],
            "token_diff": pair["token_diff"],
        }

        for framing, prompt in [("cognitive", pair["cognitive_padded"]),
                                 ("metacognitive", pair["metacognitive"])]:
            t0 = time.time()
            try:
                traj = run_trajectory(model, tokenizer, prompt)
                elapsed = time.time() - t0
                result[framing] = {
                    "prompt": prompt,
                    "encode_features": traj["encode_features"],
                    "trajectory": traj["trajectory"],
                    "total_gen_tokens": traj["total_gen_tokens"],
                    "input_len": traj["input_len"],
                    "response_text": traj["response_text"],
                    "elapsed_s": round(elapsed, 2),
                }
                print(f"    {framing}: {traj['total_gen_tokens']} gen tokens, "
                      f"input={traj['input_len']}, {len(traj['trajectory'])} checkpoints, {elapsed:.1f}s")
            except Exception as e:
                print(f"    {framing}: ERROR — {e}")
                result[framing] = {"error": str(e)}
            gc.collect()
            torch.cuda.empty_cache()

        result["timestamp"] = datetime.now().isoformat()
        result["model"] = model_name

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nExperiment D (matched) complete for {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen", "llama", "all"], default="qwen")
    parser.add_argument("--output-dir", default="results/mode_switching")
    args = parser.parse_args()

    models_to_run = ["qwen", "llama"] if args.model == "all" else [args.model]

    for model_key in models_to_run:
        model, tokenizer, model_name = load_model(model_key)
        run_experiment_d_matched(model, tokenizer, model_name, args.output_dir)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAll Experiment D (matched) runs complete!")


if __name__ == "__main__":
    main()
