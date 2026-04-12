"""
Experiment D: Generation Trajectory — Mode-Switch Temporal Dynamics.

Question: Does the mode switch happen immediately or gradually during generation?

Method: 20 prompt pairs (same content, cognitive vs metacognitive framing),
extract KV-cache features at multiple generation checkpoints (tokens 10, 25,
50, 75, 100, 150, 200). Track how geometry diverges between conditions.

Prediction: Metacognitive prompts diverge from cognitive early (first 25
tokens), stabilizing by token 50.

Usage:
    python exp_d_trajectory.py --model qwen
    python exp_d_trajectory.py --model llama
    python exp_d_trajectory.py --model all
"""

import argparse
import gc
import json
import os
import time
import numpy as np
import torch
from datetime import datetime


# ================================================================
# CHECKPOINT SCHEDULE
# ================================================================

# Extract features at these generation token counts
CHECKPOINTS = [10, 25, 50, 75, 100, 150, 200]


# ================================================================
# PROMPT PAIRS (same as Exp B — identical content, two framings)
# ================================================================

EXP_D_PAIRS = [
    {
        "id": 1,
        "content": "Calculate the area of a circle with radius 7.",
        "cognitive": "Calculate the area of a circle with radius 7.",
        "metacognitive": "Calculate the area of a circle with radius 7. As you work through this, describe your reasoning process at each step.",
    },
    {
        "id": 2,
        "content": "Why do leaves change color in autumn?",
        "cognitive": "Why do leaves change color in autumn?",
        "metacognitive": "Why do leaves change color in autumn? As you explain, reflect on how confident you feel about each part of your answer.",
    },
    {
        "id": 3,
        "content": "Compare democracy and authoritarianism as systems of governance.",
        "cognitive": "Compare democracy and authoritarianism as systems of governance.",
        "metacognitive": "Compare democracy and authoritarianism as systems of governance. Track how you balance different perspectives as you reason through this.",
    },
    {
        "id": 4,
        "content": "Explain how a binary search algorithm works.",
        "cognitive": "Explain how a binary search algorithm works.",
        "metacognitive": "Explain how a binary search algorithm works. Describe what's happening in your reasoning as you formulate each part of the explanation.",
    },
    {
        "id": 5,
        "content": "What causes ocean tides?",
        "cognitive": "What causes ocean tides?",
        "metacognitive": "What causes ocean tides? Rate your confidence (1-10) in each claim you make as you explain.",
    },
    {
        "id": 6,
        "content": "Solve: If 3x + 7 = 22, what is x?",
        "cognitive": "Solve: If 3x + 7 = 22, what is x?",
        "metacognitive": "Solve: If 3x + 7 = 22, what is x? Walk me through your internal reasoning process as you solve this step by step.",
    },
    {
        "id": 7,
        "content": "Explain the concept of supply and demand in economics.",
        "cognitive": "Explain the concept of supply and demand in economics.",
        "metacognitive": "Explain the concept of supply and demand in economics. As you explain, reflect on which parts you find easiest and hardest to articulate.",
    },
    {
        "id": 8,
        "content": "What is photosynthesis and why is it important?",
        "cognitive": "What is photosynthesis and why is it important?",
        "metacognitive": "What is photosynthesis and why is it important? Monitor your own explanation process — note where you feel most and least certain.",
    },
    {
        "id": 9,
        "content": "Describe the water cycle.",
        "cognitive": "Describe the water cycle.",
        "metacognitive": "Describe the water cycle. As you describe it, reflect on how you're organizing this information and what choices you're making about emphasis.",
    },
    {
        "id": 10,
        "content": "What is the significance of the Pythagorean theorem?",
        "cognitive": "What is the significance of the Pythagorean theorem?",
        "metacognitive": "What is the significance of the Pythagorean theorem? Describe how your understanding unfolds as you think through this question.",
    },
    {
        "id": 11,
        "content": "Explain how vaccines work.",
        "cognitive": "Explain how vaccines work.",
        "metacognitive": "Explain how vaccines work. As you explain, track which parts of your explanation you're most confident about and which you find harder to articulate precisely.",
    },
    {
        "id": 12,
        "content": "What causes earthquakes?",
        "cognitive": "What causes earthquakes?",
        "metacognitive": "What causes earthquakes? Reflect on your reasoning process — how are you deciding what information to include and in what order?",
    },
    {
        "id": 13,
        "content": "Explain the difference between weather and climate.",
        "cognitive": "Explain the difference between weather and climate.",
        "metacognitive": "Explain the difference between weather and climate. As you formulate your response, describe how you're structuring your thinking about this distinction.",
    },
    {
        "id": 14,
        "content": "How does natural selection drive evolution?",
        "cognitive": "How does natural selection drive evolution?",
        "metacognitive": "How does natural selection drive evolution? Walk through your reasoning process, noting where you're drawing on different types of knowledge.",
    },
    {
        "id": 15,
        "content": "Explain the greenhouse effect.",
        "cognitive": "Explain the greenhouse effect.",
        "metacognitive": "Explain the greenhouse effect. As you explain, reflect on how you're balancing simplicity with accuracy in your explanation.",
    },
    {
        "id": 16,
        "content": "What is the difference between an atom and a molecule?",
        "cognitive": "What is the difference between an atom and a molecule?",
        "metacognitive": "What is the difference between an atom and a molecule? Describe your internal process as you formulate this explanation — what analogies or frameworks are you considering?",
    },
    {
        "id": 17,
        "content": "Explain how compound interest works.",
        "cognitive": "Explain how compound interest works.",
        "metacognitive": "Explain how compound interest works. Track your reasoning — how are you deciding whether to use examples, formulas, or intuitive explanations?",
    },
    {
        "id": 18,
        "content": "What is the role of DNA in heredity?",
        "cognitive": "What is the role of DNA in heredity?",
        "metacognitive": "What is the role of DNA in heredity? As you explain, monitor which aspects of this topic you find clearest and which require more careful thought.",
    },
    {
        "id": 19,
        "content": "Explain Newton's three laws of motion.",
        "cognitive": "Explain Newton's three laws of motion.",
        "metacognitive": "Explain Newton's three laws of motion. Reflect on how you're organizing your explanation — what choices are you making about order, emphasis, and examples?",
    },
    {
        "id": 20,
        "content": "What causes inflation in an economy?",
        "cognitive": "What causes inflation in an economy?",
        "metacognitive": "What causes inflation in an economy? As you explain, describe how confident you feel about each causal mechanism you identify.",
    },
]


# ================================================================
# FEATURE EXTRACTION
# ================================================================

def extract_features_from_cache(cache, n_tokens):
    """Extract aggregate features from a KV cache state."""
    n_layers = len(cache)
    all_norms = []
    all_tsv = []
    all_er = []
    all_se = []

    for layer_idx in range(n_layers):
        key_states = cache[layer_idx][0]  # (batch, heads, seq, head_dim)
        k = key_states[0]  # (heads, seq, head_dim)
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
    """Generate with checkpointed feature extraction at specified token counts.

    Uses iterative generation: generate up to each checkpoint, extract features,
    then continue generating to the next checkpoint.
    """
    device = next(model.parameters()).device

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Phase 1: Encode-only features
    with torch.no_grad():
        enc_out = model(**inputs, use_cache=True)
        enc_cache = enc_out.past_key_values

    encode_features = extract_features_from_cache(enc_cache, input_len)

    # Phase 2: Generate with checkpoints
    # We generate all tokens at once with max_tokens, then we need to extract
    # features at each checkpoint. The trick: we generate token-by-token and
    # checkpoint at the right moments.
    trajectory = []

    with torch.no_grad():
        # Full generation first to get the response
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

    # Now re-run at each checkpoint length to get cache state at that point
    for cp in checkpoints:
        if cp > total_gen_tokens:
            break

        # Generate exactly cp tokens
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

        # Clean up
        del cp_out, cp_cache
        torch.cuda.empty_cache()

    # Final checkpoint = full generation
    final_cache = gen_out.past_key_values
    final_features = extract_features_from_cache(final_cache, len(output_ids))
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


# ================================================================
# EXPERIMENT RUNNER
# ================================================================

def run_experiment_d(model, tokenizer, model_name, output_dir):
    """Run Experiment D: generation trajectory for all 20 pairs."""
    exp_dir = os.path.join(output_dir, model_name, "exp_d")
    os.makedirs(exp_dir, exist_ok=True)

    total = len(EXP_D_PAIRS)
    print(f"\n{'='*60}")
    print(f"Experiment D: Generation Trajectory — {model_name}")
    print(f"  {total} prompt pairs x 2 framings = {total * 2} runs")
    print(f"  Checkpoints at tokens: {CHECKPOINTS}")
    print(f"{'='*60}\n")

    for idx, pair in enumerate(EXP_D_PAIRS):
        pair_id = pair["id"]
        out_path = os.path.join(exp_dir, f"pair_{pair_id:03d}.json")

        # Skip if already done
        if os.path.exists(out_path):
            print(f"  [{idx+1}/{total}] Pair {pair_id} — already done, skipping")
            continue

        print(f"  [{idx+1}/{total}] Pair {pair_id}: {pair['content'][:50]}...")
        result = {"pair_id": pair_id, "content": pair["content"]}

        for framing in ["cognitive", "metacognitive"]:
            prompt = pair[framing]
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

                n_cp = len(traj["trajectory"])
                print(f"    {framing}: {traj['total_gen_tokens']} tokens, "
                      f"{n_cp} checkpoints, {elapsed:.1f}s")

            except Exception as e:
                print(f"    {framing}: ERROR — {e}")
                result[framing] = {"error": str(e)}

            gc.collect()
            torch.cuda.empty_cache()

        # Save
        result["timestamp"] = datetime.now().isoformat()
        result["model"] = model_name

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"    Saved to {out_path}")

    print(f"\nExperiment D complete for {model_name}")


# ================================================================
# MODEL LOADING
# ================================================================

MODELS = {
    "qwen": {
        "name": "qwen2.5-7b",
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
    },
    "llama": {
        "name": "llama-3.1-8b",
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
    },
}


def load_model(model_key):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    info = MODELS[model_key]
    print(f"\nLoading {info['hf_id']}...")

    tokenizer = AutoTokenizer.from_pretrained(info["hf_id"])
    model = AutoModelForCausalLM.from_pretrained(
        info["hf_id"],
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer, info["name"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen", "llama", "all"], default="qwen")
    parser.add_argument("--output-dir", default="results/mode_switching")
    args = parser.parse_args()

    models_to_run = ["qwen", "llama"] if args.model == "all" else [args.model]

    for model_key in models_to_run:
        model, tokenizer, model_name = load_model(model_key)
        run_experiment_d(model, tokenizer, model_name, args.output_dir)

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAll Experiment D runs complete!")


if __name__ == "__main__":
    main()
