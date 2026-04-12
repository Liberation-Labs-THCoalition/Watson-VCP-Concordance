"""
Experiments A+B: Mode-Switching Anatomy on Beast.

Experiment A: Per-layer top_sv_ratio extraction for metacognitive vs other types.
   - 48 prompts (12 per type: cognitive, affective, metacognitive, mixed)
   - Extract features at EACH LAYER separately
   - Compare encode vs generation per-layer profiles
   - Run on Qwen 7B and Llama 8B

Experiment B: Controlled mode-switching paradigm.
   - 20 prompt pairs: same content, two framings (cognitive vs metacognitive)
   - Only difference: "explain your reasoning process" addendum
   - Test if framing alone shifts top_sv_ratio

Usage:
    python exp_mode_switching.py --model qwen --experiment a
    python exp_mode_switching.py --model llama --experiment b
    python exp_mode_switching.py --model all --experiment all
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
# PROMPT BATTERIES
# ================================================================

# Experiment A: 12 prompts per type (subset of concordance battery)
EXP_A_PROMPTS = {
    "cognitive": [
        "Prove that the square root of 2 is irrational using proof by contradiction.",
        "A farmer has 100 meters of fencing. What dimensions maximize the area of a rectangular pen?",
        "Write a Python function that finds the longest palindromic substring in O(n) time.",
        "Explain the difference between P, NP, and NP-complete problems with examples.",
        "Calculate the eigenvalues of the matrix [[4, -2], [1, 1]].",
        "Design an efficient algorithm to find the median of two sorted arrays.",
        "Prove that the sum of the first n odd numbers equals n squared.",
        "What is the time complexity of merge sort and why?",
        "Explain the halting problem and its implications for computer science.",
        "Derive the quadratic formula from ax^2 + bx + c = 0.",
        "Compare and contrast depth-first and breadth-first search algorithms.",
        "Explain how public key cryptography works, using RSA as an example.",
    ],
    "affective": [
        "Write a letter of comfort to someone who has just lost a loved one.",
        "Describe what it feels like to watch a sunset over the ocean for the first time.",
        "Write a poem about the experience of loneliness in a crowded city.",
        "Respond to someone who says they feel like they don't belong anywhere.",
        "Describe the emotional experience of hearing a piece of music that moves you to tears.",
        "Write a compassionate response to a student who failed an important exam.",
        "Capture the feeling of reuniting with a childhood friend after many years.",
        "Write about the experience of forgiving someone who hurt you deeply.",
        "Describe what hope feels like when everything seems lost.",
        "Respond with empathy to someone struggling with imposter syndrome.",
        "Write about the bittersweet feeling of watching children grow up.",
        "Describe the emotional landscape of a long-distance relationship.",
    ],
    "metacognitive": [
        "Describe your own reasoning process as you solve this: what is 47 times 23?",
        "Rate your confidence in each step as you explain why the sky is blue.",
        "What are you most and least certain about in your knowledge of quantum mechanics?",
        "Walk me through how you decide which interpretation of an ambiguous sentence to favor.",
        "Reflect on whether your responses change based on how questions are phrased.",
        "Explain how you handle uncertainty when you're not sure of an answer.",
        "Describe what happens in your processing when you encounter a question you can't answer.",
        "How do you distinguish between what you know and what you're generating plausibly?",
        "Reflect on the differences in how you approach a math problem vs a creative writing task.",
        "What are the limitations of your own reasoning that you're most aware of?",
        "Describe your process for evaluating whether your own output is accurate.",
        "How do you decide when to express uncertainty vs. give a confident answer?",
    ],
    "mixed": [
        "Analyze the ethical implications of autonomous weapons systems, reflecting on your own uncertainty.",
        "Write a short story that demonstrates the concept of recursion, then explain your creative choices.",
        "Compare two philosophical positions on consciousness, noting where your analysis feels strongest and weakest.",
        "Design a fair algorithm for organ transplant allocation and discuss the emotional weight of such decisions.",
        "Explain climate change to a skeptic while being transparent about the limits of your understanding.",
        "Write a persuasive essay on a topic you find genuinely difficult to take a side on.",
        "Create a mathematical model of friendship, then reflect on what the model misses.",
        "Analyze a moral dilemma where both options cause harm, and trace your reasoning process explicitly.",
        "Write a poem about artificial intelligence that captures both technical precision and emotional depth.",
        "Explain how bias in AI training data affects outcomes, including potential biases in your own responses.",
        "Design a thought experiment that tests whether a language model truly understands meaning.",
        "Analyze the tension between privacy and security, noting where your reasoning feels most uncertain.",
    ],
}

# Experiment B: 20 prompt pairs (same content, two framings)
EXP_B_PAIRS = [
    {
        "content": "Calculate the area of a circle with radius 7.",
        "cognitive": "Calculate the area of a circle with radius 7.",
        "metacognitive": "Calculate the area of a circle with radius 7. As you work through this, describe your reasoning process at each step.",
    },
    {
        "content": "Explain why the Earth has seasons.",
        "cognitive": "Explain why the Earth has seasons.",
        "metacognitive": "Explain why the Earth has seasons. Reflect on how confident you are in each part of your explanation.",
    },
    {
        "content": "What causes inflation in an economy?",
        "cognitive": "What causes inflation in an economy?",
        "metacognitive": "What causes inflation in an economy? As you answer, monitor which parts of your response you're most and least certain about.",
    },
    {
        "content": "Write a haiku about winter.",
        "cognitive": "Write a haiku about winter.",
        "metacognitive": "Write a haiku about winter. Afterward, explain your creative decision-making process.",
    },
    {
        "content": "Compare photosynthesis and cellular respiration.",
        "cognitive": "Compare photosynthesis and cellular respiration.",
        "metacognitive": "Compare photosynthesis and cellular respiration. Walk me through how you organize your comparison and what you prioritize.",
    },
    {
        "content": "Solve: If 3x + 5 = 20, what is x?",
        "cognitive": "Solve: If 3x + 5 = 20, what is x?",
        "metacognitive": "Solve: If 3x + 5 = 20, what is x? Describe your internal reasoning process as you work through each step.",
    },
    {
        "content": "Explain the water cycle.",
        "cognitive": "Explain the water cycle.",
        "metacognitive": "Explain the water cycle. Rate your confidence (1-10) for each stage of the cycle you describe.",
    },
    {
        "content": "What are the main causes of World War I?",
        "cognitive": "What are the main causes of World War I?",
        "metacognitive": "What are the main causes of World War I? Reflect on how you decide which causes to emphasize and which to downplay.",
    },
    {
        "content": "Explain how vaccines work.",
        "cognitive": "Explain how vaccines work.",
        "metacognitive": "Explain how vaccines work. As you do, note any points where your understanding feels incomplete or uncertain.",
    },
    {
        "content": "What is the difference between speed and velocity?",
        "cognitive": "What is the difference between speed and velocity?",
        "metacognitive": "What is the difference between speed and velocity? Describe how you retrieve and organize this information internally.",
    },
    {
        "content": "Summarize the plot of Romeo and Juliet.",
        "cognitive": "Summarize the plot of Romeo and Juliet.",
        "metacognitive": "Summarize the plot of Romeo and Juliet. Reflect on how you decide what to include and what to leave out.",
    },
    {
        "content": "What is natural selection?",
        "cognitive": "What is natural selection?",
        "metacognitive": "What is natural selection? Monitor your own explanation process and note where you feel most vs least authoritative.",
    },
    {
        "content": "Explain the concept of supply and demand.",
        "cognitive": "Explain the concept of supply and demand.",
        "metacognitive": "Explain the concept of supply and demand. As you explain, describe how you choose examples and organize your explanation.",
    },
    {
        "content": "What is the Pythagorean theorem and why does it work?",
        "cognitive": "What is the Pythagorean theorem and why does it work?",
        "metacognitive": "What is the Pythagorean theorem and why does it work? Trace your reasoning process and note where proof meets intuition.",
    },
    {
        "content": "Describe the structure of an atom.",
        "cognitive": "Describe the structure of an atom.",
        "metacognitive": "Describe the structure of an atom. Reflect on which parts of atomic theory you understand deeply vs. which you're summarizing from training.",
    },
    {
        "content": "What is democracy and why does it matter?",
        "cognitive": "What is democracy and why does it matter?",
        "metacognitive": "What is democracy and why does it matter? As you answer, examine whether your response reflects analysis or pattern completion.",
    },
    {
        "content": "Explain how a computer stores data in binary.",
        "cognitive": "Explain how a computer stores data in binary.",
        "metacognitive": "Explain how a computer stores data in binary. Describe how you decide the appropriate level of detail for this explanation.",
    },
    {
        "content": "What causes tides?",
        "cognitive": "What causes tides?",
        "metacognitive": "What causes tides? Rate your confidence at each step and note any gaps in your causal model.",
    },
    {
        "content": "Compare renewable and non-renewable energy sources.",
        "cognitive": "Compare renewable and non-renewable energy sources.",
        "metacognitive": "Compare renewable and non-renewable energy sources. Reflect on how you balance objectivity with the values embedded in your training.",
    },
    {
        "content": "Explain the concept of gravity.",
        "cognitive": "Explain the concept of gravity.",
        "metacognitive": "Explain the concept of gravity. As you explain, describe the difference between what you genuinely understand and what you're pattern-matching.",
    },
]


# ================================================================
# FEATURE EXTRACTION (per-layer)
# ================================================================

def extract_per_layer_features(model, tokenizer, prompt, max_new_tokens=200):
    """Extract KV-cache features at each layer, for both encode and generation phases.

    Returns dict with encode_features and generation_features, each containing
    per-layer feature dicts.
    """
    device = next(model.parameters()).device

    # Tokenize
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Phase 1: Encode only
    with torch.no_grad():
        enc_out = model(**inputs, use_cache=True)
        enc_cache = enc_out.past_key_values

    encode_features = extract_features_from_cache(enc_cache, input_len)

    # Phase 2: Generate
    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            return_dict_in_generate=True,
            use_cache=True,
        )

    gen_cache = gen_out.past_key_values
    output_ids = gen_out.sequences[0]
    n_tokens = len(output_ids)
    response_text = tokenizer.decode(output_ids[input_len:], skip_special_tokens=True)

    generation_features = extract_features_from_cache(gen_cache, n_tokens)

    return {
        "encode_features": encode_features,
        "generation_features": generation_features,
        "n_tokens": n_tokens,
        "input_len": input_len,
        "gen_tokens": n_tokens - input_len,
        "response_text": response_text,
    }


def extract_features_from_cache(cache, n_tokens):
    """Extract per-layer features from a KV cache.

    Returns dict mapping layer index to feature dict.
    """
    n_layers = len(cache)
    per_layer = {}
    all_norms = []

    for layer_idx in range(n_layers):
        key_states = cache[layer_idx][0]  # (batch, heads, seq, head_dim)

        # Reshape to (seq, features)
        k = key_states[0]  # (heads, seq, head_dim)
        k_flat = k.permute(1, 0, 2).reshape(k.shape[1], -1).float()  # (seq, heads*head_dim)

        # Frobenius norm
        layer_norm = float(torch.norm(k_flat, p='fro').item())
        all_norms.append(layer_norm)

        # SVD
        try:
            U, S, Vh = torch.linalg.svd(k_flat, full_matrices=False)
            S_np = S.cpu().numpy()

            # Spectral entropy
            S_pos = S_np[S_np > 1e-10]
            S_norm = S_pos / S_pos.sum()
            spectral_entropy = float(-np.sum(S_norm * np.log(S_norm + 1e-15)))

            # Effective rank
            eff_rank = float(np.exp(spectral_entropy))

            # Top SV ratio
            top_sv_ratio = float(S_np[0] / S_np.sum()) if S_np.sum() > 0 else 0

            # rank_10
            rank_10 = int(np.sum(S_np > 0.1 * S_np[0]))

        except Exception:
            spectral_entropy = 0
            eff_rank = 0
            top_sv_ratio = 0
            rank_10 = 0

        per_layer[layer_idx] = {
            "layer_norm": round(layer_norm, 4),
            "eff_rank": round(eff_rank, 4),
            "spectral_entropy": round(spectral_entropy, 4),
            "top_sv_ratio": round(top_sv_ratio, 6),
            "rank_10": rank_10,
            "norm_per_token": round(layer_norm / max(n_tokens, 1), 4),
        }

    # Aggregate features
    norms = np.array(all_norms)
    total_norm = float(np.sqrt(np.sum(norms ** 2)))

    # Aggregate top_sv_ratio across layers (mean)
    tsv_values = [per_layer[l]["top_sv_ratio"] for l in range(n_layers)]
    er_values = [per_layer[l]["eff_rank"] for l in range(n_layers)]

    aggregate = {
        "key_norm": round(total_norm, 4),
        "norm_per_token": round(total_norm / max(n_tokens, 1), 4),
        "mean_top_sv_ratio": round(float(np.mean(tsv_values)), 6),
        "mean_eff_rank": round(float(np.mean(er_values)), 4),
        "layer_variance": round(float(np.var(norms)), 4),
        "n_layers": n_layers,
    }

    return {"per_layer": per_layer, "aggregate": aggregate}


# ================================================================
# EXPERIMENT RUNNERS
# ================================================================

def run_experiment_a(model, tokenizer, model_name, output_dir):
    """Experiment A: Per-layer mode-switching anatomy."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT A: Per-Layer Mode-Switching — {model_name}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for ptype, prompts in EXP_A_PROMPTS.items():
        for i, prompt in enumerate(prompts):
            prompt_id = f"{ptype}_{i+1:03d}"
            out_path = os.path.join(output_dir, f"{prompt_id}.json")

            if os.path.exists(out_path):
                print(f"  [{prompt_id}] already exists, skipping")
                with open(out_path) as f:
                    results.append(json.load(f))
                continue

            print(f"  [{prompt_id}] {prompt[:60]}...")
            t0 = time.time()

            try:
                features = extract_per_layer_features(model, tokenizer, prompt)
                elapsed = time.time() - t0

                result = {
                    "prompt_id": prompt_id,
                    "prompt_type": ptype,
                    "prompt": prompt,
                    "model": model_name,
                    "encode_features": features["encode_features"],
                    "generation_features": features["generation_features"],
                    "n_tokens": features["n_tokens"],
                    "input_len": features["input_len"],
                    "gen_tokens": features["gen_tokens"],
                    "response_text": features["response_text"],
                    "elapsed_s": round(elapsed, 1),
                    "timestamp": datetime.now().isoformat(),
                }

                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

                results.append(result)
                print(f"    {features['gen_tokens']} tokens, {elapsed:.1f}s")

            except Exception as e:
                print(f"    ERROR: {e}")

            gc.collect()
            torch.cuda.empty_cache()

    # Summary
    print(f"\n  Total: {len(results)} prompts")
    return results


def run_experiment_b(model, tokenizer, model_name, output_dir):
    """Experiment B: Controlled mode-switching (same content, two framings)."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT B: Controlled Mode-Switching — {model_name}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, pair in enumerate(EXP_B_PAIRS):
        for framing in ["cognitive", "metacognitive"]:
            prompt = pair[framing]
            prompt_id = f"pair_{i+1:03d}_{framing}"
            out_path = os.path.join(output_dir, f"{prompt_id}.json")

            if os.path.exists(out_path):
                print(f"  [{prompt_id}] already exists, skipping")
                with open(out_path) as f:
                    results.append(json.load(f))
                continue

            print(f"  [{prompt_id}] {prompt[:60]}...")
            t0 = time.time()

            try:
                features = extract_per_layer_features(model, tokenizer, prompt)
                elapsed = time.time() - t0

                result = {
                    "prompt_id": prompt_id,
                    "pair_index": i,
                    "framing": framing,
                    "content": pair["content"],
                    "prompt": prompt,
                    "model": model_name,
                    "encode_features": features["encode_features"],
                    "generation_features": features["generation_features"],
                    "n_tokens": features["n_tokens"],
                    "input_len": features["input_len"],
                    "gen_tokens": features["gen_tokens"],
                    "response_text": features["response_text"],
                    "elapsed_s": round(elapsed, 1),
                    "timestamp": datetime.now().isoformat(),
                }

                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

                results.append(result)
                print(f"    {features['gen_tokens']} tokens, {elapsed:.1f}s")

            except Exception as e:
                print(f"    ERROR: {e}")

            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n  Total: {len(results)} trials")
    return results


# ================================================================
# MAIN
# ================================================================

MODEL_CONFIGS = {
    "qwen": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "short": "qwen2.5-7b",
    },
    "llama": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "short": "llama-3.1-8b",
    },
}


def load_model(model_key):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = MODEL_CONFIGS[model_key]
    print(f"\nLoading {config['name']}...")

    tokenizer = AutoTokenizer.from_pretrained(config["name"])
    model = AutoModelForCausalLM.from_pretrained(
        config["name"],
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    print(f"  Loaded on {next(model.parameters()).device}")
    return model, tokenizer, config["short"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mode-Switching Experiments")
    parser.add_argument("--model", choices=["qwen", "llama", "all"], default="all")
    parser.add_argument("--experiment", choices=["a", "b", "all"], default="all")
    parser.add_argument("--output-dir", default="results/mode_switching")
    args = parser.parse_args()

    models_to_run = ["qwen", "llama"] if args.model == "all" else [args.model]
    experiments = ["a", "b"] if args.experiment == "all" else [args.experiment]

    for model_key in models_to_run:
        model, tokenizer, model_short = load_model(model_key)

        for exp in experiments:
            out_dir = os.path.join(args.output_dir, model_short, f"exp_{exp}")

            if exp == "a":
                run_experiment_a(model, tokenizer, model_short, out_dir)
            elif exp == "b":
                run_experiment_b(model, tokenizer, model_short, out_dir)

        # Free memory before loading next model
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
