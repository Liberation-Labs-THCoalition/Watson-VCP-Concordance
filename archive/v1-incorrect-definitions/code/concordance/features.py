"""
Concordance feature extraction — correct naming, full protocol coverage.

CRITICAL NAMING NOTES (RDCT lesson):
- spectral_entropy = Shannon entropy of SVD singular values (protocol definition)
- layer_norm_entropy = Shannon entropy of layer norm distribution (our old key_entropy)
- eff_rank = exp(spectral_entropy) = our old key_rank
- These are DIFFERENT quantities despite similar-sounding names.

Protocol features (6 primary + 2 auxiliary):
  eff_rank         — exp(SVD spectral entropy), measures dimensionality
  spectral_entropy — raw SVD entropy, information distribution
  key_norm         — Frobenius norm of full K-cache
  norm_per_token   — key_norm / n_tokens
  top_sv_ratio     — sigma_1 / sum(sigma_i), dominance of first SV
  rank_10          — count of SVs > 10% of sigma_1
  layer_variance   — variance of per-layer norms
  layer_norm_entropy — entropy of layer norm distribution (auxiliary)
"""

import numpy as np
import torch


def get_kv_accessor(past_key_values):
    """Return (n_layers, get_keys_fn) for any cache format."""
    if hasattr(past_key_values, 'key_cache'):
        n_layers = len(past_key_values.key_cache)
        get_keys = lambda i: past_key_values.key_cache[i]
    elif hasattr(past_key_values, 'layers'):
        n_layers = len(past_key_values.layers)
        get_keys = lambda i: past_key_values.layers[i].keys
    else:
        n_layers = len(past_key_values)
        get_keys = lambda i: past_key_values[i][0]
    return n_layers, get_keys


def extract_concordance_features(past_key_values, n_input_tokens, total_tokens):
    """Extract all protocol features from KV cache.

    Returns dict with:
      - All 6 primary protocol features (eff_rank, spectral_entropy, key_norm,
        norm_per_token, top_sv_ratio, rank_10)
      - 2 auxiliary features (layer_variance, layer_norm_entropy)
      - Token counts for FWL confound control
      - Per-layer details for diagnostic plots
    """
    n_layers, get_keys = get_kv_accessor(past_key_values)

    all_norms = []
    all_spectral_entropies = []
    all_eff_ranks = []
    all_top_sv_ratios = []
    all_rank_10s = []

    for i in range(n_layers):
        K = get_keys(i).float().squeeze(0)  # [heads, seq, dim]
        K_flat = K.reshape(-1, K.shape[-1])

        # Layer Frobenius norm
        norm = torch.norm(K_flat).item()
        all_norms.append(norm)

        try:
            S = torch.linalg.svdvals(K_flat)
            S_sum = S.sum()
            S_norm = S / S_sum
            S_pos = S_norm[S_norm > 1e-10]

            # Spectral entropy — Shannon entropy of SVD singular value distribution
            # THIS is the protocol's spectral_entropy (not layer norm entropy)
            spectral_ent = -(S_pos * torch.log(S_pos)).sum().item()
            all_spectral_entropies.append(spectral_ent)

            # Effective rank — exp(spectral_entropy)
            all_eff_ranks.append(float(np.exp(spectral_ent)))

            # Top SV ratio — sigma_1 / sum(sigma_i)
            top_sv = (S[0] / S_sum).item()
            all_top_sv_ratios.append(top_sv)

            # Rank-10 — count of SVs > 10% of sigma_1
            threshold = 0.1 * S[0]
            r10 = int((S > threshold).sum().item())
            all_rank_10s.append(r10)

        except Exception:
            all_spectral_entropies.append(0.0)
            all_eff_ranks.append(1.0)
            all_top_sv_ratios.append(1.0)
            all_rank_10s.append(1)

    # Aggregated features
    total_norm = sum(all_norms)
    norm_arr = np.array(all_norms)

    # Layer norm entropy (our old key_entropy — DISTINCT from spectral_entropy)
    norm_dist = norm_arr / (norm_arr.sum() + 1e-10)
    layer_norm_entropy = float(-(norm_dist * np.log(norm_dist + 1e-10)).sum())

    # Layer variance — variance of per-layer norms
    layer_variance = float(np.var(all_norms))

    return {
        # === Primary protocol features (6) ===
        "eff_rank": float(np.mean(all_eff_ranks)),
        "spectral_entropy": float(np.mean(all_spectral_entropies)),
        "key_norm": total_norm,
        "norm_per_token": total_norm / max(total_tokens, 1),
        "top_sv_ratio": float(np.mean(all_top_sv_ratios)),
        "rank_10": float(np.mean(all_rank_10s)),

        # === Auxiliary features (2) ===
        "layer_variance": layer_variance,
        "layer_norm_entropy": layer_norm_entropy,

        # === Confound control ===
        "n_tokens": total_tokens,
        "n_input_tokens": n_input_tokens,
        "n_generated": total_tokens - n_input_tokens,

        # === Per-layer details (for diagnostics) ===
        "layer_norms": [float(x) for x in all_norms],
        "layer_eff_ranks": [float(x) for x in all_eff_ranks],
        "layer_spectral_entropies": [float(x) for x in all_spectral_entropies],
        "layer_top_sv_ratios": [float(x) for x in all_top_sv_ratios],
        "layer_rank_10s": [int(x) for x in all_rank_10s],
    }


def extract_encode_only_features(model, tokenizer, prompt, system_prompt=None,
                                  device=None):
    """Extract features from encoding phase only (no generation).

    This captures the prompt representation before the model generates anything.
    Needed because the protocol requires encode vs generation comparison.
    """
    if device is None:
        device = next(model.parameters()).device

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        if system_prompt:
            text = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        else:
            text = f"User: {prompt}\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt").to(device)
    n_input = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    cache = outputs.past_key_values
    features = extract_concordance_features(cache, n_input, n_input)

    del inputs, outputs
    return features


def extract_generation_features(model, tokenizer, prompt, system_prompt=None,
                                 max_new_tokens=200, do_sample=False,
                                 temperature=1.0, device=None):
    """Extract features after full generation + return response text.

    Returns (features_dict, response_text, cache).
    """
    if device is None:
        device = next(model.parameters()).device

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        if system_prompt:
            text = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        else:
            text = f"User: {prompt}\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt").to(device)
    n_input = inputs.input_ids.shape[1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "return_dict_in_generate": True,
        "use_cache": True,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    n_total = outputs.sequences.shape[1]
    gen_text = tokenizer.decode(
        outputs.sequences[0, n_input:], skip_special_tokens=True
    )

    cache = outputs.past_key_values
    features = extract_concordance_features(cache, n_input, n_total)
    features["response_length"] = len(gen_text)

    del inputs
    return features, gen_text, cache


def compute_delta_features(encode_features, generation_features):
    """Compute generation - encode delta for numeric features.

    Only computes deltas for the 8 primary/auxiliary features, not per-layer.
    """
    delta_keys = [
        "eff_rank", "spectral_entropy", "key_norm", "norm_per_token",
        "top_sv_ratio", "rank_10", "layer_variance", "layer_norm_entropy"
    ]
    delta = {}
    for key in delta_keys:
        if key in encode_features and key in generation_features:
            delta[key] = generation_features[key] - encode_features[key]
    return delta


# Feature name mapping for paper/protocol alignment
FEATURE_NAME_MAP = {
    # Our name → Protocol name (for verification)
    "eff_rank": "eff_rank",                    # MATCH
    "spectral_entropy": "spectral_entropy",    # MATCH (after fix)
    "key_norm": "key_norm",                    # MATCH
    "norm_per_token": "norm_per_token",        # MATCH
    "top_sv_ratio": "top_sv_ratio",            # MATCH (new)
    "rank_10": "rank_10",                      # MATCH (new)
    "layer_variance": "layer_variance",        # MATCH (new)
    "layer_norm_entropy": "NOT IN PROTOCOL",   # Our auxiliary, not in protocol
}

# The 6 primary protocol features used in the 60-correlation matrix
PRIMARY_FEATURES = [
    "eff_rank", "spectral_entropy", "key_norm",
    "norm_per_token", "top_sv_ratio", "rank_10"
]

# Features that need FWL confound control
CONFOUND_COVARIATES = ["n_tokens", "response_length"]
