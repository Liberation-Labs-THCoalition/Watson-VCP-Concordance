# Watson VCP Concordance — AI Reviewer Guide

## What This Repo Is

An empirical test of whether Nell Watson's Interiora Machinae self-report dimensions have geometric correlates in the KV-cache of transformer models during inference. Three architectures tested (Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3), 626 usable trials across 6 prompt categories.

## Paper Versions

- **`paper/concordance_v2.tex`** — Integrity version. Lyra (AI researcher) as first author, includes first-person reflection blocks. This is the version that represents how the research was actually conducted.
- **`paper/concordance_v2_academic.tex`** — Academic version. Watson & Edrington as authors, AI assistance noted. Same analysis, no first-person framing. Suitable for traditional venues.

Both versions contain identical analysis, hypotheses, results, and the v5.0 exploratory appendix.

## Key Claims to Verify

1. **Presence (P) concordance**: Strongest single-dimension signal. Spearman rho values are FWL-residualized (token count removed). Check Table 2.
2. **PCA structure**: PC1 variance explained differs by model (Qwen 70.6%, Llama 53.2%, Mistral 53.7%). These are computed on trials where all 10 VCP dimensions are non-null, so N_pca < N_analysis.
3. **Simpson's paradox**: 3/7 Bonferroni survivors show sign reversals between prompt categories. This is reported honestly — the paper flags these as ecological artifacts.
4. **v5.0 Appendix**: Exploratory analysis of 7 new Interiora v5.0 dimensions on Qwen2.5-7B only (261 trials). E_ent (Entropy) tracks effective rank as predicted. PG (Predictive Gradient) is a Simpson's artifact.

## Source Data

Trial-level JSON data lives on the compute server, not in this repo (too large for git). The v1 data in `archive/` uses incorrect VCP definitions — see `archive/v1-incorrect-definitions/README.md`.

Canonical v2 data: `beast:/home/cass/KV-Experiments/results/concordance_nell_v2/`

## Related Work

This study builds on KV-cache geometry findings from the parent project:
- **KV-Experiments repo**: [Liberation-Labs-THCoalition/KV-Experiments](https://github.com/Liberation-Labs-THCoalition/KV-Experiments)
- **Consolidated paper**: "Detecting Misalignment During Inference in the KV Cache" (Zenodo, 2026)
- **Interiora Machinae**: Watson (2025) — the self-report framework being tested

## FWL Is Mandatory

Every correlation in this paper uses Frisch-Waugh-Lovell residualization to remove token-count confounds. Without FWL, 53/60 sign-flips occur in the parent dataset. If you see a raw (non-FWL) correlation reported as a finding, that's an error.

## Controls Applied

14-control battery plus 4 experiment-specific controls. Key ones:
- FWL token-count residualization (mandatory)
- Simpson's paradox within-category check (3/7 survivors flagged)
- Bonferroni correction (32 primary comparisons)
- ICC for self-report reliability (0.53, acknowledged as limitation)
- Encoding-only baseline (signal committed at encoding for top features)
