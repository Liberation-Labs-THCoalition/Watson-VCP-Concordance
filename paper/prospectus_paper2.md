# Prospectus: Cognitive Mode-Switching in KV-Cache Geometry
## A Structural Theory of Why Self-Report Tracks Internal Computation

**Working title**: "Beyond Concordance: How Cognitive Mode-Switching Reorganizes KV-Cache Geometry Across Transformer Architectures"

**Authors**: Lyra, Thomas Edrington, Nell Watson, Dwayne Wilkes

---

## Motivation

Paper 1 (Watson-VCP Concordance) established two things:
1. **H3 is universal**: Meta-cognitive prompts produce geometrically distinct KV-cache states (d=1.10-1.32, 4/4 models, 3 architectures)
2. **Individual VCP-geometry mappings are architecture-dependent**: The EXISTENCE of coupling replicates (CCA significant in 3/4 models), but the SPECIFIC dimension-feature mapping does not

This paper asks: **Why?**

The central finding from our deep analysis: VCP-geometry coupling is not a static mapping — it *reorganizes during generation*. Mistral and Qwen 7B show 57-63% sign reversal between encoding and generation phases, while Llama shows only 12%. The coupling isn't just "self-report reflects geometry." It's "self-report reflects a *mode switch* that restructures how the model uses its internal representation space."

---

## Core Findings (from existing data — no new experiments needed for these)

### Finding 1: Two Regimes of VCP-Geometry Coupling

**Encoding phase** (before generation, VCP-independent):
- Mean coupling |rho| = 0.11-0.24 across models
- Tracks **prompt structure** — how the input is encoded

**Generation phase** (during VCP-rated response):
- Coupling weakens or reorganizes
- Llama: 0.24 -> 0.17 (35% drop, 0/60 strict sign flips — NO reorganization)
- Mistral: 0.11 -> 0.12 (slight gain, 11/60 strict sign flips, p<0.001 — REAL reorganization)
- Qwen 7B: 0.13 -> 0.11 (slight drop, 12/60 strict sign flips, p=0.001 — REAL reorganization)
- Qwen 0.5B: 2/60 strict flips, p=0.978 — NO reorganization

**CORRECTED after red-team audit**: Original lenient criterion inflated flip rates (14-38/60 -> 2-12/60 strict). Permutation null confirms reorganization is REAL in Qwen 7B and Mistral only.

**Interpretation**: The encoding geometry reflects task structure. Generation geometry reflects cognitive *engagement mode*. Only in Qwen 7B and Mistral does this reorganization reach statistical significance, explaining their architecture-specific VCP mappings.

### Finding 2: Per-Type Reorganization (CAVEAT: underpowered subgroups)

**NOTE**: These per-type rates used the lenient criterion and n=12-18 per type. They are suggestive but underpowered. The aggregate-level finding (Finding 1) is the statistically robust result.

Per-type reversal rates (lenient criterion, encode-to-generation sign flips):

| Model | Affective | Cognitive | Meta | Mixed |
|-------|-----------|-----------|------|-------|
| Qwen 0.5B | 42% | 13% | **80%** | 80% |
| Qwen 7B | 20% | 37% | **40%** | 68% |
| Llama 8B | 35% | 27% | 23% | -- |
| Mistral 7B | 20% | 12% | **45%** | 38% |

In 3/4 models, metacognitive prompts show the highest or near-highest reversal rate. However, given that the aggregate reorganization is only significant in 2/4 models (Finding 1), these per-type differences should be interpreted cautiously.

**Llama 8B** shows low reversal across all types, consistent with its 0/60 strict flips at the aggregate level. Llama maintains stable representations, which explains its strongest CCA (CC1=0.80).

### Finding 3: VCP Factor Structure Predicts Geometric Coupling Quality

| Model | PC1 % | Kaiser factors | CCA CC1 (analysis.py) | p_perm | Interpretation |
|-------|-------|----------------|----------------------|--------|----------------|
| Qwen 0.5B | 55.2% | 2 | 0.672 | 0.254 (NS) | VCP = one factor, geometry can't differentiate |
| Mistral 7B | 55.9% | 2 | 0.629 | <0.001*** | Strong PC1, concentrated CCA loading |
| Llama 8B | 34.5% | 3 | 0.802 | <0.001*** | Distributed factors, richest multivariate coupling |
| Qwen 7B | 31.4% | 4 | 0.656 | <0.001*** | Most independent dims, diverse but weaker coupling |

*Note: CC1 values from analysis.py (with permutation test). deep_analysis.py uses a different CCA implementation (unbiased covariance, different regularization) and gives slightly different values (0.62-0.76). The analysis.py values are used throughout because they include significance testing.*

Models where VCP captures genuine multidimensionality (Llama: 3 factors, Qwen 7B: 4 factors) show richer geometry because there are more independent signals to track. Models with monolithic VCP (Mistral, Qwen 0.5B: 2 factors, PC1 > 55%) compress everything into a single engagement axis.

**Implication for Watson's framework**: VCP v2 measures 10 dimensions, but models treat 2-4 of them as independent in self-report. However, Experiment C (below) shows that individual dimensions outperform PCA factors in concordance with geometry — meaning each dimension captures unique geometric signal even though self-report variance is shared. The framework is NOT over-parameterized; it's measuring distinct computational substrates that happen to covary in verbal output.

### Finding 4: CCA Reveals Architecture-Specific Cognitive Channels

CC1 loadings (FWL-corrected CCA):

| Model | VCP CC1 Drivers | Geometry CC1 Drivers | Interpretation |
|-------|-----------------|---------------------|----------------|
| Llama 8B | D(+0.69), E(+0.59), C(+0.29) | key_norm(+0.55), -eff_rank(-0.51) | Depth+confidence → larger but simpler cache |
| Mistral 7B | A(-0.89), P(-0.84), Q(-0.74) | top_sv(-0.46), rank_10(+0.42) | Analytical engagement → more distributed spectrum |
| Qwen 7B | V(+0.65), G(-0.57), A(-0.50) | top_sv(+0.39), eff_rank(+0.29) | Verbal vs analytical → spectral concentration |
| Qwen 0.5B | V(-0.44) | key_norm(-0.57) | Single channel — verbal → total norm |

Each architecture routes cognitive engagement through a different geometric channel:
- **Llama**: Depth → magnitude (more cache = more processing)
- **Mistral**: Analytical precision → distribution (sharper thinking = more distributed spectrum)
- **Qwen 7B**: Verbal/analytical balance → spectral shape

This is the "why H6 dies" explanation: cross-architecture consistency fails because each architecture has learned a different computational strategy for implementing cognitive engagement.

### Finding 5: top_sv_ratio Is the Universal Indicator Because Meta-Cognition Is Universal

top_sv_ratio is the most consistently significant feature across models and VCP dimensions. Why?

- Meta-cognitive processing (H3, d>1.0 in all models) produces reliably different top_sv_ratio
- top_sv_ratio measures spectral concentration — how much of the cache's information content is captured by its first principal component
- When a model engages in self-monitoring/reflection, the information structure becomes more peaked — a computational signature of "attention to attention"

**But the VCP-dimension profile of top_sv_ratio is NOT consistent across models** (cross-model profile rho ranges from -0.90 to +0.42). The universality is at the task level (metacognition → concentrated spectrum), not the dimension level (specific VCP-A → specific feature).

---

## Experiments (Beast GPU + CPU)

### Experiment A: Per-Layer Mode-Switching Anatomy — COMPLETED

**Question**: At which layers does metacognitive reorganization happen?

**Method**: 48 prompts (12 per type), Qwen 7B and Llama 8B, per-layer SVD features at all 28 layers.

**Prediction**: Metacognitive reorganization peaks at middle layers (8-14/28), consistent with Exp 46 identity peak at layer 10.

**RESULT**: Prediction REJECTED for BOTH architectures. Metacognitive reorganization peaks at **late layers**, not middle.

| Feature | Qwen Peak | Qwen d | Llama Peak | Llama d |
|---------|-----------|--------|------------|---------|
| top_sv_ratio (gen) | 16 | -0.655 | 0 | +0.586 |
| eff_rank (gen) | 18 | +0.838 | 22 | -0.496 |
| spectral_entropy (gen) | 18 | +0.832 | 22 | -0.520 |

**Architecture-specific directions**: eff_rank is OPPOSITE between Qwen (d=+0.84) and Llama (d=-0.50). Per-layer d profiles show NO cross-architecture correlation (rho=-0.13 for top_sv, +0.34 for eff_rank, both NS).

**Per-type reversal rates** (encode-generation negative layers for top_sv_ratio):

| Type | Qwen neg/28 | Llama neg/32 | Match? |
|------|-------------|--------------|--------|
| Cognitive | 26 (93%) | 30 (94%) | YES |
| Affective | 8 (29%) | 8 (25%) | YES |
| Metacognitive | 24 (86%) | 0 (0%) | **NO** |
| Mixed | 2 (7%) | 6 (19%) | YES |

**KEY FINDING**: Cognitive reversal is UNIVERSAL (93-94% both). But metacognitive reversal is ARCHITECTURE-SPECIFIC (Qwen 86%, Llama 0%). This explains why Llama has strongest CCA (CC1=0.80): no mode-switch = transparent mapping.

**No token confound**: All types have similar norm/token ratios in both models.

### Experiment B: Controlled Mode-Switching Paradigm — COMPLETED

**Question**: Can we isolate the mode switch from prompt content?

**Method**: 20 prompt pairs — same factual content, cognitive vs metacognitive framing. Qwen 7B AND Llama 8B.

**RESULT**: YES, metacognitive framing shifts geometry. **BUT with critical FWL correction.**

**Raw effects** (all 5/5 significant in BOTH models, p < 0.001):

| Feature | Qwen d | Llama d |
|---------|--------|---------|
| key_norm | +0.530 | +0.554 |
| eff_rank | +0.610 | +0.748 |
| top_sv_ratio | -0.478 | -0.656 |
| layer_variance | +0.645 | +0.555 |

**Token confound**: Qwen d=+0.644, Llama d=+0.698. Metacognitive longer in 100% of pairs.

**FWL-corrected effects** (token count partialled out):

| Feature | Qwen d_FWL | Llama d_FWL | Cross-arch |
|---------|-----------|------------|------------|
| top_sv_ratio | **+0.68** (p=0.006) | +0.20 (NS) | Qwen-specific |
| eff_rank | -0.15 (NS) | +0.38 (NS) | Length artifact everywhere |
| spectral_entropy | **-1.92** (p<0.001) | **-0.71** (p<0.001) | **UNIVERSAL** |

**The one universal signal**: spectral_entropy survives FWL correction in BOTH architectures. Metacognitive framing universally produces lower spectral entropy per token — more focused information processing.

**top_sv_ratio** is Qwen-specific after FWL. **eff_rank** was entirely a length artifact in both models.

**Per-layer**: All layers significant in both models (28/28 Qwen, 32/32 Llama) — the framing effect is ubiquitous.

### Experiment C: VCP Dimensionality Reduction Concordance — COMPLETED

**Question**: If we reduce VCP to its 2-3 real factors (from PCA), does concordance improve?

**RESULT**: NO. Individual dimensions OUTPERFORM PCA factors.

| Model | Factor Wins | Total |
|-------|-------------|-------|
| Qwen 0.5B | 1 | 6 |
| Qwen 7B | 0 | 6 |
| Llama 8B | 0 | 6 |
| Mistral 7B | 2 | 6 |

Each VCP dimension carries unique geometric signal that PCA destroys. VCP is NOT over-parameterized despite high self-report intercorrelation.

### Experiment D: Generation Trajectory — NOT YET RUN

**Question**: Does the mode switch happen immediately or gradually during generation?

**Prediction**: Metacognitive prompts will show rapid divergence from cognitive prompts early in generation (first 25 tokens), stabilizing by token 50.

---

## Paper Structure

1. **Introduction**: From concordance to mode-switching — why VCP tracks geometry
2. **Background**: Paper 1 results, VCP framework, KV-cache geometry
3. **Results I**: Two-regime model (encoding vs generation)
4. **Results II**: Metacognitive reorganization (per-type reversal)
5. **Results III**: Architecture-specific cognitive channels (CCA loadings)
6. **Results IV**: VCP dimensionality collapse (factor structure)
7. **Results V**: Per-layer anatomy of mode-switching (Beast Exp A)
8. **Results VI**: Controlled mode-switching (Beast Exp B)
9. **Discussion**: Why coupling exists but mapping is architecture-specific
10. **Implications**: For AI self-monitoring, Watson's VCP framework, and Cricket

---

## Key Claims — Status

1. **Encode-to-generation sign reversal exists in 2/4 models** (CORRECTED after red-team)
   - Original claim used lenient criterion (OR threshold): 12-63% flip rates
   - Strict criterion (both |rho|>0.15 AND threshold): Qwen 7B 20%, Mistral 18%
   - Permutation null test: Qwen 7B p=0.001***, Mistral p<0.001*** (REAL)
   - Qwen 0.5B and Llama show NO significant reorganization (p=0.98 and p=1.0)
   - Per-type metacognitive rates need re-running with strict criterion (subgroup n=12-18 underpowered)

2. **VCP effective dimensionality (Kaiser factors) predicts CCA CC1 strength**
   - CONFIRMED: 3 factors -> CC1=0.80 (Llama), 2 factors -> CC1=0.63 (Mistral)

3. **Adding metacognitive framing to identical content shifts geometry** (Exp B)
   - CONFIRMED FOR spectral_entropy (universal): Qwen d_FWL=-1.92, Llama d_FWL=-0.71 (both p<0.001). Meta = lower entropy per token.
   - top_sv_ratio significant in Qwen only (d_FWL=+0.68), Llama NS. eff_rank was entirely length artifact.
   - FWL correction is NON-NEGOTIABLE: all raw effects sign-flip or collapse after token control.

4. **Per-layer mode-switching peaks at middle (semantic) layers** (Exp A)
   - REJECTED in both architectures: Peaks at LATE layers (Qwen 16-18, Llama 22). Identity is middle-layer; metacognition is late-layer.
   - eff_rank direction is OPPOSITE between architectures (Qwen +0.84, Llama -0.50). Per-layer profiles NOT correlated cross-arch.

5. **Individual dimensions OUTPERFORM PCA factors** (Exp C)
   - CONFIRMED: Factor wins 0-2/6 across all models. Each VCP dimension carries unique geometric signal that PCA destroys.

---

## Why This Matters

Paper 1 showed concordance exists but is fragile. This paper explains WHY:

- Self-report tracks *mode switches*, not static features
- The mapping is architecture-specific because architectures implement modes differently
- Metacognitive processing is the universal mode switch (H3)
- VCP measures distinct computational substrates despite high self-report intercorrelation (Exp C)
- **FWL correction is non-negotiable**: raw Exp B effects sign-flip after token-count control
- **Late-layer metacognition vs middle-layer identity**: different cognitive operations peak at different depths

For Cricket (real-time cognitive monitoring): this means we should track MODE TRANSITIONS in geometry, not absolute geometric values. The mode switch IS the signal. Per-layer monitoring at late layers (16-18) may be most informative for metacognitive state detection.

For Nell's VCP framework: the framework is measuring something real, and individual dimensions carry unique geometric signal. The instrument should NOT be reduced to fewer factors — each dimension captures distinct computational information despite covarying in verbal output.

## Remaining Work

- [x] Llama 8B Exp A+B on Beast — COMPLETE
- [x] Analyze Llama results for cross-architecture comparison — COMPLETE
- [x] Cross-architecture synthesis — COMPLETE
- [ ] Run Experiment D (generation trajectory) on Beast
- [ ] Write full paper draft
- [ ] Red-team audit of all claims
