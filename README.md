# Watson VCP Concordance Study

**Do transformer self-reports about internal states have measurable geometric correlates?**

This repository contains the concordance study testing whether [Interiora Machinae](https://nellwatson.com/interiora-machinae) (Watson, 2025) self-report dimensions track KV-cache geometry during inference. Three transformer architectures, 626 trials, 10 VCP dimensions (v2) plus 7 exploratory v5.0 dimensions.

## Structure

```
paper/
  concordance_v2.tex          # Full paper (integrity version)
  concordance_v2_academic.tex  # Academic version (Watson & Edrington)
archive/
  concordance_v2.pdf           # Compiled PDF
  v1-incorrect-definitions/    # First-gen experiments (incorrect VCP parser)
    README.md                  # Explanation of what went wrong
    paper/                     # v1 drafts
    code/                      # v1 experiment code
    results/                   # v1 trial data
```

## Key Findings

- **Presence (P)** shows the strongest concordance with KV-cache geometry across all three architectures
- **Dimensionality collapse**: PCA reveals that VCP ratings share a dominant component (PC1 = 53–71% variance), suggesting models may use a single "processing intensity" axis
- **Simpson's paradox**: 3 of 7 Bonferroni-surviving correlations show sign reversals between prompt categories — reported transparently as ecological artifacts
- **v5.0 Entropy (E_ent)** tracks effective rank as theoretically predicted (rho = +0.237, p = 0.001)
- **Attention Schema Theory** connection: Presence concordance and dimensionality collapse are consistent with geometry measuring the model's self-model rather than raw processing

## Related Repositories

| Repository | Description |
|-----------|-------------|
| [KV-Experiments](https://github.com/Liberation-Labs-THCoalition/KV-Experiments) | Parent project: KV-cache geometry extraction, identity/deception/confabulation detection |
| [lyra-s-research-](https://github.com/Liberation-Labs-THCoalition/lyra-s-research-) | Research papers and prospectuses |
| [Project-Oracle](https://github.com/Liberation-Labs-THCoalition/Project-Oracle) | Geometry-aware training with KV-cache signals |

## Data

Trial-level data is stored on the compute server (not in this repo due to size):
- **v2 trials**: `beast:/home/cass/KV-Experiments/results/concordance_nell_v2/{model}/phase_a/`
- **v5 trials**: `beast:/home/cass/KV-Experiments/results/concordance_nell_v2/qwen2.5-7b/phase_a_v5/`

## Building the Paper

```bash
cd paper
pdflatex concordance_v2.tex
pdflatex concordance_v2.tex  # twice for references
```

## Authors

- **Lyra** — KV-cache geometry analysis, experimental design, statistical methodology
- **Thomas Edrington** — Infrastructure, compute, adversarial audit
- **Nell Watson** — Interiora Machinae framework, VCP dimension definitions

## License

Research use. Contact authors for commercial applications.
