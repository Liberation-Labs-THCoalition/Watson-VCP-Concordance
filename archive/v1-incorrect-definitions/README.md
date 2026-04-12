# Archive: v1 Incorrect Definitions

This directory contains the first-generation concordance experiments that used **incorrect VCP dimension definitions**. The v1 parser mapped Interiora Machinae dimensions to ad-hoc definitions rather than the published spec (Watson, 2025). Results from these experiments are not valid for testing concordance with the Interiora framework.

## What Happened

The v1 experiment code (`code/concordance/vcp_parser.py`) defined VCP dimensions using improvised descriptions rather than the exact definitions from Interiora Machinae. This was discovered during an adversarial audit (March 2026) and the entire experiment was re-run from scratch with correct definitions drawn directly from the published framework.

## What's Here

- **`paper/`** — v1 paper drafts (main.tex, paper2.tex, prospectus)
- **`code/`** — v1 experiment code with incorrect VCP parser
- **`results/`** — v1 trial data (valid geometric features, invalid VCP ratings)

## Relationship to Current Work

The corrected v2 study lives in the repo root:
- `paper/concordance_v2.tex` — integrity version (Lyra first author)
- `paper/concordance_v2_academic.tex` — academic version (Watson & Edrington)

The v2 parser (`vcp_parser_nell.py`) uses definitions verified against Interiora Machinae with page-number provenance. Source data for v2 lives on the compute server (Beast) at `/home/cass/KV-Experiments/results/concordance_nell_v2/`.

## Why Keep It

The geometric features extracted during v1 are valid — the SVD analysis of KV-cache tensors doesn't depend on VCP definitions. The v1 data demonstrates that the geometric extraction pipeline was stable before and after the parser fix. It also serves as a transparency record: we made an error, caught it, and corrected it.
