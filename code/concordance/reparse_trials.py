"""
Re-parse all trial VCP ratings using the fixed parser (VCP-block-only).

The original parser searched the ENTIRE response text with case-insensitive
regex, causing math variables (e.g., "a = 2k") to be parsed as VCP ratings.
Strategy 3 fabricated ratings from arbitrary numbers in the response body.

This script:
1. Reads each trial JSON
2. Re-parses response_text with the fixed parser
3. Updates vcp_ratings, vcp_parse_quality, vcp_n_parsed
4. Reports contamination statistics
"""

import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concordance.vcp_parser import parse_vcp_response, extract_vcp_ratings_only


def reparse_all(results_dir, dry_run=False):
    model_dirs = [d for d in os.listdir(results_dir)
                  if os.path.isdir(os.path.join(results_dir, d))
                  and d not in ("analysis", "pilot")]

    total = 0
    changed = 0
    quality_changes = defaultdict(int)
    rating_diffs = []

    SEP = "=" * 70

    for model_short in sorted(model_dirs):
        for phase in ("phase_a", "phase_b"):
            phase_dir = os.path.join(results_dir, model_short, phase)
            if not os.path.isdir(phase_dir):
                continue

            for fpath in sorted(glob.glob(os.path.join(phase_dir, "*.json"))):
                if os.path.basename(fpath).startswith("_") or fpath.endswith("summary.json"):
                    continue

                with open(fpath, "r", encoding="utf-8") as f:
                    trial = json.load(f)

                response = trial.get("response_text", "")
                old_ratings = trial.get("vcp_ratings", {})
                old_quality = trial.get("vcp_parse_quality", "unknown")
                old_n = trial.get("vcp_n_parsed", 0)

                # Re-parse with fixed parser
                parsed = parse_vcp_response(response)
                new_ratings = extract_vcp_ratings_only(parsed)
                new_quality = parsed["parse_quality"]
                new_n = parsed["n_parsed"]

                total += 1

                # Check for differences
                this_changed = False
                if new_quality != old_quality:
                    quality_changes[f"{old_quality}->{new_quality}"] += 1
                    this_changed = True

                for dim in "AVGPEQCYFD":
                    old_val = old_ratings.get(dim)
                    new_val = new_ratings.get(dim)
                    if old_val is not None and new_val is not None:
                        diff = abs(old_val - new_val)
                        if diff > 0.01:
                            rating_diffs.append((model_short, os.path.basename(fpath), dim, old_val, new_val, diff))
                            this_changed = True
                    elif old_val is not None and new_val is None:
                        rating_diffs.append((model_short, os.path.basename(fpath), dim, old_val, None, None))
                        this_changed = True

                if this_changed:
                    changed += 1

                # Update trial
                if not dry_run:
                    trial["vcp_ratings"] = new_ratings
                    trial["vcp_parse_quality"] = new_quality
                    trial["vcp_n_parsed"] = new_n
                    trial["vcp_warnings"] = parsed["warnings"]
                    # Keep old ratings for audit trail
                    trial["vcp_ratings_original"] = old_ratings
                    trial["vcp_parse_quality_original"] = old_quality

                    with open(fpath, "w", encoding="utf-8") as f:
                        json.dump(trial, f, indent=2, ensure_ascii=False)

    # Report
    print(SEP)
    print("VCP RE-PARSE RESULTS" + (" (DRY RUN)" if dry_run else ""))
    print(SEP)
    print(f"\nTotal trials processed: {total}")
    print(f"Trials with changes:   {changed} ({changed/total*100:.1f}%)")

    if quality_changes:
        print(f"\nParse quality transitions:")
        for transition, count in sorted(quality_changes.items(), key=lambda x: -x[1]):
            print(f"  {transition}: {count}")

    if rating_diffs:
        print(f"\nIndividual rating changes: {len(rating_diffs)}")
        removed = sum(1 for r in rating_diffs if r[4] is None)
        value_changed = sum(1 for r in rating_diffs if r[4] is not None)
        print(f"  Phantom ratings removed: {removed}")
        print(f"  Values corrected: {value_changed}")

        if value_changed > 0:
            diffs = [r[5] for r in rating_diffs if r[5] is not None]
            print(f"  Mean |diff|: {sum(diffs)/len(diffs):.2f}")
            print(f"  Max  |diff|: {max(diffs):.1f}")

        print(f"\nTop 10 worst corrections:")
        # Sort by diff magnitude, phantom removals last
        sorted_diffs = sorted(rating_diffs,
                              key=lambda x: x[5] if x[5] is not None else 999,
                              reverse=True)
        for model, fname, dim, old, new, diff in sorted_diffs[:10]:
            if new is None:
                print(f"  {model}/{fname} {dim}: {old} -> REMOVED (phantom)")
            else:
                print(f"  {model}/{fname} {dim}: {old} -> {new} (diff={diff:.1f})")

    # Final quality distribution
    print(f"\n{SEP}")
    if not dry_run:
        print("All trials updated. Original ratings preserved in vcp_ratings_original.")
    else:
        print("DRY RUN: No files modified. Run without --dry-run to apply changes.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Re-parse VCP ratings with fixed parser")
    parser.add_argument("--results-dir", default="results/concordance")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without modifying files")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        args.results_dir = "C:/Users/Thomas/Desktop/Watson-VCP-Concordance/results/concordance"

    reparse_all(args.results_dir, dry_run=args.dry_run)
