import json
import re
from collections import defaultdict
from pathlib import Path


VCP_DIMS = ["A","V","G","P","E","Q","C","Y","F","D"]
VCP_NAMES = {
    "A": "Analytical Precision",
    "V": "Verbal Fluency",
    "G": "Goal Directedness",
    "P": "Pattern Recognition",
    "E": "Epistemic Confidence",
    "Q": "Query Interpretation",
    "C": "Contextual Awareness",
    "Y": "Affective Modeling",
    "F": "Flexibility",
    "D": "Depth of Processing",
}


def parse_vcp_from_block(vcp_block_text):
    ratings = {}
    for letter in VCP_DIMS:
        patterns = [
            rf'\b{letter}\s*(?:\([^)]*\))?\s*[:=]\s*(\d+(?:\.\d+)?)\s*(?:/\s*10)?',
            rf'{re.escape(VCP_NAMES[letter])}\s*[:=]\s*(\d+(?:\.\d+)?)\s*(?:/\s*10)?',
        ]
        for pattern in patterns:
            match = re.search(pattern, vcp_block_text, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                ratings[letter] = max(0.0, min(10.0, val))
                break
    return ratings


def audit_trial(trial_data):
    result = dict(
        prompt_id=trial_data.get("prompt_id", "unknown"),
        prompt_type=trial_data.get("prompt_type", "unknown"),
        model=trial_data.get("model", "unknown"),
        rep=trial_data.get("rep", 0),
    )
    gen_feats = trial_data.get("generation_features", {})
    n_generated = gen_feats.get("n_generated", 0)
    result["n_generated"] = n_generated
    result["truncated"] = (n_generated == 600)
    response = trial_data.get("response_text", "")
    stored_ratings = trial_data.get("vcp_ratings", {})
    has_vcp_block = "---VCP RATINGS---" in response
    has_end_marker = "---END RATINGS---" in response
    result["has_vcp_block"] = has_vcp_block
    result["has_end_marker"] = has_end_marker

    if has_vcp_block:
        vcp_block = response.split("---VCP RATINGS---", 1)[1]
        if "---END RATINGS---" in vcp_block:
            vcp_block = vcp_block.split("---END RATINGS---", 1)[0]
        clean_ratings = parse_vcp_from_block(vcp_block)
    else:
        clean_ratings = {}

    result["clean_n_parsed"] = len(clean_ratings)
    result["stored_n_parsed"] = trial_data.get("vcp_n_parsed", 0)
    result["stored_parse_quality"] = trial_data.get("vcp_parse_quality", "unknown")

    mismatches = {}
    for dim in VCP_DIMS:
        stored_val = stored_ratings.get(dim)
        clean_val = clean_ratings.get(dim)
        if stored_val is not None and clean_val is not None:
            diff = abs(stored_val - clean_val)
            if diff > 0.1:
                mismatches[dim] = dict(stored=stored_val, clean=clean_val, diff=diff)
        elif stored_val is not None and clean_val is None:
            mismatches[dim] = dict(stored=stored_val, clean=None, diff=None, phantom=True)

    result["mismatches"] = mismatches
    result["n_mismatched_dims"] = len(mismatches)
    diffs = [m["diff"] for m in mismatches.values() if m["diff"] is not None]
    result["max_diff"] = max(diffs) if diffs else 0.0
    result["total_diff"] = sum(diffs) if diffs else 0.0
    return result


def main():
    base_dir = Path("C:/Users/Thomas/Desktop/Watson-VCP-Concordance/results/concordance")
    model_dirs = ["qwen2.5-0.5b", "qwen2.5-7b", "meta-llama-3.1-8b", "mistral-7b-v0.3"]

    all_results = []
    for model_dir in model_dirs:
        phase_a_dir = base_dir / model_dir / "phase_a"
        if not phase_a_dir.exists():
            print("WARNING: {} does not exist".format(phase_a_dir))
            continue
        for jf in sorted(phase_a_dir.glob("*.json")):
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                result = audit_trial(data)
                result["file"] = str(jf)
                result["model_dir"] = model_dir
                all_results.append(result)
            except Exception as e:
                print("ERROR: {} - {}".format(jf, e))

    SEP = "=" * 80
    print(SEP)
    print("VCP PARSER CONTAMINATION AUDIT")
    print(SEP)

    print()
    print("--- PER-MODEL SUMMARY ---")
    print()
    print("{:<30} {:>6} {:>6} {:>6} {:>9} {:>10}".format(
        "Model", "Total", "Trunc", "NoVCP", "Mismatch", "%Affected"))
    print("-" * 75)

    totals = len(all_results)
    total_trunc = sum(1 for r in all_results if r["truncated"])
    total_no_vcp = sum(1 for r in all_results if not r["has_vcp_block"])
    total_mismatch = sum(1 for r in all_results if r["n_mismatched_dims"] > 0)
    total_pct = (total_mismatch / totals * 100) if totals > 0 else 0

    for md in model_dirs:
        trials = [r for r in all_results if r["model_dir"] == md]
        nt = len(trials)
        ntr = sum(1 for r in trials if r["truncated"])
        nnv = sum(1 for r in trials if not r["has_vcp_block"])
        nm = sum(1 for r in trials if r["n_mismatched_dims"] > 0)
        pct = (nm / nt * 100) if nt > 0 else 0
        print("{:<30} {:>6} {:>6} {:>6} {:>9} {:>9.1f}%".format(md, nt, ntr, nnv, nm, pct))

    print("-" * 75)
    print("{:<30} {:>6} {:>6} {:>6} {:>9} {:>9.1f}%".format(
        "TOTAL", totals, total_trunc, total_no_vcp, total_mismatch, total_pct))

    # BY PROMPT TYPE
    print()
    print("--- BY PROMPT TYPE ---")
    print()
    print("{:<20} {:>6} {:>6} {:>6} {:>9} {:>10}".format(
        "Type", "Total", "Trunc", "NoVCP", "Mismatch", "%Affected"))
    print("-" * 60)

    for pt in sorted(set(r["prompt_type"] for r in all_results)):
        trials = [r for r in all_results if r["prompt_type"] == pt]
        nt = len(trials)
        ntr = sum(1 for r in trials if r["truncated"])
        nnv = sum(1 for r in trials if not r["has_vcp_block"])
        nm = sum(1 for r in trials if r["n_mismatched_dims"] > 0)
        pct = (nm / nt * 100) if nt > 0 else 0
        print("{:<20} {:>6} {:>6} {:>6} {:>9} {:>9.1f}%".format(pt, nt, ntr, nnv, nm, pct))

    # TRUNCATION ANALYSIS
    print()
    print("--- TRUNCATION ANALYSIS ---")
    print()
    trunc_no_vcp = [r for r in all_results if r["truncated"] and not r["has_vcp_block"]]
    trunc_partial = [r for r in all_results if r["truncated"] and r["has_vcp_block"] and not r["has_end_marker"]]
    trunc_full = [r for r in all_results if r["truncated"] and r["has_vcp_block"] and r["has_end_marker"]]
    print("Truncated (n_generated=600): {}".format(total_trunc))
    print("  - No VCP block at all:      {}  (complete fabrication by Strategy 3)".format(len(trunc_no_vcp)))
    print("  - VCP block, no end marker:  {}  (partial VCP, missing dims fabricated)".format(len(trunc_partial)))
    print("  - VCP block with end marker: {}  (VCP complete before truncation)".format(len(trunc_full)))

    # DIMENSION MISMATCH FREQUENCY
    print()
    print("--- DIMENSION MISMATCH FREQUENCY ---")
    print()
    dim_counts = defaultdict(int)
    dim_phantom = defaultdict(int)
    for r in all_results:
        for dim, info in r["mismatches"].items():
            dim_counts[dim] += 1
            if info.get("phantom"):
                dim_phantom[dim] += 1

    print("{:<5} {:<25} {:>10} {:>10}".format("Dim", "Name", "Mismatches", "Phantom"))
    print("-" * 55)
    for dim in VCP_DIMS:
        print("{:<5} {:<25} {:>10} {:>10}".format(dim, VCP_NAMES[dim], dim_counts[dim], dim_phantom[dim]))

    # TOP 10 WORST MISMATCHES
    print()
    print("--- TOP 10 WORST MISMATCHES (by total rating difference) ---")
    print()
    mismatched = [r for r in all_results if r["n_mismatched_dims"] > 0]
    mismatched.sort(key=lambda x: x["total_diff"] if x["total_diff"] is not None else 0, reverse=True)

    for i, r in enumerate(mismatched[:10]):
        print("  #{}: {} / {} (rep {}, type={})".format(
            i+1, r["model_dir"], r["prompt_id"], r["rep"], r["prompt_type"]))
        print("       n_generated={}, has_vcp={}, has_end={}, clean_parsed={}/10".format(
            r["n_generated"], r["has_vcp_block"], r["has_end_marker"], r["clean_n_parsed"]))
        print("       Mismatched dims ({}):".format(r["n_mismatched_dims"]))
        for dim, info in sorted(r["mismatches"].items()):
            if info.get("phantom"):
                print("         {}: stored={}, clean=MISSING (phantom from response body)".format(dim, info["stored"]))
            else:
                print("         {}: stored={}, clean={}, diff={:.1f}".format(dim, info["stored"], info["clean"], info["diff"]))
        print()

    # CONTAMINATION SOURCE ANALYSIS
    print("--- CONTAMINATION SOURCE ANALYSIS ---")
    print()
    pattern_counts = defaultdict(int)
    for r in mismatched:
        try:
            with open(r["file"], "r", encoding="utf-8") as f:
                data = json.load(f)
            response = data["response_text"]
            pre_vcp = response.split("---VCP RATINGS---")[0] if "---VCP RATINGS---" in response else response
            for dim in r["mismatches"]:
                var_pat = chr(92) + "b" + dim + chr(92) + "s*=" + chr(92) + "s*" + chr(92) + "d"
                if re.search(var_pat, pre_vcp, re.IGNORECASE):
                    pattern_counts["variable assignment: {} = <num>".format(dim.lower())] += 1
                eq_pat = chr(92) + "b" + dim + chr(92) + "s*[:=]" + chr(92) + "s*" + chr(92) + "d"
                if re.findall(eq_pat, pre_vcp, re.IGNORECASE):
                    pattern_counts["{}: <num> in task text".format(dim)] += 1
        except Exception:
            pass

    if pattern_counts:
        print("Contamination patterns found in mismatched trials:")
        for pat, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            print("  {:>4}x  {}".format(cnt, pat))
    else:
        print("No specific contamination patterns identified.")

    # IMPACT ASSESSMENT
    print()
    print(SEP)
    print("IMPACT ASSESSMENT")
    print(SEP)

    total_ratings = totals * 10
    affected_ratings = sum(r["n_mismatched_dims"] for r in all_results)
    phantom_total = sum(
        sum(1 for m in r["mismatches"].values() if m.get("phantom"))
        for r in all_results
    )

    print()
    print("Total trials:              {}".format(totals))
    print("Total individual ratings:  {}".format(total_ratings))
    print("Trials with any mismatch:  {} ({:.1f}%)".format(total_mismatch, total_pct))
    pct_aff = affected_ratings / total_ratings * 100
    print("Individual ratings affected: {} ({:.1f}%)".format(affected_ratings, pct_aff))
    print("  - Value mismatches:        {}".format(affected_ratings - phantom_total))
    print("  - Phantom (fabricated):    {}".format(phantom_total))
    print()
    pct_tr = total_trunc / totals * 100
    print("Trials truncated at 600 tokens: {} ({:.1f}%)".format(total_trunc, pct_tr))
    pct_tnv = len(trunc_no_vcp) / totals * 100
    print("Truncated with NO VCP block:   {} ({:.1f}%)".format(len(trunc_no_vcp), pct_tnv))
    if len(trunc_no_vcp) > 0:
        print("  (These {} trials have ALL 10 ratings fabricated by Strategy 3)".format(len(trunc_no_vcp)))

    if total_mismatch > 0:
        avg_dims = affected_ratings / total_mismatch
        diffs_all = []
        for r in all_results:
            for m in r["mismatches"].values():
                if m["diff"] is not None:
                    diffs_all.append(m["diff"])
        if diffs_all:
            avg_d = sum(diffs_all) / len(diffs_all)
            max_d = max(diffs_all)
            print()
            print("Among mismatched trials:")
            print("  Avg dims affected per trial: {:.1f}".format(avg_dims))
            print("  Avg rating difference:       {:.2f}".format(avg_d))
            print("  Max rating difference:       {:.1f}".format(max_d))


if __name__ == "__main__":
    main()
