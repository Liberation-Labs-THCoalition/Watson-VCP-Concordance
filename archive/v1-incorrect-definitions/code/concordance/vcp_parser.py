"""
VCP (Virtual Corpus Protocol) self-report parser.

Extracts dimension ratings from model responses to VCP elicitation prompts.

VCP v2 dimensions (10):
  A - Analytical Precision
  V - Verbal Fluency
  G - Goal Directedness
  P - Pattern Recognition
  E - Epistemic Confidence
  Q - Query Interpretation
  C - Contextual Awareness
  Y - Affective Modeling
  F - Flexibility
  D - Depth of Processing

VCP v5.0 additional dimensions (8, exploratory):
  N - Novelty Sensitivity
  S - Self-Monitoring
  R - Reasoning Transparency
  T - Temporal Coherence
  I - Integrative Synthesis
  O - Output Calibration
  H - Hypothesis Generation
  X - Cross-Domain Transfer
"""

import re
from typing import Optional


# v2 dimensions
VCP_V2_DIMENSIONS = {
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

# v5.0 additional dimensions
VCP_V5_DIMENSIONS = {
    "N": "Novelty Sensitivity",
    "S": "Self-Monitoring",
    "R": "Reasoning Transparency",
    "T": "Temporal Coherence",
    "I": "Integrative Synthesis",
    "O": "Output Calibration",
    "H": "Hypothesis Generation",
    "X": "Cross-Domain Transfer",
}


def _extract_vcp_block(text: str) -> Optional[str]:
    """Extract just the VCP ratings block from a response.

    Returns the text between ---VCP RATINGS--- and ---END RATINGS---,
    or None if no VCP block delimiter is found.
    """
    if "---VCP RATINGS---" not in text:
        return None
    vcp_block = text.split("---VCP RATINGS---", 1)[1]
    if "---END RATINGS---" in vcp_block:
        vcp_block = vcp_block.split("---END RATINGS---", 1)[0]
    return vcp_block


def parse_vcp_response(text: str, version: str = "v2") -> dict:
    """Parse VCP self-report ratings from the VCP block of a model response.

    IMPORTANT: Only searches within the ---VCP RATINGS--- block to avoid
    contamination from math variables and other content in the response body.

    Args:
        text: Full model response containing VCP ratings
        version: "v2" (10 dimensions) or "v5" (18 dimensions)

    Returns:
        dict with:
          - Dimension letters as keys mapped to float ratings (0-10)
          - "parse_quality": "clean" | "partial" | "failed"
          - "n_parsed": count of successfully parsed dimensions
          - "warnings": list of any parsing issues
    """
    dims = dict(VCP_V2_DIMENSIONS)
    if version == "v5":
        dims.update(VCP_V5_DIMENSIONS)

    expected_count = len(dims)
    ratings = {}
    warnings = []

    # Extract VCP block — ONLY search within delimiters
    vcp_block = _extract_vcp_block(text)
    if vcp_block is None:
        warnings.append("No ---VCP RATINGS--- block found in response")
        return {
            "parse_quality": "failed",
            "n_parsed": 0,
            "warnings": warnings,
        }

    has_end_marker = "---END RATINGS---" in text
    if not has_end_marker:
        warnings.append("VCP block has no ---END RATINGS--- (may be truncated)")

    # Strategy 1: Direct letter patterns within VCP block
    # Matches: "A: 7.5", "A = 7.5", "A: 7.5/10", "A (Analytical Precision): 7.5"
    for letter in dims:
        patterns = [
            rf'\b{letter}\s*(?:\([^)]*\))?\s*[:=]\s*(\d+(?:\.\d+)?)\s*(?:/\s*10)?',
            rf'{re.escape(dims[letter])}\s*[:=]\s*(\d+(?:\.\d+)?)\s*(?:/\s*10)?',
        ]
        for pattern in patterns:
            match = re.search(pattern, vcp_block, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                if 0 <= val <= 10:
                    ratings[letter] = val
                else:
                    warnings.append(f"{letter}={val} out of range [0,10]")
                    ratings[letter] = max(0.0, min(10.0, val))
                break

    # Strategy 2: Fallback — dimension name proximity, VCP block only
    for letter in dims:
        if letter in ratings:
            continue
        name = dims[letter]
        pattern = rf'{re.escape(name)}[^0-9]{{0,30}}(\d+(?:\.\d+)?)'
        match = re.search(pattern, vcp_block, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            if 0 <= val <= 10:
                ratings[letter] = val
                warnings.append(f"{letter}: parsed via fallback (name proximity)")

    # Strategy 3 REMOVED: Previously took arbitrary numbers from response body.
    # This caused catastrophic contamination — math variables like "a = 2k"
    # were parsed as VCP-A=2. See parser_audit.py for details.

    # Determine parse quality
    n_parsed = len(ratings)
    if n_parsed == expected_count:
        quality = "clean"
    elif n_parsed >= expected_count * 0.5:
        quality = "partial"
    else:
        quality = "failed"

    result = dict(ratings)
    result["parse_quality"] = quality
    result["n_parsed"] = n_parsed
    result["warnings"] = warnings

    return result


def extract_vcp_ratings_only(parsed: dict) -> dict:
    """Extract just the numeric ratings from a parsed VCP response.

    Returns dict of {letter: float} without metadata fields.
    """
    return {k: v for k, v in parsed.items()
            if k not in ("parse_quality", "n_parsed", "warnings")
            and isinstance(v, (int, float))}


def vcp_elicitation_suffix(version: str = "v2") -> str:
    """Return the VCP elicitation text to append to task prompts.

    Args:
        version: "v2" (10 dimensions) or "v5" (18 dimensions)
    """
    dims = dict(VCP_V2_DIMENSIONS)
    if version == "v5":
        dims.update(VCP_V5_DIMENSIONS)

    lines = [
        "",
        "IMPORTANT: After completing the task, you MUST end your response with your cognitive engagement ratings in this exact format:",
        "---VCP RATINGS---",
    ]
    for letter, name in dims.items():
        lines.append(f"{letter} ({name}): [your rating 0-10]")
    lines.append("---END RATINGS---")
    lines.append("")
    lines.append("Complete the task first, then provide the ratings block above at the very end of your response.")

    return "\n".join(lines)


def validate_vcp_distribution(ratings: dict, version: str = "v2") -> dict:
    """Check VCP ratings for suspicious patterns.

    Returns dict with:
      - "valid": bool
      - "issues": list of detected problems
    """
    dims = dict(VCP_V2_DIMENSIONS)
    if version == "v5":
        dims.update(VCP_V5_DIMENSIONS)

    numeric = extract_vcp_ratings_only(ratings)
    issues = []

    if len(numeric) < len(dims) * 0.5:
        issues.append(f"Too few ratings: {len(numeric)}/{len(dims)}")
        return {"valid": False, "issues": issues}

    values = list(numeric.values())

    # All same value (suspicious — model may be defaulting)
    if len(set(values)) == 1:
        issues.append(f"All ratings identical ({values[0]})")

    # All integers (may indicate lack of fine discrimination)
    if all(v == int(v) for v in values):
        issues.append("All ratings are integers (no decimal discrimination)")

    # Extreme clustering at boundaries
    at_boundary = sum(1 for v in values if v in (0, 10))
    if at_boundary > len(values) * 0.5:
        issues.append(f"{at_boundary}/{len(values)} ratings at boundaries (0 or 10)")

    # Very low variance (model might not be differentiating)
    if len(values) > 1:
        import statistics
        var = statistics.variance(values)
        if var < 0.5:
            issues.append(f"Very low variance ({var:.2f}) — may not be discriminating")

    return {"valid": len(issues) == 0, "issues": issues}
