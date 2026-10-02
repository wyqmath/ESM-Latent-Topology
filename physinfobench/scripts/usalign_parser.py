"""Parse the two length-normalized TM-scores printed by US-align."""
import math
import re


_TM_LINE = re.compile(
    r"TM-score\s*=\s*([0-9]*\.?[0-9]+).*?normalized by length of\s+"
    r"(Structure_1|Structure_2|Chain_1|Chain_2)", re.IGNORECASE
)


def parse_usalign_scores(stdout):
    """Return (score-for-input-1, score-for-input-2, min); raise if incomplete."""
    aliases = {"structure_1": 1, "chain_1": 1, "structure_2": 2, "chain_2": 2}
    scores = {}
    for line in stdout.splitlines():
        match = _TM_LINE.search(line)
        if not match:
            continue
        value = float(match.group(1))
        label = aliases[match.group(2).lower()]
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid TM-score: {value}")
        if label in scores and not math.isclose(scores[label], value, abs_tol=1e-12):
            raise ValueError(f"Conflicting TM-scores for input {label}")
        scores[label] = value
    if set(scores) != {1, 2}:
        raise ValueError(f"Expected two normalized TM-scores; parsed {scores}")
    return scores[1], scores[2], min(scores.values())
