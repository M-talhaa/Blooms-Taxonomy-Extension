"""Label new verbs with Bloom cognitive levels using the published production model.

A level is accepted when its calibrated score reaches the level's threshold. As in the
published verb bank (Algorithm 2 of the article), when two or more levels pass and the
highest calibrated score leads the second by at least 0.060, only the highest level is
kept; --no-margin reports every level that passes its threshold.

Examples
    python classify.py weld debug summarise
    python classify.py --file verbs.txt --out labelled.csv --soft
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bt4_scorer import BT4Scorer, LEVELS  # noqa: E402

ENCODER = "sentence-transformers/all-mpnet-base-v2"
TOP2_MARGIN = 0.060  # top-2 margin of the published bank (revision_pipeline/retrain.py)


def embed(verbs):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(ENCODER)
    return np.asarray(model.encode(list(verbs), normalize_embeddings=True), dtype=float)


def apply_top2_margin(scores, hard, delta=TOP2_MARGIN):
    """Keep only the top level when two or more levels pass and the top leads the second by delta."""
    hard = hard.copy()
    for i in range(hard.shape[0]):
        idx = np.where(hard[i] == 1)[0]
        if idx.size >= 2:
            order = idx[np.argsort(scores[i, idx])[::-1]]
            if scores[i, order[0]] - scores[i, order[1]] >= delta:
                hard[i, :] = 0
                hard[i, order[0]] = 1
    return hard


def label(verbs, soft_alpha=None, margin=True):
    scorer = BT4Scorer(HERE)
    X = embed(verbs)
    hard, scores = scorer.predict_binary(X)
    if margin:
        hard = apply_top2_margin(scores, hard)
    soft = scorer.suggest_binary(X, soft_alpha)[0] if soft_alpha else None
    rows = []
    for i, verb in enumerate(verbs):
        accepted = [LEVELS[j] for j in range(len(LEVELS)) if hard[i, j] == 1]
        primary = max(accepted, key=lambda l: scores[i, LEVELS.index(l)]) if accepted else "None"
        row = {
            "verb": verb,
            "primary": primary,
            "accepted": " ".join(accepted) if accepted else "",
        }
        if soft is not None:
            row["soft"] = " ".join(LEVELS[j] for j in range(len(LEVELS)) if soft[i, j] == 1)
        for j, lv in enumerate(LEVELS):
            row[f"score_{lv}"] = round(float(scores[i, j]), 3)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Assign Bloom levels to verbs.")
    ap.add_argument("verbs", nargs="*", help="verbs to label")
    ap.add_argument("--file", help="text file with one verb per line")
    ap.add_argument("--out", help="write results to this CSV instead of the screen")
    ap.add_argument("--soft", action="store_true",
                    help="also report the soft suggestion tier (thresholds scaled by 0.85)")
    ap.add_argument("--alpha", type=float, default=0.85, help="soft tier scale (default 0.85)")
    ap.add_argument("--no-margin", action="store_true",
                    help="report every level that passes its threshold (skip the top-2 margin)")
    args = ap.parse_args()

    verbs = list(args.verbs)
    if args.file:
        verbs += [ln.strip() for ln in Path(args.file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    verbs = [v.lower() for v in dict.fromkeys(verbs)]
    if not verbs:
        ap.error("give at least one verb, or use --file")

    rows = label(verbs, args.alpha if args.soft else None, margin=not args.no_margin)

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} verbs written to {args.out}")
        return

    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    main()
