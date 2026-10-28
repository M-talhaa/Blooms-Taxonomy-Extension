"""Normalization pass over the raw candidate verb pool (Section 2.2.1 of the article).

Inputs (both committed, so this runs on a fresh clone):
  raw_candidates_1824.txt    the 1,824 raw candidate verbs of the original submission, one per
                             line, in the row order of the original-submission bank (the Verb
                             column of results/Verb List Classified by Model.csv at commit
                             2a87a33).
  core_notebook_original.csv the 358-verb core of the original submission. Normalization was run
                             against this core, before the core was audited and merged; the two
                             verbs that the merge with the Stanny and Newton lists added to the
                             core ('break', 'practice') are removed from the pool afterwards by
                             merge_core.py (1,652 -> 1,650).

Rules, in order, each recorded in the audit trail:
1. WordNet verb lemmatization (surface forms -> lemma), lowercase.
2. British -ise/-yse spelling merged into the American -ize/-yze form only when
   doing so resolves a real duplicate, that is, when the American form exists in
   the core list or in the candidate pool. Never invents forms. No WordNet-synset
   condition is applied in this rule.
3. Drop pure auxiliaries: be, have, do.
4. Drop candidates whose normalized lemma is a core verb (leakage removal).
5. Dedupe: multiple candidates collapsing to one lemma keep a single entry
   (the first in input order).

Outputs: cleaned_candidates.txt (1,652 lemmas), normalization_audit.csv (one row per raw
candidate) and a printed summary. By default they are written next to this script, which
reproduces the committed files exactly; --outdir writes them elsewhere.

Needs the NLTK WordNet data. It is downloaded once if missing; otherwise no network is used.

Run: python normalize.py [--outdir DIR]
"""
import argparse
import csv
import os
from collections import Counter

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_CANDIDATES = os.path.join(HERE, "raw_candidates_1824.txt")
CORE_AT_NORMALIZATION = os.path.join(HERE, "core_notebook_original.csv")


def ensure_wordnet():
    """Use the local WordNet corpus; download it only when it is not installed."""
    for res in ("corpora/wordnet", "corpora/wordnet.zip"):
        try:
            nltk.data.find(res)
            return
        except LookupError:
            pass
    nltk.download("wordnet", quiet=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", default=HERE, help="where to write the outputs (default: this folder)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ensure_wordnet()
    wnl = WordNetLemmatizer()
    aux = {"be", "have", "do"}

    with open(CORE_AT_NORMALIZATION, encoding="utf-8") as f:
        core = {row["verb"].strip().lower() for row in csv.DictReader(f)}
    assert len(core) == 358, f"expected the 358-verb original core, got {len(core)}"

    with open(RAW_CANDIDATES, encoding="utf-8") as f:
        candidates = [line.strip().lower() for line in f if line.strip()]
    assert len(candidates) == 1824, f"expected 1,824 raw candidates, got {len(candidates)}"

    pool = set(candidates)

    def american_variant(w):
        """Return the American form if mapping is safe, else None."""
        subs = []
        if w.endswith("ise"):
            subs.append(w[:-3] + "ize")
        if "yse" in w:
            subs.append(w.replace("yse", "yze"))
        if "ise" in w and not w.endswith("ise"):
            subs.append(w.replace("ise", "ize"))
        for am in subs:
            if am == w:
                continue
            # Only merge when it resolves a real duplicate; never invent variants.
            if am in core or am in pool:
                return am
        return None

    audit = []
    cleaned = {}
    for v in candidates:
        lemma = wnl.lemmatize(v, "v").lower()
        action = "kept" if lemma == v else "lemmatized"
        am = american_variant(lemma)
        if am:
            lemma, action = am, action + "+spelling"
        if lemma in aux:
            audit.append((v, lemma, "dropped-auxiliary", ""))
            continue
        if lemma in core:
            audit.append((v, lemma, "dropped-core-leak", lemma))
            continue
        if lemma in cleaned:
            audit.append((v, lemma, "merged-duplicate", cleaned[lemma]))
            continue
        cleaned[lemma] = v
        audit.append((v, lemma, action, ""))

    with open(os.path.join(args.outdir, "cleaned_candidates.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(cleaned)))
    with open(os.path.join(args.outdir, "normalization_audit.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original", "normalized_lemma", "action", "merged_into_or_core"])
        w.writerows(audit)

    c = Counter(a[2] for a in audit)
    print("input candidates:", len(candidates))
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")
    print("cleaned candidate lemmas:", len(cleaned))
    changed = [a for a in audit if a[2] != "kept"]
    print("sample changes:", changed[:12])
    print("written to", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
