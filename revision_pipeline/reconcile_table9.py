"""Reconcile the two transcription passes of Sharunova et al. (2019) appendix Table 9 and
compare them with the original submission's core (core_notebook_original.csv, 358 verbs /
559 assignments).

Input: table9_pass1.csv and table9_pass2.csv (verb, level, freq). Both passes were produced
by AI-assisted transcription and the two files are byte-identical, so report A below can
only confirm that they agree; it is not an independent check against the printed table.

Reports:
  A. Disagreements between the two transcription passes (any would need to be
     resolved by hand against the printed table before anything downstream is used).
  B. Entries in the published table missing from the notebook core, split into
     multi-word phrases (missing from the original notebook core; all 12 are kept
     in the final core, see merge_core.py) and single-word verbs (real omissions).
  C. Entries in the notebook not present in the published table.
  D. Frequency mismatches.
"""
import csv, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CROSS = {"Knowledge": "Kn", "Comprehension": "Cm", "Application": "Ap",
         "Analysis": "An", "Synthesis": "Sn", "Evaluation": "Ev"}

def load(path, cross=True):
    d = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lv = r["level"].strip()
            lv = CROSS.get(lv, lv)
            d[(r["verb"].strip().lower(), lv)] = float(r["freq"])
    return d

p1 = load(os.path.join(HERE, "table9_pass1.csv"))
p2 = load(os.path.join(HERE, "table9_pass2.csv"))
nb = load(os.path.join(HERE, "core_notebook_original.csv"))

print("=" * 70)
print("A. TRANSCRIPTION AGREEMENT (pass 1 vs pass 2)")
print("=" * 70)
only1 = sorted(set(p1) - set(p2))
only2 = sorted(set(p2) - set(p1))
freqdiff = sorted(k for k in set(p1) & set(p2) if p1[k] != p2[k])
print(f"pass1: {len(p1)} entries | pass2: {len(p2)} entries")
print(f"only in pass1 ({len(only1)}): {only1[:30]}")
print(f"only in pass2 ({len(only2)}): {only2[:30]}")
print(f"frequency disagreements ({len(freqdiff)}):")
for k in freqdiff[:30]:
    print(f"   {k}: pass1={p1[k]:g} pass2={p2[k]:g}")
agreed = {k: p1[k] for k in set(p1) & set(p2) if p1[k] == p2[k]}
print(f"\nAGREED ENTRIES: {len(agreed)} (used as the authoritative table below)")
if only1 or only2 or freqdiff:
    print("The disagreements above need to be resolved by hand against the printed table.")

print()
print("=" * 70)
print("B. PUBLISHED TABLE vs NOTEBOOK CORE")
print("=" * 70)
per_level_pub = defaultdict(int)
per_level_nb = defaultdict(int)
for (v, lv) in agreed:
    per_level_pub[lv] += 1
for (v, lv) in nb:
    per_level_nb[lv] += 1
print(f"{'level':6} {'published':>10} {'notebook':>9} {'diff':>6}")
for lv in ["Kn", "Cm", "Ap", "An", "Sn", "Ev"]:
    print(f"{lv:6} {per_level_pub[lv]:>10} {per_level_nb[lv]:>9} {per_level_pub[lv]-per_level_nb[lv]:>+6}")
print(f"{'TOTAL':6} {len(agreed):>10} {len(nb):>9} {len(agreed)-len(nb):>+6}")

missing = sorted(set(agreed) - set(nb))
multi = [k for k in missing if " " in k[0]]
single = [k for k in missing if " " not in k[0]]
extra = sorted(set(nb) - set(agreed))
fdiff = sorted(k for k in set(agreed) & set(nb) if agreed[k] != nb[k])

print(f"\nMISSING from notebook: {len(missing)} assignments")
print(f"  multi-word phrases ({len(multi)}): {[f'{v} [{l}] f={agreed[(v,l)]:g}' for v, l in multi]}")
print(f"  SINGLE-WORD verbs ({len(single)}):")
for v, l in single:
    print(f"    {v} [{l}] freq={agreed[(v, l)]:g}")

print(f"\nIn NOTEBOOK but not in published table ({len(extra)}):")
for v, l in extra[:40]:
    print(f"    {v} [{l}] freq={nb[(v, l)]:g}")

print(f"\nFREQUENCY MISMATCHES ({len(fdiff)}):")
for v, l in fdiff[:40]:
    print(f"    {v} [{l}]: published={agreed[(v, l)]:g} notebook={nb[(v, l)]:g}")

pub_verbs = {v for v, _ in agreed}
nb_verbs = {v for v, _ in nb}
print(f"\nUnique verbs: published {len(pub_verbs)}, notebook {len(nb_verbs)}")
print(f"Verbs entirely absent from notebook: {len(pub_verbs - nb_verbs)}")
single_absent = sorted(v for v in pub_verbs - nb_verbs if " " not in v)
print(f"  single-word ({len(single_absent)}): {single_absent}")

verdict = "FAITHFUL" if not (missing or extra or fdiff) else "DISCREPANCIES FOUND"
print(f"\nVERDICT: {verdict}")
