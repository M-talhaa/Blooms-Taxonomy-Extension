"""Structure of the delivered verb bank (paper Section 3.2, Table 9 and the gap-trend tables).

Computes, for the delivered bank results/Verb List Classified by Model.csv:
  1. coverage and cardinality: acceptance, entries per level, verbs by number of levels,
     primary-domain distribution, and BOTH density ratios (entries per input verb and entries per
     accepted verb). The original-submission bank (git commit 2a87a33) is summarized the same way
     when the git history is available, so the two versions can be compared like for like.
  2. the extended-bank 6 x 6 cosine distance matrix between level centroids (Eq. 5 and 6: centroid =
     L2-normalized mean of the L2-normalized all-mpnet-base-v2 embeddings of the verbs holding a
     hard entry at that level; distance = 1 - cosine), the mean distance d_n at taxonomic gap n
     (Eq. 7), next to the same gap means of the 381-verb core (all, high-frequency freq >= 10,
     low-frequency freq < 10).
  3. the gap trend: Spearman rho of d_n against n = 1..5 with the EXACT one-tailed p (share of the
     120 orderings of five values whose rho is at least the observed one), for the delivered
     (post-collapse) labeling and for the pre-collapse labeling (cal_score >= threshold).
  4. the same gap trend in two encoders that played no part in training or selection
     (all-MiniLM-L6-v2, all-distilroberta-v1), loaded from the local Hugging Face cache with
     HF_HUB_OFFLINE=1 (nothing is downloaded). The training encoder all-mpnet-base-v2 is read from
     the stored embeddings and, if cached, re-encoded as a check.

Outputs: results/bank_structure/ (bank_structure.md and CSV tables). Read-only on every input.
Run: python -I -B revision_pipeline/bank_structure.py
"""
import io
import os
import subprocess
from itertools import permutations

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LEVELS = ["Kn", "Cm", "Ap", "An", "Sn", "Ev"]
NAMES = dict(Kn="Knowledge", Cm="Comprehension", Ap="Application", An="Analysis", Sn="Synthesis", Ev="Evaluation")
BANK = os.path.join(REPO, "results", "Verb List Classified by Model.csv")
CORE = os.path.join(HERE, "core_merged.csv")
EMB = os.path.join(HERE, "all_embeddings.npz")
V1_COMMIT = "2a87a33"
V1_PATH = "results/Verb List Classified by Model.csv"
V1_COPY = os.path.join(REPO, "legacy", "original_submission_bank_v1.csv")
FREQ_SPLIT = 10
OUT = os.path.join(REPO, "results", "bank_structure")
os.makedirs(OUT, exist_ok=True)
REPORT = []


def w(s=""):
    REPORT.append(s)


def truthy(x):
    return x.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


# ---------------------------------------------------------------- 1. coverage and cardinality


def bank_summary(b):
    acc = truthy(b["Accepted"])
    H = b[[f"accept_{l}" for l in LEVELS]].to_numpy(int)
    n_lab = H.sum(1)
    assert (acc.to_numpy() == (n_lab > 0)).all()
    prim = b["PrimaryDomain"].where(acc, "None").fillna("None")
    entries = int(H.sum())
    return dict(inputs=len(b), accepted=int(acc.sum()), unassigned=int((~acc).sum()), entries=entries,
                acceptance=acc.mean(), entries_per_input=entries / len(b), entries_per_accepted=entries / acc.sum(),
                per_level={l: int(H[:, j].sum()) for j, l in enumerate(LEVELS)},
                cardinality={k: int((n_lab == k).sum()) for k in range(7)},
                primary={l: int((prim == l).sum()) for l in LEVELS + ["None"]})


bank = pd.read_csv(BANK)
bank["verb"] = bank["Verb"].astype(str).str.strip().str.lower()
assert len(bank) == 1650
S2 = bank_summary(bank)

v1 = None
try:
    # The committed copy is a byte copy of results/Verb List Classified by Model.csv at commit
    # 2a87a33, so an archive download without git history still reproduces the v1 column.
    if os.path.exists(V1_COPY):
        with open(V1_COPY, "rb") as fh:
            raw = fh.read()
    else:
        raw = subprocess.run(["git", "-C", REPO, "show", f"{V1_COMMIT}:{V1_PATH}"],
                             capture_output=True, check=True).stdout
    v1 = pd.read_csv(io.BytesIO(raw))
    S1 = bank_summary(v1)
except Exception as exc:  # neither the committed copy nor git history is available
    S1 = None
    v1_err = f"{type(exc).__name__}"

rows = []
for tag, S in (("v2 delivered bank", S2), (f"v1 bank (commit {V1_COMMIT})", S1)):
    if S is None:
        continue
    r = dict(bank=tag, inputs=S["inputs"], accepted=S["accepted"], unassigned=S["unassigned"], entries=S["entries"],
             acceptance=S["acceptance"], entries_per_input=S["entries_per_input"],
             entries_per_accepted=S["entries_per_accepted"])
    r.update({f"entries_{l}": S["per_level"][l] for l in LEVELS})
    r.update({f"verbs_with_{k}_levels": S["cardinality"][k] for k in range(7)})
    r.update({f"primary_{l}": S["primary"][l] for l in LEVELS + ["None"]})
    rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "coverage_cardinality.csv"), index=False, float_format="%.6f")

# ---------------------------------------------------------------- 2. centroid matrices and gap means
_z = np.load(EMB)
MPNET = {str(a): np.asarray(b, float) for a, b in zip(_z["words"], _z["vectors"])}
core = pd.read_csv(CORE)
core["verb"] = core["verb"].astype(str).str.strip().str.lower()
assert core["verb"].nunique() == 381 and len(core) == 607


def assignments_bank(b, pre_collapse=False):
    col = "meets_thr_{}" if pre_collapse else "accept_{}"
    return {l: b.loc[b[col.format(l)].astype(int) == 1, "verb"].tolist() for l in LEVELS}


def assignments_core(sel=None):
    c = core if sel is None else core[sel(core["freq"].astype(float))]
    return {l: c.loc[c["level"] == l, "verb"].tolist() for l in LEVELS}


def dist_matrix(assign, vec):
    cents = {}
    for l in LEVELS:
        V = np.stack([vec[v] for v in assign[l]])
        m = V.mean(0)
        cents[l] = m / np.linalg.norm(m)
    D = np.array([[1.0 - float(cents[a] @ cents[b]) for b in LEVELS] for a in LEVELS])
    np.fill_diagonal(D, 0.0)
    return D


def gap_means(D):
    return [float(np.mean([D[i, i + n] for i in range(6 - n)])) for n in range(1, 6)]


PERMS = list(permutations(range(5)))


def trend(d):
    """Spearman rho of d_n against n = 1..5 and the exact one-tailed p over the 120 orderings."""
    rho = float(stats.spearmanr(range(1, 6), d).statistic)
    rd = stats.rankdata(d)
    null = [stats.spearmanr(range(1, 6), [rd[i] for i in p]).statistic for p in PERMS]
    return rho, sum(1 for x in null if x >= rho - 1e-12) / len(PERMS)


A_post = assignments_bank(bank)
A_pre = assignments_bank(bank, pre_collapse=True)
A_core = assignments_core()
A_hi = assignments_core(lambda f: f >= FREQ_SPLIT)
A_lo = assignments_core(lambda f: f < FREQ_SPLIT)
LABELINGS = [("Extended bank, delivered (post-collapse)", A_post), ("Extended bank, pre-collapse", A_pre),
             ("Core, all (381 verbs, 607 assignments)", A_core),
             (f"Core, high-frequency (freq >= {FREQ_SPLIT})", A_hi), (f"Core, low-frequency (freq < {FREQ_SPLIT})", A_lo)]
D_bank = dist_matrix(A_post, MPNET)
pd.DataFrame(D_bank, index=LEVELS, columns=LEVELS).to_csv(os.path.join(OUT, "centroid_matrix_extended_bank.csv"),
                                                         float_format="%.6f")
GAP = []
for name, A in LABELINGS:
    d = gap_means(dist_matrix(A, MPNET))
    rho, p = trend(d)
    GAP.append(dict(labeling=name, entries=sum(len(A[l]) for l in LEVELS), **{f"d{n}": d[n - 1] for n in range(1, 6)},
                    spearman_rho=rho, p_exact_one_tailed=p))
gap = pd.DataFrame(GAP)
gap.to_csv(os.path.join(OUT, "gap_means_mpnet.csv"), index=False, float_format="%.6f")

# ---------------------------------------------------------------- 3. encoder trend table
ENCODERS = [("sentence-transformers/all-mpnet-base-v2", "all-mpnet-base-v2 (training encoder)"),
            ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2 (independent)"),
            ("sentence-transformers/all-distilroberta-v1", "all-distilroberta-v1 (independent)")]
needed = sorted({v for A in (A_post, A_pre, A_core) for l in LEVELS for v in A[l]})
ENC = []
enc_notes = []
try:
    from sentence_transformers import SentenceTransformer
    for name, label in ENCODERS:
        model = SentenceTransformer(name, device="cpu")
        E = model.encode(needed, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
        vec = dict(zip(needed, np.asarray(E, float)))
        if "mpnet" in name:
            diff = max(float(np.abs(vec[v] - MPNET[v]).max()) for v in needed)
            enc_notes.append(f"all-mpnet-base-v2 re-encoded from the local cache matches the stored embeddings to "
                             f"{diff:.1e} (max abs difference over {len(needed)} verbs); the table uses the stored embeddings.")
            vec = MPNET
        r = dict(encoder=label)
        for tag, A in (("delivered", A_post), ("pre_collapse", A_pre), ("core", A_core)):
            d = gap_means(dist_matrix(A, vec))
            rho, p = trend(d)
            r.update({f"{tag}_rho": rho, f"{tag}_p_exact": p, f"{tag}_gap_means": " ".join(f"{x:.4f}" for x in d)})
        ENC.append(r)
except Exception as exc:
    enc_notes.append(f"Encoder trend table skipped: {type(exc).__name__}: {str(exc)[:160]}")
if ENC:
    pd.DataFrame(ENC).to_csv(os.path.join(OUT, "encoder_trend.csv"), index=False, float_format="%.6f")

# ---------------------------------------------------------------- report


def fmt_int(x):
    return f"{x:,}"


w("# Structure of the delivered verb bank")
w()
w("Generated by `revision_pipeline/bank_structure.py`. Do not edit by hand; rerun the script.")
w()
w("## 1. Coverage, cardinality and density (Table 9)")
w()
cols = [("v2", S2)] + ([("v1", S1)] if S1 else [])
w("| Quantity | " + " | ".join("v2 delivered bank" if t == "v2" else f"v1 bank (commit {V1_COMMIT})" for t, _ in cols) + " |")
w("|---|" + "---|" * len(cols))
w("| Input verbs | " + " | ".join(fmt_int(S["inputs"]) for _, S in cols) + " |")
w("| Accepted (at least one level) | " + " | ".join(fmt_int(S["accepted"]) for _, S in cols) + " |")
w("| Unassigned | " + " | ".join(fmt_int(S["unassigned"]) for _, S in cols) + " |")
w("| Acceptance rate | " + " | ".join(f"{S['acceptance']:.3f}" for _, S in cols) + " |")
w("| Verb-level entries | " + " | ".join(fmt_int(S["entries"]) for _, S in cols) + " |")
w("| Entries per input verb | " + " | ".join(f"{S['entries_per_input']:.3f}" for _, S in cols) + " |")
w("| Entries per accepted verb | " + " | ".join(f"{S['entries_per_accepted']:.3f}" for _, S in cols) + " |")
for l in LEVELS:
    w(f"| Entries, {NAMES[l]} ({l}) | " + " | ".join(fmt_int(S["per_level"][l]) for _, S in cols) + " |")
for k in range(7):
    w(f"| Verbs with {k} levels | " + " | ".join(fmt_int(S["cardinality"][k]) for _, S in cols) + " |")
for l in LEVELS + ["None"]:
    lab = "Primary domain: none (unassigned)" if l == "None" else f"Primary domain: {NAMES[l]} ({l})"
    w(f"| {lab} | " + " | ".join(fmt_int(S["primary"][l]) for _, S in cols) + " |")
w()
if S1 is None:
    w(f"The v1 bank could not be read from git commit {V1_COMMIT} ({v1_err}); only the v2 column is shown.")
else:
    w(f"The v1 manuscript (Table 6) reported entries per input verb ({S1['entries_per_input']:.3f}); the v2 manuscript "
      f"reported entries per accepted verb ({S2['entries_per_accepted']:.2f}). Like for like, both ratios fell: per input "
      f"{S1['entries_per_input']:.3f} to {S2['entries_per_input']:.3f}, per accepted {S1['entries_per_accepted']:.3f} to "
      f"{S2['entries_per_accepted']:.3f}.")
w()
w("## 2. Extended-bank centroid distances (training encoder, delivered labeling)")
w()
w("Cosine distance between level centroids of the verbs holding a hard entry at each level (Eq. 5 and 6).")
w()
w("| | " + " | ".join(LEVELS) + " |")
w("|---|" + "---|" * 6)
for i, l in enumerate(LEVELS):
    w(f"| {l} | " + " | ".join(f"{D_bank[i, j]:.3f}" for j in range(6)) + " |")
w()
w("Mean centroid distance d_n by taxonomic gap n (Eq. 7), extended bank beside the core, with the Spearman trend of "
  "d_n against n and its exact one-tailed p (120 orderings):")
w()
w("| Labeling | Entries | d1 | d2 | d3 | d4 | d5 | Spearman rho | Exact one-tailed p |")
w("|---|---|---|---|---|---|---|---|---|")
for r in GAP:
    w(f"| {r['labeling']} | {r['entries']:,} | " + " | ".join(f"{r[f'd{n}']:.3f}" for n in range(1, 6))
      + f" | {r['spearman_rho']:+.2f} | {r['p_exact_one_tailed']:.3f} |")
w()
w("## 3. Gap trend across encoders")
w()
for n_ in enc_notes:
    w(f"- {n_}")
w()
if ENC:
    w("| Encoder | Delivered bank rho (exact p) | Pre-collapse rho (exact p) | Core rho (exact p) |")
    w("|---|---|---|---|")
    for r in ENC:
        w(f"| {r['encoder']} | {r['delivered_rho']:+.2f} ({r['delivered_p_exact']:.3f}) | "
          f"{r['pre_collapse_rho']:+.2f} ({r['pre_collapse_p_exact']:.3f}) | {r['core_rho']:+.2f} ({r['core_p_exact']:.3f}) |")
    w()
    w("Gap means per encoder are in `encoder_trend.csv`. With five gap values the smallest attainable one-tailed p is "
      "1/120 = 0.008.")
open(os.path.join(OUT, "bank_structure.md"), "w", encoding="utf-8").write("\n".join(REPORT) + "\n")
print("\n".join(REPORT))
print("written to", OUT)
