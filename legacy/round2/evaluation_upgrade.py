"""Three upgrades to the evaluation design.

1. Rejection-aware expert evaluation. The previous analysis scored only verbs the
   model accepted, which throws away the model's rejections. But experts could vote
   "None of the above", so a rejection is a testable decision too. This scores all
   rated verbs on a 2x2 of model decision against expert judgement.

2. Matched-acceptance structural comparison. The structural objective contains an
   acceptance-band penalty, and acceptance is controlled by the threshold percentile,
   not by model quality. So the objective may be ranking models by how close their
   chosen threshold lands to the target rather than by structural coherence. This
   re-compares every family at a COMMON acceptance level and re-tests the correlation
   with cross-validated generalization.

3. Independent-encoder structural check. The structural metrics are computed in the
   same embedding space the classifier uses, so they partly validate the geometry
   rather than the assignments. This recomputes the ordinal-separation trend in a
   different encoder that played no part in training or selection.
"""
import csv, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LEVELS = ["Kn", "Cm", "Ap", "An", "Sn", "Ev"]
RES = os.path.join(REPO, "results")
np.random.seed(13)
out = []
def w(s=""):
    out.append(s)
    print(s)

# ----------------------------------------------------------------------
w("# Evaluation upgrades")
w()
votes = pd.read_excel(os.path.join(RES, "Final File(with model & Expert Classification).xlsx"))
votes["clean"] = votes["Verb_clean"].astype(str).str.lower().str.strip()
votes = votes[["clean"] + [f"cnt_{l}" for l in LEVELS] + ["cnt_None"]]
bank = pd.read_csv(os.path.join(RES, "Verb List Classified by Model.csv"))
bank["clean"] = bank["Verb"].astype(str).str.lower().str.strip()
m = votes.merge(bank, on="clean", how="inner")

m["E25"] = m.apply(lambda r: {l for l in LEVELS if r[f"cnt_{l}"] >= 2}, axis=1)
m["top_vote"] = m[[f"cnt_{l}" for l in LEVELS]].max(axis=1)
m["none_dominant"] = m["cnt_None"] > m["top_vote"]
m["none_heavy"] = m["cnt_None"] >= 3
m["accepts"] = m["Accepted"] == 1

w("## 1. Rejection-aware expert evaluation")
w()
w(f"Rated verbs present in the bank: {len(m)} (accepted {int(m['accepts'].sum())}, "
  f"rejected {int((~m['accepts']).sum())}).")
w()
w("A rejection is a decision the experts can also judge, because they could vote")
w("'None of the above'. Scoring only accepted verbs discards that evidence.")
w()

# 2x2 using "experts consider it out of domain" = None is the plurality response
tab = pd.crosstab(m["accepts"], m["none_dominant"])
w("| | experts: None is plurality | experts: a level leads |")
w("|---|---|---|")
for acc_val, label in [(True, "model accepts"), (False, "model rejects")]:
    r = [int(tab.loc[acc_val, c]) if (acc_val in tab.index and c in tab.columns) else 0
         for c in [True, False]]
    w(f"| {label} | {r[0]} | {r[1]} |")
a = int(((~m["accepts"]) & m["none_dominant"]).sum())      # correct rejection
b = int(((~m["accepts"]) & ~m["none_dominant"]).sum())     # false rejection
c_ = int((m["accepts"] & m["none_dominant"]).sum())        # false acceptance
d = int((m["accepts"] & ~m["none_dominant"]).sum())        # correct acceptance
w()
w(f"- Correct acceptances (model assigns, experts back a level): {d}")
w(f"- Correct rejections   (model declines, experts say None):   {a}")
w(f"- False acceptances    (model assigns, experts say None):    {c_}")
w(f"- False rejections     (model declines, experts back a level): {b}")
acc_rate = (a + d) / len(m)
w(f"- **Gate accuracy (accept/reject decision): {a + d}/{len(m)} = {acc_rate:.1%}**")
if min(a + b, c_ + d) > 0 and min(a + c_, b + d) > 0:
    p = stats.fisher_exact([[d, c_], [b, a]]).pvalue
    w(f"- Fisher exact on the 2x2: p = {p:.4f}")
w()
w("None-vote rates by model decision:")
w(f"- accepted verbs: mean {m[m['accepts']]['cnt_None'].mean():.2f}/8")
w(f"- rejected verbs: mean {m[~m['accepts']]['cnt_None'].mean():.2f}/8")
if (~m["accepts"]).sum() > 0:
    u = stats.mannwhitneyu(m[~m["accepts"]]["cnt_None"], m[m["accepts"]]["cnt_None"],
                           alternative="greater")
    w(f"- Mann-Whitney (rejected have more None votes): p = {u.pvalue:.4f}")
w()
w("Rejected verbs in detail:")
for _, r in m[~m["accepts"]].iterrows():
    w(f"  {r['clean']:14} None {int(r['cnt_None'])}/8, top level vote {int(r['top_vote'])}/8, "
      f"expert-endorsed levels {sorted(r['E25']) if r['E25'] else 'none'}")
w()

# ----------------------------------------------------------------------
w("## 2. Matched-acceptance structural comparison")
w()
scan = pd.read_csv(os.path.join(RES, "structural_scan_components.csv"))
scan = scan[scan["PassedMinAccept"] != False]
cv = pd.read_csv(os.path.join(HERE, "cv_results_mpnet.csv")).groupby("model")[["macro_AUC"]].mean()
CVMAP = {"SGDClassifier": "SGD-hinge-1e4", "LogisticRegression": "LogReg-C1",
         "RidgeClassifier": "Ridge-a1", "LinearSVC": "LinearSVC-C1",
         "KNeighborsClassifier": "kNN-21", "NearestCentroid": "NearestCentroid",
         "PassiveAggressive": "PassiveAggressive-C1"}

w("The structural objective penalizes deviation from a target acceptance band, but")
w("acceptance is set by the threshold percentile, not by model quality. Comparing")
w("families at a COMMON acceptance removes that confound.")
w()
for target in (0.80, 0.86):
    rows = []
    for fam, g in scan.groupby("Model"):
        if fam not in CVMAP:
            continue
        g = g.dropna(subset=["Eval_acceptance", "SCS_eval"])
        if g.empty:
            continue
        pick = g.iloc[(g["Eval_acceptance"] - target).abs().argsort().iloc[0]]
        rows.append({"family": fam, "P": int(pick["Percentile"]),
                     "acceptance": round(pick["Eval_acceptance"], 3),
                     "SCS": round(pick["SCS_eval"], 3),
                     "trend": pick["Eval_rho"] if "Eval_rho" in pick else np.nan,
                     "CV_AUC": round(float(cv.loc[CVMAP[fam], "macro_AUC"]), 3)})
    d_ = pd.DataFrame(rows).sort_values("SCS", ascending=False)
    r_s = stats.spearmanr(d_["SCS"], d_["CV_AUC"])
    w(f"### At acceptance ~{target:.2f}")
    w()
    w("| family | percentile | acceptance | SCS | CV AUC |")
    w("|---|---|---|---|---|")
    for _, r in d_.iterrows():
        w(f"| {r['family']} | P{r['P']} | {r['acceptance']:.3f} | {r['SCS']:.3f} | {r['CV_AUC']:.3f} |")
    w()
    w(f"Spearman(SCS at matched acceptance, CV AUC) = {r_s.statistic:+.3f}, p = {r_s.pvalue:.3f}")
    w()

raw = []
for fam, g in scan.groupby("Model"):
    if fam not in CVMAP:
        continue
    b_ = g.loc[g["PenalizedObj_eval"].idxmax()]
    raw.append({"family": fam, "obj": b_["PenalizedObj_eval"],
                "CV_AUC": float(cv.loc[CVMAP[fam], "macro_AUC"])})
rd = pd.DataFrame(raw)
r0 = stats.spearmanr(rd["obj"], rd["CV_AUC"])
w(f"For comparison, the penalized objective (with the acceptance penalty) against CV AUC: "
  f"Spearman = {r0.statistic:+.3f}, p = {r0.pvalue:.3f}")
w()

# ----------------------------------------------------------------------
w("## 3. Independent-encoder structural check")
w()
w("The structural metrics are computed in the same embedding space the classifier")
w("uses, so they partly measure that geometry rather than the assignments. Recomputing")
w("the ordinal-separation trend in an encoder that played no part in training or")
w("selection gives evidence that is not circular.")
w()
acc_bank = bank[bank["Accepted"] == 1].copy()
verbs = acc_bank["Verb"].astype(str).str.lower().tolist()
assign = {lv: acc_bank[acc_bank[f"accept_{lv}"] == 1]["Verb"].astype(str).str.lower().tolist()
          for lv in LEVELS}

def gap_trend(vec_of):
    cents = {}
    for lv in LEVELS:
        V = np.array([vec_of[v] for v in assign[lv] if v in vec_of])
        if len(V) == 0:
            return None
        c = V.mean(axis=0)
        cents[lv] = c / (np.linalg.norm(c) + 1e-12)
    D = np.zeros((6, 6))
    for i, a_ in enumerate(LEVELS):
        for j, b_ in enumerate(LEVELS):
            D[i, j] = 1 - float(cents[a_] @ cents[b_])
    dk = [np.mean([D[i, i + k] for i in range(6 - k)]) for k in range(1, 6)]
    rho = stats.spearmanr(range(1, 6), dk).statistic
    return dk, rho

from sentence_transformers import SentenceTransformer
for name, tag in [("sentence-transformers/all-mpnet-base-v2", "mpnet (used in training)"),
                  ("sentence-transformers/all-MiniLM-L6-v2", "MiniLM (independent)"),
                  ("sentence-transformers/all-distilroberta-v1", "distilroberta (independent)")]:
    enc = SentenceTransformer(name)
    V = enc.encode(verbs, normalize_embeddings=True, convert_to_numpy=True,
                   show_progress_bar=False, batch_size=64)
    vec = dict(zip(verbs, V))
    dk, rho = gap_trend(vec)
    exact_p = sum(1 for perm in __import__("itertools").permutations(range(1, 6))
                  if stats.spearmanr(range(1, 6), perm).statistic >= rho - 1e-12) / 120
    w(f"- {tag}: gap means {[round(x,4) for x in dk]}, Spearman rho = {rho:.2f}, "
      f"exact one-tailed p = {exact_p:.4f}")
w()
w("If the monotone trend survives in encoders that were never used for training or")
w("model selection, the ordinal structure of the extended bank is a property of the")
w("assignments rather than an artifact of the embedding space used to produce them.")

open(os.path.join(HERE, "evaluation_upgrade.md"), "w", encoding="utf-8").write("\n".join(out))
print("\nsaved:", os.path.join(HERE, "evaluation_upgrade.md"))
