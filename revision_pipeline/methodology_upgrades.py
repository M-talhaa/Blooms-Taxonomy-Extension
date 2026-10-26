"""methodology_upgrades.py - ROUND-2 held-out full-pipeline protocol, kept for traceability.

STATUS IN ROUND 3
  This is the round-2 DEFAULT-CONFIGURATION protocol: five families at one fixed configuration
  each (MODELS below), all at P50, one outer split (MultilabelStratifiedKFold seed 13), plus a
  three-member ensemble. For model selection it is superseded by supervised_eval.py (the same
  held-out protocol for every configuration of the model zoo at every percentile, outer seeds 13
  to 17, matched to each family's structural best; outputs in results/selection/). It is kept
  because round-2 Table 11 and the round-2 two-stage selection read its output; the round-2
  reference of select_model.py asserts that its exact seed-13 values round to heldout_results.csv.

  Outputs (--out-dir, default revision_pipeline/):
    methodology_upgrades.md   the round-2 table
    heldout_results.csv       the same numbers as a table, rounded as printed (3 decimals; mean
                              ordinal distance 2 decimals). Before round 3 this file had no
                              producing script in the repository: the committed copy (written
                              2026-08-13 20:21, between methup_v4.log and methup_v5.log) had a
                              different ensemble row (gate 0.911, within-one 0.778) than
                              methodology_upgrades.md and no LinearSVC-C1 row. This script now
                              writes it; the four single-model rows read by the round-2 code
                              (kNN-21, LogReg-C1, NearestCentroid, CalibratedSVC) are unchanged.
  Row weights: the weighted one-vs-rest fit is attempted as in retrain.py; scikit-learn 1.9.0
  rejects sample_weight and the fit falls back to the unweighted fit, so the row weights reach
  only the threshold percentiles.

  USAGE (from the repository root):  python -I -B revision_pipeline/methodology_upgrades.py

Three candidate methodology upgrades, tested with zero new human data.

A. Held-out FULL-PIPELINE evaluation. The existing CV scores the classifier, but the
   delivered bank is produced by classifier + calibration + thresholds + top-2 margin
   collapse, and the accept/reject gate has never been tested on labeled data. Here,
   held-out core verbs are pushed through the complete production pipeline as if they
   were unseen candidates. Every one of them is a known Bloom verb, so gate recall
   (how many survive) and level accuracy are both measurable.

B. Ordinal-aware error metrics. Bloom levels are ordered; predicting Comprehension for
   a Knowledge verb is a smaller error than predicting Evaluation. Reports within-one-
   level accuracy and mean ordinal distance of the primary prediction.

C. Score-averaging ensemble of the top families (kNN-21, LogisticRegression,
   NearestCentroid). If their errors decorrelate, the ensemble should beat each member
   on CV while keeping structure; tested on both axes.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)   # python -I does not put the script directory on sys.path
from retrain import _fit_cdf_per_class, _apply_cdf_per_class, thresholds_from_percentile_weighted

_ap = argparse.ArgumentParser(description="Round-2 held-out full-pipeline protocol (kept for traceability).")
_ap.add_argument("--out-dir", default=HERE, help="folder for methodology_upgrades.md and heldout_results.csv")
ARGS = _ap.parse_args()
LEVELS = ["Kn", "Cm", "Ap", "An", "Sn", "Ev"]
TOP2_MARGIN = 0.060
np.random.seed(13)
out = []
def w(s=""):
    out.append(s); print(s)

# ---- data ----
core = pd.read_csv(os.path.join(HERE, "core_merged.csv"))
verbs = sorted(core["verb"].unique())
v2i = {v: i for i, v in enumerate(verbs)}
Y = np.zeros((len(verbs), 6), dtype=int)
freq = np.zeros(len(verbs))
for _, r in core.iterrows():
    Y[v2i[r["verb"]], LEVELS.index(r["level"])] = 1
    freq[v2i[r["verb"]]] = max(freq[v2i[r["verb"]]], float(r["freq"]))
wts = 1.0 + np.log1p(freq)
wts = np.clip(wts, 0.5, 3.0); wts /= wts.mean()
z = np.load(os.path.join(HERE, "all_embeddings.npz"))
vec = dict(zip([str(x) for x in z["words"]], z["vectors"]))
X = np.array([vec[v] for v in verbs], dtype=float)

MODELS = {
    "CalibratedSVC": lambda: CalibratedClassifierCV(SVC(kernel="linear", C=0.5,
                                 class_weight="balanced"), method="sigmoid", cv=3),
    "LinearSVC-C1": lambda: LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, tol=1e-4, random_state=13),
    "kNN-21": lambda: KNeighborsClassifier(n_neighbors=21, metric="cosine", weights="distance", algorithm="brute"),
    "LogReg-C1": lambda: LogisticRegression(C=1.0, solver="liblinear", penalty="l2",
                                            class_weight="balanced", max_iter=4000, random_state=13),
    "NearestCentroid": lambda: NearestCentroid(metric="euclidean"),
}

def scores(clf, Xq):
    s = clf.decision_function(Xq) if hasattr(clf, "decision_function") else clf.predict_proba(Xq)
    return np.asarray(s, dtype=float)

def run_fold(tr, te, members):
    """Full production pipeline per member: inner-OOF calibration, P50 thresholds,
    hard labels, top-2 collapse. Returns per-member calibrated test scores and
    collapsed hard labels."""
    from sklearn.model_selection import KFold
    te_scores, te_hard = {}, {}
    for name, mk in members.items():
        S_oof = np.zeros((len(tr), 6))
        for itr, iva in KFold(5, shuffle=True, random_state=13).split(X[tr]):
            iclf = OneVsRestClassifier(mk())
            try: iclf.fit(X[tr][itr], Y[tr][itr], sample_weight=wts[tr][itr])
            except (TypeError, ValueError): iclf.fit(X[tr][itr], Y[tr][itr])
            S_oof[iva] = scores(iclf, X[tr][iva])
        clf = OneVsRestClassifier(mk())
        try: clf.fit(X[tr], Y[tr], sample_weight=wts[tr])
        except (TypeError, ValueError): clf.fit(X[tr], Y[tr])
        cdfs = _fit_cdf_per_class(S_oof)
        S_tr01 = _apply_cdf_per_class(S_oof, cdfs)
        S_te01 = _apply_cdf_per_class(scores(clf, X[te]), cdfs)
        thr = thresholds_from_percentile_weighted(S_tr01, Y[tr], 50, wts[tr])
        hard = (S_te01 >= thr.reshape(1, -1)).astype(int)
        # top-2 margin collapse, as in Algorithm 2
        for i in range(len(hard)):
            on = np.where(hard[i] == 1)[0]
            if len(on) >= 2:
                s_on = S_te01[i, on]
                o = np.argsort(s_on)[::-1]
                if s_on[o[0]] - s_on[o[1]] >= TOP2_MARGIN:
                    hard[i] = 0; hard[i, on[o[0]]] = 1
        te_scores[name], te_hard[name] = S_te01, hard
    return te_scores, te_hard

def evaluate(tag, S01, hard):
    accepted = hard.sum(1) > 0
    gate_recall = accepted.mean()
    prim = np.where(accepted, np.argmax(S01 * hard, axis=1), -1)
    idx = np.where(accepted)[0]
    prim_in = np.mean([Y_te[i, prim[i]] == 1 for i in idx]) if len(idx) else 0.0
    true_ranks = [np.where(Y_te[i] == 1)[0] for i in idx]
    dists = [min(abs(prim[i] - t) for t in true_ranks[j]) for j, i in enumerate(idx)]
    within1 = np.mean([d <= 1 for d in dists]) if dists else 0.0
    mdist = np.mean(dists) if dists else np.nan
    f1 = f1_score(Y_te, hard, average="macro", zero_division=0)
    try:
        auc = float(np.mean([roc_auc_score(Y_te[:, j], S01[:, j]) for j in range(6)]))
    except ValueError:
        auc = np.nan
    return dict(tag=tag, gate=gate_recall, prim=prim_in, within1=within1,
                mdist=mdist, f1=f1, auc=auc)

recs = []
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=13)
for tr, te in mskf.split(X, Y):
    Y_te = Y[te]
    S, H = run_fold(tr, te, MODELS)
    for name in MODELS:
        recs.append(evaluate(name, S[name], H[name]))
    ENS = ["kNN-21", "LogReg-C1", "NearestCentroid"]
    ens = np.mean([S[n] for n in ENS], axis=0)
    # ensemble hard labels: threshold at per-level P50 of ensemble train scores is not
    # available here, so use the members' vote: level on if >=2 of 3 members set it,
    # then the same top-2 collapse on the averaged scores.
    vote = (np.sum([H[n] for n in ENS], axis=0) >= 2).astype(int)
    for i in range(len(vote)):
        on = np.where(vote[i] == 1)[0]
        if len(on) >= 2:
            s_on = ens[i, on]; o = np.argsort(s_on)[::-1]
            if s_on[o[0]] - s_on[o[1]] >= TOP2_MARGIN:
                vote[i] = 0; vote[i, on[o[0]]] = 1
    recs.append(evaluate("Ensemble(3, vote>=2)", ens, vote))

df = pd.DataFrame(recs).groupby("tag").mean(numeric_only=True)

w("# Methodology upgrade tests")
w()
w("## A + B. Held-out full-pipeline evaluation with ordinal metrics")
w()
w("Held-out core verbs pushed through the COMPLETE production pipeline (calibration,")
w("P50 thresholds, top-2 collapse). All are known Bloom verbs, so the gate should")
w("accept them; 5-fold multi-label stratified, means over folds:")
w()
w("| Pipeline | Gate recall | Primary-in-labels | Within-1-level | Mean ordinal dist | macro F1 | macro AUC |")
w("|---|---|---|---|---|---|---|")
for tag, r in df.iterrows():
    w(f"| {tag} | {r['gate']:.3f} | {r['prim']:.3f} | {r['within1']:.3f} | "
      f"{r['mdist']:.2f} | {r['f1']:.3f} | {r['auc']:.3f} |")
w()
w("Reading guide: 'Gate recall' is the share of held-out KNOWN Bloom verbs that survive")
w("the acceptance gate; this is the first labeled test of the gate. 'Primary-in-labels'")
w("scores the primary domain against the verb's published level set. 'Within-1-level'")
w("credits adjacent-level predictions, reflecting the ordinal nature of the taxonomy.")

os.makedirs(ARGS.out_dir, exist_ok=True)
open(os.path.join(ARGS.out_dir, "methodology_upgrades.md"), "w", encoding="utf-8").write("\n".join(out))
print("\nsaved:", os.path.join(ARGS.out_dir, "methodology_upgrades.md"))

# heldout_results.csv: the table above, rounded exactly as printed (read by the round-2 selection
# code and by select_model.v2_reference, which checks it against the unrounded seed-13 values).
ho = pd.DataFrame({
    "model": df.index.astype(str),
    "gate_recall": [f"{v:.3f}" for v in df["gate"]],
    "primary_in_labels": [f"{v:.3f}" for v in df["prim"]],
    "within1": [f"{v:.3f}" for v in df["within1"]],
    "mean_ordinal_dist": [f"{v:.2f}" for v in df["mdist"]],
    "macro_F1": [f"{v:.3f}" for v in df["f1"]],
    "macro_AUC": [f"{v:.3f}" for v in df["auc"]],
})
ho.to_csv(os.path.join(ARGS.out_dir, "heldout_results.csv"), index=False)
print("saved:", os.path.join(ARGS.out_dir, "heldout_results.csv"))
