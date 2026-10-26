"""cv_metrics.py - ROUND-2 cross-validation protocol, kept for traceability.

STATUS IN ROUND 3
  This is the round-2 DEFAULT-CONFIGURATION protocol: each family is evaluated at one fixed
  configuration (the `zoo` entries in main(), mostly library defaults) and at the scan winner's
  percentile (P50), while the structural objective uses each family's own best configuration and
  percentile. For model selection it is superseded by supervised_eval.py (every configuration of
  the model zoo at every percentile, matched to each family's structural best; outputs in
  results/selection/). It is kept because the round-2 manuscript (Tables 6 and 7) and the round-2
  reference of select_model.py (v2_default_config_reference.csv) were computed from its outputs:
    cv_results_mpnet.csv, cv_results_minilm.csv, cv_results_droberta.csv  (model x repeat rows)
    cv_summary_mpnet.md, cv_summary_minilm.md, cv_summary_droberta.md    (summaries)
  supervised_eval.py reproduces cv_results_mpnet.csv at these configurations to 5.6e-17, except
  SGD-hinge-1e4 (seed 15), SGD-log-1e5 and PassiveAggressive-C1, which react to the 1.6e-7
  difference between one-batch and per-word encodings (results/selection/v2_reproduction_check.csv).
  Rerunning this script reproduces the committed cv_results_<tag>.csv rows exactly; the MiniLM and
  DistilRoBERTa files gained the four families added in round 2 after their first run
  (CalibratedSVC, GaussianNB, LDA, QDA; results/selection/RUN_LOG.md). supervised_eval.py and robustness.py import
  load_core, row_weights, decision_scores, baselines, bootstrap_ci and mcnemar_exact from this
  file, so these functions must stay unchanged.

  Summary reference (--reference): the per-level F1 block and the paired McNemar tests of
  cv_summary_<tag>.md compare every model with the reference configuration, by default kNN-21
  (the production model). Round 2 hard-coded SGD-hinge-1e4, the structural-only winner of an
  interim run, as the "winner".

Protocol: repeated multi-label stratified 5-fold CV (5 repeats, seeds 13..17).
Inside each fold the production decision rule is mirrored: fit OvR on the train fold, fit
per-level empirical CDFs on inner out-of-fold train scores (KFold 5, shuffle, seed 13), set
per-level thresholds at the winner percentile of train-fold positives (row-weighted), then
threshold the held-out fold's calibrated scores (no top-2 collapse). Row weights: the weighted
one-vs-rest fit is attempted as in retrain.py; scikit-learn 1.9.0 rejects sample_weight and the
fit falls back to the unweighted fit, so the row weights reach only the threshold percentiles.

Models: the zoo below (default configurations) plus majority-level and frequency-weighted
random baselines. The verbs are encoded live in one batch with --encoder (production uses
per-word vectors from all_embeddings.npz; the two differ by at most 1.6e-7, which changes only
stochastic-gradient families).

USAGE (from the repository root; encoders are read from the local Hugging Face cache)
  python -I -B revision_pipeline/cv_metrics.py
  python -I -B revision_pipeline/cv_metrics.py --encoder sentence-transformers/all-MiniLM-L6-v2 --tag minilm
  python -I -B revision_pipeline/cv_metrics.py --encoder sentence-transformers/all-distilroberta-v1 --tag droberta
Outputs: <out-prefix>_results_<tag>.csv and <out-prefix>_summary_<tag>.md
(default out-prefix revision_pipeline/cv).
"""
import argparse, json, math, os, sys
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, hamming_loss, roc_auc_score
# Use the pipeline's own calibration and thresholding, not re-implementations
PIPE = os.path.dirname(os.path.abspath(__file__))
if PIPE not in sys.path:
    sys.path.insert(0, PIPE)   # python -I does not put the script directory on sys.path
from retrain import _fit_cdf_per_class, _apply_cdf_per_class, thresholds_from_percentile_weighted

LEVELS = ["Kn", "Cm", "Ap", "An", "Sn", "Ev"]

def load_core(path):
    df = pd.read_csv(path)
    if "level" in df.columns:  # long format from merge_core.py
        verbs = sorted(df["verb"].unique())
        v2i = {v: i for i, v in enumerate(verbs)}
        Y = np.zeros((len(verbs), 6), dtype=int)
        freq = np.zeros(len(verbs))
        for _, r in df.iterrows():
            i = v2i[r["verb"]]
            Y[i, LEVELS.index(r["level"])] = 1
            freq[i] = max(freq[i], float(r["freq"]))
        return verbs, Y, freq
    raise ValueError("expected long-format core csv")

def row_weights(freq):
    w = 1.0 + np.log1p(freq)
    w = np.clip(np.nan_to_num(w, nan=1.0, posinf=3.0, neginf=0.5), 0.5, 3.0)
    return w / w.mean()

def decision_scores(clf, X):
    if hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
    else:
        s = clf.predict_proba(X)
    return np.asarray(s, dtype=float)

def run_model(name, make_clf, X, Y, w, percentile, seeds=(13, 14, 15, 16, 17)):
    recs = []
    for seed in seeds:
        mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_pred = np.zeros_like(Y)
        oof_score = np.zeros(Y.shape, dtype=float)
        for fold, (tr, te) in enumerate(mskf.split(X, Y)):
            clf = OneVsRestClassifier(make_clf())
            try:
                clf.fit(X[tr], Y[tr], sample_weight=w[tr])
            except (TypeError, ValueError):
                clf.fit(X[tr], Y[tr])
            # Mirror the production rule: calibration CDFs and thresholds are fit on
            # INNER out-of-fold train scores (as retrain.py's _scores_oof does), which
            # matters greatly for kNN whose in-sample scores saturate at 1.0.
            from sklearn.model_selection import KFold
            S_tr_oof = np.zeros((len(tr), Y.shape[1]), dtype=float)
            inner = KFold(n_splits=5, shuffle=True, random_state=13)
            for itr, iva in inner.split(X[tr]):
                iclf = OneVsRestClassifier(make_clf())
                try:
                    iclf.fit(X[tr][itr], Y[tr][itr], sample_weight=w[tr][itr])
                except (TypeError, ValueError):
                    iclf.fit(X[tr][itr], Y[tr][itr])
                S_tr_oof[iva] = decision_scores(iclf, X[tr][iva])
            S_te = decision_scores(clf, X[te])
            cdfs = _fit_cdf_per_class(S_tr_oof)
            S_tr01 = _apply_cdf_per_class(S_tr_oof, cdfs)
            S_te01 = _apply_cdf_per_class(S_te, cdfs)
            thr = thresholds_from_percentile_weighted(S_tr01, Y[tr], percentile, w[tr])
            oof_pred[te] = (S_te01 >= thr.reshape(1, -1)).astype(int)
            oof_score[te] = S_te01
        aucs = [roc_auc_score(Y[:, j], oof_score[:, j]) for j in range(6)]
        rec = {"model": name, "seed": seed,
               "macro_AUC": float(np.mean(aucs)),
               "macro_F1": f1_score(Y, oof_pred, average="macro", zero_division=0),
               "micro_F1": f1_score(Y, oof_pred, average="micro", zero_division=0),
               "subset_acc": accuracy_score(Y, oof_pred),
               "hamming": hamming_loss(Y, oof_pred)}
        for j, lv in enumerate(LEVELS):
            rec[f"AUC_{lv}"] = aucs[j]
            rec[f"F1_{lv}"] = f1_score(Y[:, j], oof_pred[:, j], zero_division=0)
            rec[f"P_{lv}"] = precision_score(Y[:, j], oof_pred[:, j], zero_division=0)
            rec[f"R_{lv}"] = recall_score(Y[:, j], oof_pred[:, j], zero_division=0)
        rec["_oof"] = oof_pred
        rec["_oof_score"] = oof_score
        recs.append(rec)
    return recs

def baselines(Y, seeds=(13, 14, 15, 16, 17)):
    recs = []
    rng = np.random.default_rng(13)
    prev = Y.mean(axis=0)
    maj = np.zeros_like(Y); maj[:, np.argmax(prev)] = 1
    rec = {"model": "majority-level", "seed": 13,
           "macro_F1": f1_score(Y, maj, average="macro", zero_division=0),
           "micro_F1": f1_score(Y, maj, average="micro", zero_division=0),
           "subset_acc": accuracy_score(Y, maj), "hamming": hamming_loss(Y, maj), "_oof": maj}
    for j, lv in enumerate(LEVELS):
        rec[f"F1_{lv}"] = f1_score(Y[:, j], maj[:, j], zero_division=0)
    recs.append(rec)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        rnd = (rng.random(Y.shape) < prev.reshape(1, -1)).astype(int)
        rec = {"model": "freq-random", "seed": seed,
               "macro_F1": f1_score(Y, rnd, average="macro", zero_division=0),
               "micro_F1": f1_score(Y, rnd, average="micro", zero_division=0),
               "subset_acc": accuracy_score(Y, rnd), "hamming": hamming_loss(Y, rnd), "_oof": rnd}
        for j, lv in enumerate(LEVELS):
            rec[f"F1_{lv}"] = f1_score(Y[:, j], rnd[:, j], zero_division=0)
        recs.append(rec)
    return recs

def bootstrap_ci(Y, oof, n=10000, seed=13):
    rng = np.random.default_rng(seed)
    N = len(Y)
    vals = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, N, N)
        vals[b] = f1_score(Y[idx], oof[idx], average="macro", zero_division=0)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def mcnemar_exact(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b)); c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = sum(math.comb(n, k) for k in range(min(b, c) + 1)) * 2 / 2 ** n
    return b, c, min(1.0, p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-csv", default=os.path.join(PIPE, "core_merged.csv"))
    ap.add_argument("--winner-json", default=os.path.join(os.path.dirname(PIPE), "results", "winner_and_constants.json"),
                    help="retrain.py winner summary; its percentile sets the CV threshold percentile")
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--tag", default="mpnet")
    ap.add_argument("--out-prefix", default=os.path.join(PIPE, "cv"))
    ap.add_argument("--reference", default="kNN-21",
                    help="zoo model used as the reference of the summary (default kNN-21, the production model)")
    args = ap.parse_args()
    ref = args.reference

    verbs, Y, freq = load_core(args.core_csv)
    w = row_weights(freq)
    print(f"core: {len(verbs)} verbs, {Y.sum()} assignments")

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(args.encoder)
    X = np.asarray(enc.encode(verbs, normalize_embeddings=True, show_progress_bar=False), dtype=float)
    print("embedded:", X.shape)

    winner = json.load(open(args.winner_json, encoding="utf-8"))
    wname = winner["winner"]["name"]
    pct = int(winner["winner"]["percentile"])
    print("winner from scan:", wname, "| percentile:", pct)

    zoo = {
        "SGD-hinge-1e4": lambda: SGDClassifier(loss="hinge", alpha=1e-4, class_weight="balanced",
                                               max_iter=4000, tol=1e-4, random_state=13),
        "SGD-log-1e5": lambda: SGDClassifier(loss="log_loss", alpha=1e-5, class_weight="balanced",
                                             max_iter=4000, tol=1e-4, random_state=13),
        "SGD-hinge-1e5": lambda: SGDClassifier(loss="hinge", alpha=1e-5, class_weight="balanced",
                                               max_iter=4000, tol=1e-4, random_state=13),
        "LinearSVC-C1": lambda: LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, tol=1e-4, random_state=13),
        "LogReg-C1": lambda: LogisticRegression(C=1.0, solver="liblinear", penalty="l2",
                                                class_weight="balanced", max_iter=4000, random_state=13),
        "Ridge-a1": lambda: RidgeClassifier(alpha=1.0),
        "PassiveAggressive-C1": lambda: PassiveAggressiveClassifier(C=1.0, class_weight="balanced",
                                                                     max_iter=4000, tol=1e-4, random_state=13),
        "kNN-21": lambda: KNeighborsClassifier(n_neighbors=21, metric="cosine", weights="distance", algorithm="brute"),
        "kNN-11": lambda: KNeighborsClassifier(n_neighbors=11, metric="cosine", weights="distance", algorithm="brute"),
        "NearestCentroid": lambda: NearestCentroid(metric="euclidean"),
        "GaussianNB": lambda: GaussianNB(var_smoothing=1e-9),
        "LDA": lambda: make_pipeline(PCA(n_components=24, whiten=True, random_state=13),
                                     LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "QDA": lambda: make_pipeline(PCA(n_components=24, whiten=True, random_state=13),
                                     QuadraticDiscriminantAnalysis(reg_param=0.05)),
        "CalibratedSVC": lambda: CalibratedClassifierCV(SVC(kernel="linear", C=0.5,
                                     class_weight="balanced"), method="sigmoid", cv=3),
    }

    if ref not in zoo:
        raise SystemExit(f"--reference {ref!r} is not a zoo model: {sorted(zoo)}")

    all_recs = []
    for name, mk in zoo.items():
        print("CV:", name, flush=True)
        try:
            all_recs += run_model(name, mk, X, Y, w, pct)
        except Exception as e:
            print(f"  {name} FAILED: {type(e).__name__}: {e}", flush=True)
    all_recs += baselines(Y)

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_oof"} for r in all_recs])
    df.to_csv(f"{args.out_prefix}_results_{args.tag}.csv", index=False)

    lines = [f"# CV summary ({args.tag}, {len(verbs)} core verbs, repeated 5x5 stratified CV, winner percentile P{pct})", ""]
    lines.append("| Model | macro AUC | macro F1 (mean over repeats) | 95% CI (seed 13 repeat) | micro F1 | subset acc |")
    lines.append("|---|---|---|---|---|---|")
    ref_oof = None
    for name in list(zoo) + ["majority-level", "freq-random"]:
        sub = [r for r in all_recs if r["model"] == name]
        m = np.mean([r["macro_F1"] for r in sub])
        first = sub[0]
        lo, hi = bootstrap_ci(Y, first["_oof"])
        if name == ref:  # reference configuration (default kNN-21, the production model)
            ref_oof = first["_oof"]
        auc_m = np.mean([r.get('macro_AUC', float('nan')) for r in sub])
        lines.append(f"| {name} | {auc_m:.3f} | {m:.3f} | [{lo:.3f}, {hi:.3f}] | {np.mean([r['micro_F1'] for r in sub]):.3f} | {np.mean([r['subset_acc'] for r in sub]):.3f} |")
    lines.append("")
    lines.append(f"Per-level F1 (reference {ref}, mean over 5 repeats):")
    sub = [r for r in all_recs if r["model"] == ref]
    lines.append("| " + " | ".join(LEVELS) + " |")
    lines.append("|" + "---|" * 6)
    lines.append("| " + " | ".join(f"{np.mean([r[f'F1_{lv}'] for r in sub]):.3f}" for lv in LEVELS) + " |")
    lines.append("")
    lines.append(f"Paired exact McNemar (reference {ref} vs competitor, per verb-level decision, seed-13 repeat):")
    ycorr_w = (ref_oof == Y).ravel()
    for name in list(zoo) + ["majority-level", "freq-random"]:
        if name == ref:
            continue
        first = [r for r in all_recs if r["model"] == name][0]
        ycorr_c = (first["_oof"] == Y).ravel()
        b, c, p = mcnemar_exact(ycorr_w, ycorr_c)
        lines.append(f"- vs {name}: reference-only correct {b}, competitor-only correct {c}, exact p = {p:.4g}")
    open(f"{args.out_prefix}_summary_{args.tag}.md", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[:12]))
    print("saved:", f"{args.out_prefix}_summary_{args.tag}.md")

if __name__ == "__main__":
    main()
