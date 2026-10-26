"""supervised_eval.py - supervised evaluation of every one-vs-rest model-zoo configuration (round 3).

WHAT IT DOES
  For every one-vs-rest (OvR) family and EVERY hyperparameter configuration listed in
  results/winner_and_constants.json "model_zoo" (27 configurations; the vote-based "Legacy kNN"
  is not an OvR classifier and is not evaluated here) and EVERY threshold percentile
  P20, P25, ..., P60, this script runs
    (a) repeated stratified cross-validation: MultilabelStratifiedKFold(5, shuffle=True,
        random_state=seed) for the five repeats seed = 13..17 (the cv_metrics.py protocol), and
    (b) the held-out full-pipeline protocol (the methodology_upgrades.py protocol) over the
        same outer partitions, seeds 13..17: per-level calibrated scores, per-level percentile
        thresholds, top-2 margin collapse, accept/reject gate, primary level, soft tier.
  Each model is fitted ONCE per (configuration, repeat, outer fold) and all nine percentiles
  are evaluated from the same calibrated out-of-fold scores (fits and calibration do not
  depend on the percentile).

PIPELINE PER OUTER FOLD (identical to round 2; nothing is tuned on the test fold)
  * data: revision_pipeline/core_merged.csv (381 verbs sorted, 607 assignments) and the
    production MPNet vectors revision_pipeline/all_embeddings.npz (cast to float64).
  * row weights: cv_metrics.row_weights (1 + log1p(freq), clipped to [0.5, 3], divided by the
    mean). The weighted OvR fit is ATTEMPTED; with scikit-learn 1.9.0 OneVsRestClassifier.fit
    raises ValueError for sample_weight and the fit falls back to the UNWEIGHTED fit (the same
    try/except as retrain.py, cv_metrics.py and methodology_upgrades.py). The row weights
    therefore only reach the weighted threshold percentiles. The number of fallbacks is counted
    and written to grid_meta.json ("weighted_fit_fallbacks").
  * calibration: inner KFold(5, shuffle=True, random_state=13) out-of-fold scores on the training
    part -> per-level empirical CDF (retrain._fit_cdf_per_class) -> applied to the outer-fold
    model's test scores (retrain._apply_cdf_per_class).
  * thresholds: retrain.thresholds_from_percentile_weighted (row-weighted percentile among the
    training positives of each level).
  * CV decisions: calibrated score >= threshold (no collapse), as cv_metrics.py.
  * held-out decisions: CV decisions -> retrain._apply_top2_margin(delta =
    retrain.TOP2_MARGIN_DEFAULT = 0.060); accepted = at least one level on; primary = argmax of
    the calibrated scores among accepted levels (retrain._choose_primary); soft tier = calibrated
    score >= retrain.SUGGEST_ALPHA (0.85) x threshold. Held-out metrics are computed per outer
    fold and averaged over the five folds, as methodology_upgrades.py.

LDA AND QDA
  Primary rows (cfg_id without a variant suffix) are built with retrain.make_estimator, i.e. the
  SAME projection as the structural scan: PCA(whiten=True, random_state=13) with
  n_components = retrain._pca_dim_for_lda_qda(X_fit, n_classes=2) = 2. Supplementary rows (cfg_id
  suffix "|PCA24") use the 24-component whitened PCA of round-2 cv_metrics.py; they exist only to
  reproduce the round-2 CV numbers and are flagged supplementary=True.

OUTPUTS (results/selection/)
  grid      supervised_grid_per_seed.csv   one row per (configuration, percentile, repeat)
            oof_decisions.npz              per-verb out-of-fold calibrated scores, fold ids,
                                           thresholds, CV decisions, held-out decisions, primaries
            grid_meta.json                 configurations, versions, runtimes, warnings, fallbacks
  analyze   supervised_grid.csv            one row per (configuration, percentile): mean and SD
                                           over the 5 repeats, every repeat, bootstrap CIs and
                                           exact McNemar vs kNN-21 P50 (seed 13 and pooled)
            supervised_grid_bootstrap.csv  95% percentile-bootstrap CIs (10,000 resamples of the
                                           381 verbs), per repeat and averaged over repeats
            supervised_grid_mcnemar.csv    exact McNemar vs kNN-21 P50, every configuration,
                                           percentile and repeat (+ pooled, descriptive only)
            matched_supervised.csv         each family at its structural best configuration
            mcnemar_vs_production.csv      exact McNemar vs kNN-21 P50 at the matched configurations
            baselines.csv                  majority-level and frequency-random baselines with CIs
            production_per_level.csv       per-level metrics of the production model kNN-21 P50
            v2_reproduction_check.csv      grid values at the round-2 configurations vs the
                                           committed round-2 outputs
  encoders  encoder_ablation.csv           matched configurations of every floor-passing family
                                           under all-mpnet-base-v2, all-MiniLM-L6-v2,
                                           all-distilroberta-v1 (offline, cached models)
            encoder_vectors.npz            the encoder vectors used (so the ablation reruns
                                           without the encoders)

STRUCTURAL BEST CONFIGURATION PER FAMILY
  From results/structural_scan_components.csv (written by retrain.py): rows with
  PassedMinAccept and a finite PenalizedObj_eval; maximise PenalizedObj_eval, ties broken by
  SCS_eval then Eval_acceptance (stable). --best-configs PATH overrides it with a JSON file
  ({"families": [{"family", "params", "percentile"}, ...]}).

USAGE (from the repository root; no network access is needed or attempted)
  python -I -B revision_pipeline/supervised_eval.py all [--workers 8]
  python -I -B revision_pipeline/supervised_eval.py grid [--workers 8]
  python -I -B revision_pipeline/supervised_eval.py analyze
  python -I -B revision_pipeline/supervised_eval.py encoders [--workers 8] [--reencode]
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ["HF_HUB_OFFLINE"] = "1"          # never download; the encoders are cached locally
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys
import ast
import json
import time
import argparse
import platform
import traceback
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score,
                             hamming_loss, roc_auc_score)

import retrain as rt
import cv_metrics as cvm

warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS = os.path.join(REPO, "results")
SEL_DIR = os.path.join(RESULTS, "selection")
CORE_CSV = os.path.join(HERE, "core_merged.csv")
EMB_NPZ = os.path.join(HERE, "all_embeddings.npz")
ZOO_JSON = os.path.join(RESULTS, "winner_and_constants.json")
SCAN_CSV = os.path.join(RESULTS, "structural_scan_components.csv")

LEVELS = ["Kn", "Cm", "Ap", "An", "Sn", "Ev"]
P_GRID = list(range(20, 61, 5))
SEEDS = (13, 14, 15, 16, 17)            # outer MultilabelStratifiedKFold seeds (CV repeats)
INNER_SEED = 13                         # inner KFold for the out-of-fold ECDF calibration
TOP2_MARGIN = rt.TOP2_MARGIN_DEFAULT    # 0.060
SUGGEST_ALPHA = rt.SUGGEST_ALPHA        # 0.85
assert abs(TOP2_MARGIN - 0.060) < 1e-12 and abs(SUGGEST_ALPHA - 0.85) < 1e-12
LEGACY = "Legacy kNN"
REF_FAMILY = "KNeighborsClassifier"
REF_ID, REF_P = 'KNeighborsClassifier|{"n_neighbors": 21}', 50      # production model kNN-21 P50
ADEQUACY_FLOOR = 0.5                    # PenalizedObj floor of the selection rule (encoder ablation only)
B_BOOT = 10000                          # bootstrap resamples of the 381 verbs
BOOT_SEED = 13                          # numpy default_rng seed (= cv_metrics.bootstrap_ci)
BOOT_METHOD = ("95% percentile bootstrap, 10000 resamples of the 381 verbs with replacement, "
               "numpy default_rng(13), identical resamples for every configuration and repeat (paired); "
               "avg5 = statistic averaged over the 5 repeats inside each resample")
CONST_KEYS = {"class_weight", "max_iter", "tol", "random_state", "solver", "penalty",
              "metric", "weights", "algorithm"}
ENCODERS = [("mpnet", "sentence-transformers/all-mpnet-base-v2"),
            ("minilm", "sentence-transformers/all-MiniLM-L6-v2"),
            ("droberta", "sentence-transformers/all-distilroberta-v1")]
V2_TABLE7_KNN21_AUC = {"mpnet": 0.713, "minilm": 0.703, "droberta": 0.660}   # cv_summary_<tag>.md


def short_params(params):
    return json.dumps({k: v for k, v in sorted(params.items()) if k not in CONST_KEYS})


def config_id(family, params, variant=""):
    """Configuration id shared with select_model.py and retrain.py."""
    return f"{family}|{short_params(params)}" + (f"|{variant}" if variant else "")


# ------------------------------------------------------------------------------------------
# data (module level so that spawned worker processes have it)
# ------------------------------------------------------------------------------------------
VERBS, Y, FREQ = cvm.load_core(CORE_CSV)
W = cvm.row_weights(FREQ)
N = len(VERBS)
_z = np.load(EMB_NPZ)
_vec = dict(zip([str(x) for x in _z["words"]], _z["vectors"]))
X_MPNET = np.array([_vec[v] for v in VERBS], dtype=float)
del _z, _vec

# ------------------------------------------------------------------------------------------
# configurations
# ------------------------------------------------------------------------------------------
_ZOO_ALL = json.load(open(ZOO_JSON, encoding="utf-8"))
_ZOO = _ZOO_ALL["model_zoo"]
GRID_BASE = {label: base for label, base, _ in rt.MODEL_GRID}
_GRID_PARAMS = {label: grid for label, _, grid in rt.MODEL_GRID}
V2_CV_NAME = {   # configurations that round-2 cv_metrics.py / methodology_upgrades.py ran (at P50)
    ("SGDClassifier", '{"alpha": 0.0001, "loss": "hinge"}', ""): "SGD-hinge-1e4",
    ("SGDClassifier", '{"alpha": 1e-05, "loss": "log_loss"}', ""): "SGD-log-1e5",
    ("SGDClassifier", '{"alpha": 1e-05, "loss": "hinge"}', ""): "SGD-hinge-1e5",
    ("LinearSVC", '{"C": 1.0}', ""): "LinearSVC-C1",
    ("LogisticRegression", '{"C": 1.0}', ""): "LogReg-C1",
    ("RidgeClassifier", '{"alpha": 1.0}', ""): "Ridge-a1",
    ("PassiveAggressive", '{"C": 1.0}', ""): "PassiveAggressive-C1",
    ("KNeighborsClassifier", '{"n_neighbors": 21}', ""): "kNN-21",
    ("KNeighborsClassifier", '{"n_neighbors": 11}', ""): "kNN-11",
    ("NearestCentroid", '{}', ""): "NearestCentroid",
    ("GaussianNB", '{"var_smoothing": 1e-09}', ""): "GaussianNB",
    ("LDA", '{"shrinkage": "auto"}', "PCA24"): "LDA",
    ("QDA", '{"reg_param": 0.05}', "PCA24"): "QDA",
    ("CalibratedSVC", '{"C": 0.5}', ""): "CalibratedSVC",
}
V2_HO_NAME = {"kNN-21": "kNN-21", "LogReg-C1": "LogReg-C1", "NearestCentroid": "NearestCentroid",
              "CalibratedSVC": "CalibratedSVC"}       # rows of revision_pipeline/heldout_results.csv

OVR_FAMILIES = []
CONFIGS = []
for _entry in _ZOO:
    _fam = _entry["model"]
    if _fam == LEGACY:
        continue
    assert _entry["param_grid"] == _GRID_PARAMS[_fam], f"model_zoo and retrain.MODEL_GRID differ for {_fam}"
    OVR_FAMILIES.append(_fam)
    for _params in _entry["param_grid"]:
        for _var in (["", "PCA24"] if _fam in ("LDA", "QDA") else [""]):
            _sp = short_params(_params)
            CONFIGS.append(dict(cfg_id=config_id(_fam, _params, _var), family=_fam, params=_params,
                                params_short=_sp, variant=_var, supplementary=(_var == "PCA24"),
                                v2_cv_name=V2_CV_NAME.get((_fam, _sp, _var))))
CFG_BY_ID = {c["cfg_id"]: c for c in CONFIGS}
CI_BY_ID = {c["cfg_id"]: i for i, c in enumerate(CONFIGS)}
assert REF_ID in CFG_BY_ID
assert _ZOO_ALL["winner"]["params"] == CFG_BY_ID[REF_ID]["params"] and int(_ZOO_ALL["winner"]["percentile"]) == REF_P, \
    "production winner in winner_and_constants.json is not kNN-21 P50"

PROJECTION_NOTE = ("LDA/QDA primary rows: retrain.make_estimator -> PCA(whiten=True, random_state=13, "
                   "n_components=retrain._pca_dim_for_lda_qda(X_fit, n_classes=2)) -> estimator, the projection of the "
                   "structural scan; supplementary rows '|PCA24': PCA(24, whiten=True, random_state=13) of round-2 cv_metrics.py")


def make_factory(cfg):
    fam, params, var = cfg["family"], dict(cfg["params"]), cfg["variant"]
    if fam in ("LDA", "QDA"):
        base = LinearDiscriminantAnalysis if fam == "LDA" else QuadraticDiscriminantAnalysis
        if var == "PCA24":      # round-2 cv_metrics.py projection (supplementary)
            return lambda Xf: make_pipeline(PCA(n_components=24, whiten=True, random_state=13), base(**params))
        return lambda Xf: rt.make_estimator(fam, GRID_BASE[fam], dict(params), Xf, None)   # structural scan
    base = GRID_BASE[fam]       # retrain.MODEL_GRID constructor (CalibratedSVC -> _make_calibrated_svc)
    return lambda Xf: base(**params)


_FALLBACKS = {"weighted_fit_ok": 0, "weighted_fit_fallback_unweighted": 0}


def fit_ovr(est, Xf, Yf, wf):
    """Row-weight behaviour kept from round 2 and made explicit: the weighted one-vs-rest fit is
    attempted; on scikit-learn 1.9.0 OneVsRestClassifier.fit rejects sample_weight (ValueError),
    so the model is fitted UNWEIGHTED. Row weights then only enter the threshold percentiles."""
    clf = OneVsRestClassifier(est)
    try:
        clf.fit(Xf, Yf, sample_weight=wf)
        _FALLBACKS["weighted_fit_ok"] += 1
    except (TypeError, ValueError):
        clf.fit(Xf, Yf)
        _FALLBACKS["weighted_fit_fallback_unweighted"] += 1
    return clf


# ------------------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------------------
def cv_record(S, pred):
    """cv_metrics.run_model metric block (pooled out-of-fold over the 381 verbs of one repeat)."""
    aucs = [roc_auc_score(Y[:, j], S[:, j]) for j in range(6)]
    rec = {"macro_AUC": float(np.mean(aucs)),
           "macro_F1": f1_score(Y, pred, average="macro", zero_division=0),
           "micro_F1": f1_score(Y, pred, average="micro", zero_division=0),
           "subset_acc": accuracy_score(Y, pred),
           "hamming": hamming_loss(Y, pred)}
    for j, lv in enumerate(LEVELS):
        rec[f"AUC_{lv}"] = aucs[j]
        rec[f"F1_{lv}"] = f1_score(Y[:, j], pred[:, j], zero_division=0)
        rec[f"P_{lv}"] = precision_score(Y[:, j], pred[:, j], zero_division=0)
        rec[f"R_{lv}"] = recall_score(Y[:, j], pred[:, j], zero_division=0)
    return rec


def primary_level(S01, hard):
    """retrain._choose_primary as level indices; -1 = rejected (no level accepted)."""
    accepted = hard.sum(1) > 0
    return np.where(accepted, np.argmax(S01 * hard, axis=1), -1)


_LEVEL_GAP = np.abs(np.arange(6)[:, None] - np.arange(6)[None, :])


def level_distances(Yt):
    """(n, 6): distance from each candidate level to the nearest true level of each verb."""
    return np.where(np.asarray(Yt)[:, None, :] == 1, _LEVEL_GAP[None, :, :], 99).min(axis=2)


def ordinal_distance(Yt, prim):
    """min |primary - true level| per verb; -1 for rejected verbs."""
    prim = np.asarray(prim, dtype=int)
    d = level_distances(Yt)[np.arange(len(prim)), np.clip(prim, 0, 5)]
    return np.where(prim >= 0, d, -1)


def ho_evaluate(Y_te, S01, hard, soft):
    """methodology_upgrades.py evaluate() plus within-one over all held-out verbs and the soft gate."""
    accepted = hard.sum(1) > 0
    prim = primary_level(S01, hard)
    idx = np.where(accepted)[0]
    prim_in = np.mean([Y_te[i, prim[i]] == 1 for i in idx]) if len(idx) else 0.0
    dist = ordinal_distance(Y_te, prim)
    dists = dist[idx]
    within1 = np.mean(dists <= 1) if len(idx) else 0.0
    mdist = np.mean(dists) if len(idx) else np.nan
    f1 = f1_score(Y_te, hard, average="macro", zero_division=0)
    try:
        auc = float(np.mean([roc_auc_score(Y_te[:, j], S01[:, j]) for j in range(6)]))
    except ValueError:
        auc = np.nan
    return dict(gate=accepted.mean(), prim=prim_in, within1=within1,
                within1_all=float(np.sum(dists <= 1)) / len(Y_te), mdist=mdist, f1=f1, auc=auc,
                soft_gate=float(np.mean(soft.sum(1) > 0))), prim


def oof_scores(factory, X, seed, pcts):
    """One repeat: outer MultilabelStratifiedKFold(5, seed); per outer fold an OvR fit on the
    training part, inner KFold(5, 13) OOF scores -> per-level ECDF, calibrated test scores and
    thresholds for every percentile in pcts (fits and calibration do not depend on the percentile)."""
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    S = np.zeros((N, 6)); fold_id = np.full(N, -1, dtype=int)
    thr = np.zeros((5, len(pcts), 6))
    for f, (tr, te) in enumerate(mskf.split(X, Y)):
        clf = fit_ovr(factory(X[tr]), X[tr], Y[tr], W[tr])
        S_tr_oof = np.zeros((len(tr), 6), dtype=float)
        for itr, iva in KFold(n_splits=5, shuffle=True, random_state=INNER_SEED).split(X[tr]):
            iclf = fit_ovr(factory(X[tr][itr]), X[tr][itr], Y[tr][itr], W[tr][itr])
            S_tr_oof[iva] = cvm.decision_scores(iclf, X[tr][iva])
        S_te = cvm.decision_scores(clf, X[te])
        cdfs = rt._fit_cdf_per_class(S_tr_oof)
        S_tr01 = rt._apply_cdf_per_class(S_tr_oof, cdfs)
        S[te] = rt._apply_cdf_per_class(S_te, cdfs)
        fold_id[te] = f
        for k, p in enumerate(pcts):
            thr[f, k] = rt.thresholds_from_percentile_weighted(S_tr01, Y[tr], p, W[tr])
    return S, fold_id, thr


def evaluate_repeat(S, fold_id, thr, pcts):
    """CV and held-out metrics of one repeat for every percentile, plus per-verb decisions."""
    rows = []
    pred_cv = np.zeros((len(pcts), N, 6), dtype=np.int8)
    pred_ho = np.zeros((len(pcts), N, 6), dtype=np.int8)
    prim_ho = np.full((len(pcts), N), -1, dtype=np.int8)
    for k, p in enumerate(pcts):
        pred = (S >= thr[fold_id, k, :]).astype(int)
        pred_cv[k] = pred
        rec = {"percentile": p}
        rec.update({f"CV_{a}": b for a, b in cv_record(S, pred).items()})
        fold_recs = []
        for f in range(5):
            te = np.where(fold_id == f)[0]
            hard = rt._apply_top2_margin(S[te], pred[te], delta=TOP2_MARGIN)
            soft = (S[te] >= SUGGEST_ALPHA * thr[f, k].reshape(1, -1)).astype(int)
            pred_ho[k, te] = hard
            m, prim = ho_evaluate(Y[te], S[te], hard, soft)
            prim_ho[k, te] = prim
            fold_recs.append(m)
        fm = pd.DataFrame(fold_recs).mean(numeric_only=True)
        rec.update({f"HO_{a}": float(fm[a]) for a in fm.index})
        rec["HO_pooled_macro_F1"] = f1_score(Y, pred_ho[k], average="macro", zero_division=0)
        rows.append(rec)
    return rows, pred_cv, pred_ho, prim_ho


def _run(cfg, X, pcts):
    factory = make_factory(cfg)
    t0 = time.time()
    out = dict(cfg_id=cfg["cfg_id"], seeds={}, errors=[], rows=[], warnings={})
    _FALLBACKS.update(weighted_fit_ok=0, weighted_fit_fallback_unweighted=0)
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=FutureWarning)
        for seed in SEEDS:
            try:
                S, fold_id, thr = oof_scores(factory, X, seed, pcts)
                rows, pred_cv, pred_ho, prim_ho = evaluate_repeat(S, fold_id, thr, pcts)
                for r in rows:
                    out["rows"].append(dict(cfg_id=cfg["cfg_id"], family=cfg["family"], variant=cfg["variant"],
                                            seed=seed, **r))
                out["seeds"][seed] = dict(S=S, fold_id=fold_id, thr=thr, pred_cv=pred_cv, pred_ho=pred_ho,
                                          prim_ho=prim_ho)
            except Exception as e:  # recorded, never silently dropped
                out["errors"].append(dict(seed=seed, error=f"{type(e).__name__}: {e}",
                                          tb=traceback.format_exc()[-2000:]))
    for wi in wlog:
        key = f"{wi.category.__name__}: {str(wi.message)[:120]}"
        out["warnings"][key] = out["warnings"].get(key, 0) + 1
    out["fallbacks"] = dict(_FALLBACKS)
    out["runtime_s"] = time.time() - t0
    return out


def run_grid_config(ci):
    return _run(CONFIGS[ci], X_MPNET, P_GRID)


# ------------------------------------------------------------------------------------------
# structural best configuration per family
# ------------------------------------------------------------------------------------------
def structural_best(best_json=None):
    """{family: dict(params, percentile, PenalizedObj, SCS_eval, Eval_acceptance, source)}."""
    out = {}
    if best_json:
        sj = json.load(open(best_json, encoding="utf-8"))
        for it in sj["families"]:
            fam = it["family"]
            if fam == LEGACY or it.get("percentile") is None:
                continue
            out[fam] = dict(params=it["params"], percentile=int(it["percentile"]),
                            PenalizedObj=it.get("PenalizedObj"), SCS_eval=it.get("SCS"),
                            Eval_acceptance=it.get("acceptance"), source=os.path.basename(best_json))
        return out
    scan = pd.read_csv(SCAN_CSV)
    ok = scan["PassedMinAccept"].astype(str).str.lower().eq("true") & scan["PenalizedObj_eval"].notna()
    for fam in OVR_FAMILIES:
        s = scan[(scan["Model"] == fam) & ok]
        s = s.sort_values(["PenalizedObj_eval", "SCS_eval", "Eval_acceptance"], ascending=False, kind="mergesort")
        r = s.iloc[0]
        out[fam] = dict(params=ast.literal_eval(r["Params"]), percentile=int(r["Percentile"]),
                        PenalizedObj=float(r["PenalizedObj_eval"]), SCS_eval=float(r["SCS_eval"]),
                        Eval_acceptance=float(r["Eval_acceptance"]),
                        source="results/structural_scan_components.csv (max PenalizedObj_eval; ties SCS_eval, Eval_acceptance)")
    return out


# ------------------------------------------------------------------------------------------
# step 1: grid
# ------------------------------------------------------------------------------------------
def pool_map(fn, items, workers, label):
    results = {}
    t0 = time.time()
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, it): it for it in items}
            for fu in as_completed(futs):
                it = futs[fu]
                try:
                    r = fu.result()
                except Exception as e:
                    r = dict(cfg_id=str(it), seeds={}, rows=[], runtime_s=np.nan, warnings={}, fallbacks={},
                             errors=[dict(seed=None, error=f"worker crash {type(e).__name__}: {e}", tb="")])
                results[it] = r
                print(f"  {label} done {r['cfg_id']}: {r['runtime_s']:.1f}s, errors={len(r['errors'])} "
                      f"(elapsed {time.time() - t0:.0f}s)", flush=True)
    else:
        for it in items:
            results[it] = fn(it)
            r = results[it]
            print(f"  {label} done {r['cfg_id']}: {r['runtime_s']:.1f}s, errors={len(r['errors'])}", flush=True)
    return results


def _stack(results, n_cfg, key, shape, dtype, fill):
    A = np.full((n_cfg, len(SEEDS)) + shape, fill, dtype=dtype)
    for i in range(n_cfg):
        for si, sd in enumerate(SEEDS):
            r = results[i]["seeds"].get(sd)
            if r is not None:
                A[i, si] = r[key]
    return A


def step_grid(workers):
    os.makedirs(SEL_DIR, exist_ok=True)
    t0 = time.time()
    print(f"grid: {len(CONFIGS)} configurations ({sum(c['supplementary'] for c in CONFIGS)} supplementary) x "
          f"{len(P_GRID)} percentiles x {len(SEEDS)} repeats; workers={workers}", flush=True)
    order = sorted(range(len(CONFIGS)),
                   key=lambda i: 0 if CONFIGS[i]["family"] in ("CalibratedSVC", "SGDClassifier", "QDA", "LDA") else 1)
    res = pool_map(run_grid_config, order, workers, "grid")
    rows = [row for i in range(len(CONFIGS)) for row in res[i]["rows"]]
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(SEL_DIR, "supervised_grid_per_seed.csv"), index=False)
    nc = len(CONFIGS)
    np.savez_compressed(
        os.path.join(SEL_DIR, "oof_decisions.npz"),
        cfg_ids=np.array([c["cfg_id"] for c in CONFIGS]), verbs=np.array(VERBS), levels=np.array(LEVELS),
        Y=Y.astype(np.int8), row_weights=W, seeds=np.array(SEEDS), percentiles=np.array(P_GRID),
        S_cal_oof=_stack(res, nc, "S", (N, 6), float, np.nan),                 # calibrated out-of-fold scores
        fold_id=_stack(res, nc, "fold_id", (N,), np.int8, -1),                 # outer fold of each verb
        thr=_stack(res, nc, "thr", (5, len(P_GRID), 6), float, np.nan),        # per-fold thresholds
        pred_cv=_stack(res, nc, "pred_cv", (len(P_GRID), N, 6), np.int8, -1),  # CV decisions (no collapse)
        pred_ho=_stack(res, nc, "pred_ho", (len(P_GRID), N, 6), np.int8, -1),  # held-out decisions (top-2 collapse)
        primary_ho=_stack(res, nc, "prim_ho", (len(P_GRID), N), np.int8, -2),  # held-out primary level, -1 rejected
    )
    import sklearn
    meta = dict(
        script="revision_pipeline/supervised_eval.py grid", workers=workers, runtime_total_s=time.time() - t0,
        n_configs=nc, percentiles=P_GRID, seeds=list(SEEDS), inner_seed=INNER_SEED,
        top2_margin=TOP2_MARGIN, suggest_alpha=SUGGEST_ALPHA,
        lda_qda_projection=PROJECTION_NOTE,
        lda_qda_structural_pca_dim_full_core=int(rt._pca_dim_for_lda_qda(X_MPNET, n_classes=2, hard_cap=256)),
        row_weights="cv_metrics.row_weights; weighted OvR fit attempted, falls back to unweighted on ValueError/TypeError",
        weighted_fit_fallbacks={c["cfg_id"]: res[i]["fallbacks"] for i, c in enumerate(CONFIGS)},
        warnings={c["cfg_id"]: res[i]["warnings"] for i, c in enumerate(CONFIGS) if res[i]["warnings"]},
        errors=[dict(cfg_id=c["cfg_id"], **e) for i, c in enumerate(CONFIGS) for e in res[i]["errors"]],
        runtime_s={c["cfg_id"]: res[i]["runtime_s"] for i, c in enumerate(CONFIGS)},
        configs=[{k: v for k, v in c.items()} for c in CONFIGS],
        versions=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                      pandas=pd.__version__, sklearn=sklearn.__version__))
    with open(os.path.join(SEL_DIR, "grid_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)
    nerr = len(meta["errors"])
    print(f"grid: wrote {len(per_seed)} per-seed rows; errors={nerr}; runtime {time.time() - t0:.1f}s", flush=True)
    if nerr:
        raise SystemExit("grid finished WITH ERRORS; see grid_meta.json")


# ------------------------------------------------------------------------------------------
# bootstrap machinery (vectorized; checked against cv_metrics.bootstrap_ci in step_analyze)
# ------------------------------------------------------------------------------------------
class Boot:
    def __init__(self, Ymat, B=B_BOOT, seed=BOOT_SEED):
        self.Y = Ymat
        n = Ymat.shape[0]
        rng = np.random.default_rng(seed)        # identical resample stream to cv_metrics.bootstrap_ci
        self.Cnt = np.zeros((B, n), dtype=np.int64)
        for b in range(B):
            self.Cnt[b] = np.bincount(rng.integers(0, n, n), minlength=n)
        self.Cf = self.Cnt.astype(float)

    def f1_levels(self, pred):
        P = pred.astype(float); Yf = self.Y.astype(float)
        TP = self.Cf @ (Yf * P); FP = self.Cf @ ((1 - Yf) * P); FN = self.Cf @ (Yf * (1 - P))
        den = 2 * TP + FP + FN
        return np.where(den > 0, 2 * TP / np.where(den > 0, den, 1), 0.0)     # (B, 6)

    def f1(self, pred):
        return self.f1_levels(pred).mean(axis=1)

    def micro_f1(self, pred):
        P = pred.astype(float); Yf = self.Y.astype(float)
        TP = (self.Cf @ (Yf * P)).sum(1); FP = (self.Cf @ ((1 - Yf) * P)).sum(1); FN = (self.Cf @ (Yf * (1 - P))).sum(1)
        den = 2 * TP + FP + FN
        return np.where(den > 0, 2 * TP / np.where(den > 0, den, 1), 0.0)

    def auc_levels(self, S):
        """per-level AUC on each resample (ties count 1/2, = roc_auc_score on the resample), (B, 6)."""
        out = np.zeros((self.Cnt.shape[0], 6))
        for j in range(6):
            s = S[:, j]; y = self.Y[:, j]
            order = np.argsort(s, kind="mergesort"); ss = s[order]; ys = y[order]
            starts = np.r_[0, np.where(np.diff(ss) != 0)[0] + 1]
            Cs = self.Cnt[:, order]
            Pw = np.add.reduceat(Cs * ys[None, :], starts, axis=1).astype(float)
            Nw = np.add.reduceat(Cs * (1 - ys)[None, :], starts, axis=1).astype(float)
            below = np.cumsum(Nw, axis=1) - Nw
            out[:, j] = (Pw * (below + 0.5 * Nw)).sum(axis=1) / (Pw.sum(axis=1) * Nw.sum(axis=1))
        return out

    def auc(self, S):
        return self.auc_levels(S).mean(axis=1)

    def mean(self, v):
        """mean of a per-verb quantity on each resample."""
        return (self.Cf @ v.astype(float)) / self.Cf.sum(axis=1)

    def ratio(self, num, den):
        """sum(num)/sum(den) on each resample (NaN if the denominator is 0)."""
        a = self.Cf @ num.astype(float); b = self.Cf @ den.astype(float)
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def ci(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def mcnemar(correct_ref, correct_cmp):
    """cv_metrics.mcnemar_exact; b = only the reference (kNN-21 P50) correct, c = only the compared model correct."""
    return cvm.mcnemar_exact(np.asarray(correct_ref, bool), np.asarray(correct_cmp, bool))


def heldout_correct(prim, Ymat):
    """per verb: 1 if the held-out primary is accepted and within one level of a true level."""
    d = ordinal_distance(Ymat, prim.astype(int))
    acc = prim >= 0
    return acc, acc & (d <= 1) & (d >= 0)


# ------------------------------------------------------------------------------------------
# step 2: analysis
# ------------------------------------------------------------------------------------------
def step_analyze(best_json=None):
    t0 = time.time()
    z = np.load(os.path.join(SEL_DIR, "oof_decisions.npz"))
    cfg_ids = [str(c) for c in z["cfg_ids"]]
    assert cfg_ids == [c["cfg_id"] for c in CONFIGS], "oof_decisions.npz does not match the current configurations"
    assert [str(v) for v in z["verbs"]] == list(VERBS) and np.array_equal(z["Y"], Y)
    S_all, pred_cv_all, prim_all = z["S_cal_oof"], z["pred_cv"], z["primary_ho"]
    per_seed = pd.read_csv(os.path.join(SEL_DIR, "supervised_grid_per_seed.csv"), float_precision="round_trip")
    meta = json.load(open(os.path.join(SEL_DIR, "grid_meta.json"), encoding="utf-8"))
    bt = Boot(Y)
    ri, rk = CI_BY_ID[REF_ID], P_GRID.index(REF_P)

    # ---- check the vectorized bootstrap against cv_metrics.bootstrap_ci and sklearn
    lo_v2, hi_v2 = cvm.bootstrap_ci(Y, pred_cv_all[ri, 0, rk])
    lo_vec, hi_vec = ci(bt.f1(pred_cv_all[ri, 0, rk]))
    rng2 = np.random.default_rng(BOOT_SEED); ba = bt.auc(S_all[ri, 0]); bf = bt.f1(pred_cv_all[ri, 0, rk]); chk = []
    for b in range(5):
        idx = rng2.integers(0, N, N)
        chk.append((abs(np.mean([roc_auc_score(Y[idx, j], S_all[ri, 0][idx, j]) for j in range(6)]) - ba[b]),
                    abs(f1_score(Y[idx], pred_cv_all[ri, 0, rk][idx], average="macro", zero_division=0) - bf[b])))
    boot_check = dict(cv_metrics_bootstrap_ci=[lo_v2, hi_v2], vectorized=[lo_vec, hi_vec],
                      max_abs_auc_first5=max(c[0] for c in chk), max_abs_f1_first5=max(c[1] for c in chk))
    assert abs(lo_v2 - lo_vec) < 1e-12 and abs(hi_v2 - hi_vec) < 1e-12 and boot_check["max_abs_auc_first5"] < 1e-12 \
        and boot_check["max_abs_f1_first5"] < 1e-12, boot_check

    # ---- reference (kNN-21 P50) resample statistics per repeat
    ref, ref_w1 = {}, {}
    for si, sd in enumerate(SEEDS):
        acc, w1 = heldout_correct(prim_all[ri, si, rk], Y)
        ref_w1[sd] = w1
        ref[sd] = dict(auc=bt.auc(S_all[ri, si]), f1=bt.f1(pred_cv_all[ri, si, rk]),
                       w1=bt.ratio(w1, acc), w1all=bt.mean(w1))
    ref["avg5"] = {k: np.mean([ref[sd][k] for sd in SEEDS], axis=0) for k in ("auc", "f1", "w1", "w1all")}
    REP_KEYS = [f"seed{sd}" for sd in SEEDS] + ["avg5"]

    boot_rows, mc_rows = [], []
    for ci_, c in enumerate(CONFIGS):
        cid = c["cfg_id"]
        auc_b = {sd: bt.auc(S_all[ci_, si]) for si, sd in enumerate(SEEDS)}
        for k, p in enumerate(P_GRID):
            row = dict(cfg_id=cid, percentile=p)
            stats = {}
            hc = {}
            for si, sd in enumerate(SEEDS):
                acc, w1 = heldout_correct(prim_all[ci_, si, k], Y)
                hc[sd] = w1
                stats[sd] = dict(auc=auc_b[sd], f1=bt.f1(pred_cv_all[ci_, si, k]),
                                 w1=bt.ratio(w1, acc), w1all=bt.mean(w1))
                row[f"HO_within1_pooled_seed{sd}"] = float(w1.sum() / acc.sum()) if acc.sum() else np.nan
                row[f"HO_within1_all_pooled_seed{sd}"] = float(w1.mean())
                # exact McNemar vs kNN-21 P50 on the same repeat
                pr, pc = pred_cv_all[ri, si, rk], pred_cv_all[ci_, si, k]
                w1_ref = ref_w1[sd]
                rec = dict(cfg_id=cid, percentile=p, repeat=f"seed{sd}", independent_test=True)
                for unit, a, b_ in [("cells", (pr == Y).ravel(), (pc == Y).ravel()),
                                    ("labelset", (pr == Y).all(1), (pc == Y).all(1)),
                                    ("heldoutW1all", w1_ref, w1)]:
                    b1, c1, p1 = mcnemar(a, b_)
                    rec.update({f"{unit}_ref_only": b1, f"{unit}_cmp_only": c1, f"{unit}_p_exact": p1})
                mc_rows.append(rec)
            row["HO_within1_pooled_avg5"] = float(np.mean([row[f"HO_within1_pooled_seed{sd}"] for sd in SEEDS]))
            row["HO_within1_all_pooled_avg5"] = float(np.mean([row[f"HO_within1_all_pooled_seed{sd}"] for sd in SEEDS]))
            stats["avg5"] = {kk: np.mean([stats[sd][kk] for sd in SEEDS], axis=0) for kk in ("auc", "f1", "w1", "w1all")}
            for rep, key in zip([*SEEDS, "avg5"], REP_KEYS):
                st, rf = stats[rep], ref[rep]
                for m, name in [("auc", "CV_macro_AUC"), ("f1", "CV_macro_F1"),
                                ("w1", "HO_within1_pooled"), ("w1all", "HO_within1_all_pooled")]:
                    lo, hi = ci(st[m]); row[f"{name}_{key}_ci95_pctboot_lo"] = lo; row[f"{name}_{key}_ci95_pctboot_hi"] = hi
                    d = st[m] - rf[m]
                    lo, hi = ci(d)
                    row[f"d{name}_vs_kNN21P50_{key}_ci95_pctboot_lo"] = lo
                    row[f"d{name}_vs_kNN21P50_{key}_ci95_pctboot_hi"] = hi
                    row[f"d{name}_vs_kNN21P50_{key}_pctboot_frac_gt0"] = float(np.mean(d[np.isfinite(d)] > 0))
            boot_rows.append(row)
            # pooled McNemar over the five repeats: descriptive only (the repeats reuse the same verbs)
            pooled = dict(cfg_id=cid, percentile=p, repeat="pooled5_NOT_INDEPENDENT_descriptive", independent_test=False)
            for unit in ("cells", "labelset", "heldoutW1all"):
                a_l, b_l = [], []
                for si, sd in enumerate(SEEDS):
                    pr, pc = pred_cv_all[ri, si, rk], pred_cv_all[ci_, si, k]
                    if unit == "cells":
                        a_l.append((pr == Y).ravel()); b_l.append((pc == Y).ravel())
                    elif unit == "labelset":
                        a_l.append((pr == Y).all(1)); b_l.append((pc == Y).all(1))
                    else:
                        a_l.append(ref_w1[sd]); b_l.append(hc[sd])
                b1, c1, p1 = mcnemar(np.concatenate(a_l), np.concatenate(b_l))
                pooled.update({f"{unit}_ref_only": b1, f"{unit}_cmp_only": c1, f"{unit}_p_exact": p1})
            mc_rows.append(pooled)
    boot = pd.DataFrame(boot_rows)
    boot.insert(2, "bootstrap_method", BOOT_METHOD)
    boot.to_csv(os.path.join(SEL_DIR, "supervised_grid_bootstrap.csv"), index=False)
    mc = pd.DataFrame(mc_rows)
    mc.insert(3, "reference", "kNN-21 P50 (production), same repeat")
    mc.insert(4, "units", "cells = verb x level CV decisions; labelset = exact CV label set per verb; "
                          "heldoutW1all = held-out primary accepted and within one level (rejected = wrong)")
    mc.to_csv(os.path.join(SEL_DIR, "supervised_grid_mcnemar.csv"), index=False)

    # ---- supervised_grid.csv (wide)
    metric_cols = [c for c in per_seed.columns if c.startswith("CV_") or c.startswith("HO_")]
    g = per_seed.groupby(["cfg_id", "percentile"], sort=False)
    parts = [g.size().rename("n_repeats_ok"), g[metric_cols].mean().add_suffix("_mean5"),
             g[metric_cols].std(ddof=1).add_suffix("_sd5")]
    for sd in SEEDS:
        parts.append(per_seed[per_seed.seed == sd].set_index(["cfg_id", "percentile"])[metric_cols].add_suffix(f"_seed{sd}"))
    grid = pd.concat(parts, axis=1).reset_index()
    bsel = boot.drop(columns="bootstrap_method").set_index(["cfg_id", "percentile"])
    grid = grid.merge(bsel, left_on=["cfg_id", "percentile"], right_index=True, how="left")
    for rep, suf in [("seed13", "seed13"), ("pooled5_NOT_INDEPENDENT_descriptive", "pooled5_descriptive")]:
        mm = mc[mc.repeat == rep].set_index(["cfg_id", "percentile"]).drop(
            columns=["repeat", "independent_test", "reference", "units"])
        grid = grid.merge(mm.add_prefix("mcnemarExact_vs_kNN21P50_").add_suffix("_" + suf),
                          left_on=["cfg_id", "percentile"], right_index=True, how="left")
    info = pd.DataFrame([dict(cfg_id=c["cfg_id"], family=c["family"], params_short=c["params_short"],
                              params_full=json.dumps(c["params"], sort_keys=True), variant=c["variant"],
                              supplementary=c["supplementary"], v2_cv_name=c["v2_cv_name"],
                              lda_qda_projection=("structural scan (retrain.make_estimator, PCA 2 comps)"
                                                  if c["family"] in ("LDA", "QDA") and not c["variant"] else
                                                  ("round-2 cv_metrics PCA 24 comps" if c["variant"] == "PCA24" else "")),
                              runtime_s=meta["runtime_s"][c["cfg_id"]]) for c in CONFIGS])
    grid = info.merge(grid, on="cfg_id", how="left")
    grid["bootstrap_method"] = BOOT_METHOD
    grid.to_csv(os.path.join(SEL_DIR, "supervised_grid.csv"), index=False)
    print(f"analyze: supervised_grid.csv {grid.shape[0]} rows x {grid.shape[1]} columns", flush=True)

    # ---- matched configurations and McNemar vs production
    best = structural_best(best_json)
    mrows, mcp_rows = [], []
    for fam in OVR_FAMILIES:
        bc = best[fam]
        for var in (["", "PCA24"] if fam in ("LDA", "QDA") else [""]):
            cid = config_id(fam, bc["params"], var)
            r = grid[(grid.cfg_id == cid) & (grid.percentile == bc["percentile"])].iloc[0]
            row = dict(family=fam, variant=var, supplementary=(var == "PCA24"), cfg_id=cid,
                       percentile=bc["percentile"], PenalizedObj=bc["PenalizedObj"], SCS_eval=bc["SCS_eval"],
                       Eval_acceptance=bc["Eval_acceptance"],
                       floor_pass=(bc["PenalizedObj"] is not None and bc["PenalizedObj"] >= ADEQUACY_FLOOR),
                       structural_source=bc["source"])
            for col in grid.columns:
                if col.startswith(("CV_", "HO_", "dCV_", "dHO_", "mcnemarExact_")) or col == "n_repeats_ok":
                    row[col] = r[col]
            mrows.append(row)
            m = mc[(mc.cfg_id == cid) & (mc.percentile == bc["percentile"])]
            for _, x in m.iterrows():
                role = ("primary single test" if x["repeat"] == "seed13" else
                        "descriptive only (repeats reuse the same verbs; not independent)" if not x["independent_test"]
                        else "secondary (other repeat)")
                rec = dict(family=fam, variant=var, cfg_id=cid, percentile=bc["percentile"],
                           comparison="vs kNN-21 P50 (production)", repeat=x["repeat"], role=role)
                for unit in ("cells", "labelset", "heldoutW1all"):
                    rec[f"{unit}_ref_only_correct"] = x[f"{unit}_ref_only"]
                    rec[f"{unit}_cmp_only_correct"] = x[f"{unit}_cmp_only"]
                    rec[f"{unit}_p_exact_mcnemar"] = x[f"{unit}_p_exact"]
                mcp_rows.append(rec)
    matched = pd.DataFrame(mrows)
    matched.to_csv(os.path.join(SEL_DIR, "matched_supervised.csv"), index=False)
    mcp = pd.DataFrame(mcp_rows)
    mcp.insert(len(mcp.columns), "units", "cells = 381x6 verb-level CV decisions (cells of one verb are not independent); "
               "labelset = exact CV label set per verb; heldoutW1all = held-out primary accepted and within one level")
    mcp.to_csv(os.path.join(SEL_DIR, "mcnemar_vs_production.csv"), index=False)

    # ---- production model per-level metrics (kNN-21 P50)
    prow = []
    lv_f1 = {sd: bt.f1_levels(pred_cv_all[ri, si, rk]) for si, sd in enumerate(SEEDS)}
    lv_auc = {sd: bt.auc_levels(S_all[ri, si]) for si, sd in enumerate(SEEDS)}
    psd = per_seed[(per_seed.cfg_id == REF_ID) & (per_seed.percentile == REF_P)].set_index("seed")
    for j, lv in enumerate(LEVELS):
        r = dict(model="kNN-21 P50 (production)", level=lv, n_positive=int(Y[:, j].sum()))
        for m in ("F1", "P", "R", "AUC"):
            vals = psd[f"CV_{m}_{lv}"]
            r[f"CV_{m}_mean5"] = float(vals.mean()); r[f"CV_{m}_sd5"] = float(vals.std(ddof=1))
            for sd in SEEDS:
                r[f"CV_{m}_seed{sd}"] = float(vals.loc[sd])
        for name, src in (("F1", lv_f1), ("AUC", lv_auc)):
            lo, hi = ci(src[13][:, j]); r[f"CV_{name}_seed13_ci95_pctboot_lo"] = lo; r[f"CV_{name}_seed13_ci95_pctboot_hi"] = hi
            avg = np.mean([src[sd][:, j] for sd in SEEDS], axis=0)
            lo, hi = ci(avg); r[f"CV_{name}_avg5_ci95_pctboot_lo"] = lo; r[f"CV_{name}_avg5_ci95_pctboot_hi"] = hi
        r["bootstrap_method"] = BOOT_METHOD
        prow.append(r)
    pd.DataFrame(prow).to_csv(os.path.join(SEL_DIR, "production_per_level.csv"), index=False)

    # ---- baselines (cv_metrics.baselines definitions; prevalence over the 381 core verbs)
    base = {r["model"] + ("" if r["model"] == "majority-level" else f"|{r['seed']}"): r for r in cvm.baselines(Y)}
    prev = Y.mean(axis=0)
    maj_level = int(np.argmax(prev))
    q = prev / prev.sum()
    brow = []
    def _bl_row(name, seed, pred, extra):
        r = dict(baseline=name, seed=seed, macro_F1=f1_score(Y, pred, average="macro", zero_division=0),
                 micro_F1=f1_score(Y, pred, average="micro", zero_division=0), subset_acc=accuracy_score(Y, pred),
                 hamming=hamming_loss(Y, pred))
        for j, lv in enumerate(LEVELS):
            r[f"F1_{lv}"] = f1_score(Y[:, j], pred[:, j], zero_division=0)
        bf = bt.f1(pred); bm = bt.micro_f1(pred)
        r["macro_F1_ci95_pctboot_lo"], r["macro_F1_ci95_pctboot_hi"] = ci(bf)
        r["micro_F1_ci95_pctboot_lo"], r["micro_F1_ci95_pctboot_hi"] = ci(bm)
        r.update(extra)
        return r, bf
    # majority level: every verb gets the most prevalent level; as a held-out style primary it is accepted for all
    maj = base["majority-level"]["_oof"]
    dmaj = np.array([np.min(np.abs(np.where(Y[i] == 1)[0] - maj_level)) for i in range(N)])
    w1m = (dmaj <= 1).astype(float)
    r, _ = _bl_row("majority-level", 13, maj, dict(
        level=LEVELS[maj_level], macro_AUC=0.5, HO_gate=1.0, HO_prim=float(np.mean(Y[:, maj_level] == 1)),
        HO_within1=float(w1m.mean()), HO_within1_all=float(w1m.mean()), HO_mdist=float(dmaj.mean())))
    r["HO_within1_ci95_pctboot_lo"], r["HO_within1_ci95_pctboot_hi"] = ci(bt.mean(w1m))
    brow.append(r)
    # frequency-random (cv_metrics.baselines, seeds 13..17); held-out style primary: one level drawn with
    # probability proportional to prevalence, reported as its exact expectation over the draw
    dist_lv = np.array([[np.min(np.abs(np.where(Y[i] == 1)[0] - l)) for l in range(6)] for i in range(N)])
    ew1 = (dist_lv <= 1).astype(float) @ q
    eprim = Y.astype(float) @ q
    edist = dist_lv.astype(float) @ q
    fr_b = []
    for sd in SEEDS:
        rnd = base[f"freq-random|{sd}"]["_oof"]
        r, bf = _bl_row("freq-random", sd, rnd, dict(
            level="", macro_AUC=0.5, HO_gate=1.0, HO_prim=float(eprim.mean()), HO_within1=float(ew1.mean()),
            HO_within1_all=float(ew1.mean()), HO_mdist=float(edist.mean())))
        r["HO_within1_ci95_pctboot_lo"], r["HO_within1_ci95_pctboot_hi"] = ci(bt.mean(ew1))
        brow.append(r); fr_b.append(bf)
    bl = pd.DataFrame(brow)
    fr = bl[bl.baseline == "freq-random"]
    avg = dict(baseline="freq-random", seed="mean5")
    for col in fr.columns:
        if col not in ("baseline", "seed", "level") and not col.endswith(("_lo", "_hi")):
            avg[col] = float(fr[col].mean())
    avg["macro_F1_ci95_pctboot_lo"], avg["macro_F1_ci95_pctboot_hi"] = ci(np.mean(fr_b, axis=0))
    avg["HO_within1_ci95_pctboot_lo"], avg["HO_within1_ci95_pctboot_hi"] = ci(bt.mean(ew1))
    bl = pd.concat([bl, pd.DataFrame([avg])], ignore_index=True)
    bl["notes"] = ("macro_AUC = 0.5 by construction (constant or score-free predictions); HO_* treat the baseline "
                   "as a primary-level predictor accepting every verb; freq-random HO_* are exact expectations "
                   "over a prevalence-proportional level draw")
    bl["bootstrap_method"] = BOOT_METHOD
    bl.to_csv(os.path.join(SEL_DIR, "baselines.csv"), index=False)

    # ---- reproduction of the committed round-2 outputs at the round-2 configurations (P50)
    repo_cv = pd.read_csv(os.path.join(HERE, "cv_results_mpnet.csv"))
    repo_ho = pd.read_csv(os.path.join(HERE, "heldout_results.csv")).set_index("model")
    cvcols = ["macro_AUC", "macro_F1", "micro_F1", "subset_acc", "hamming"] + \
             [f"{m}_{l}" for l in LEVELS for m in ("AUC", "F1", "P", "R")]
    rrows = []
    for c in CONFIGS:
        if not c["v2_cv_name"]:
            continue
        gs = per_seed[(per_seed.cfg_id == c["cfg_id"]) & (per_seed.percentile == 50)].set_index("seed")
        rr = repo_cv[repo_cv.model == c["v2_cv_name"]].set_index("seed")
        diffs = {sd: max(abs(gs.loc[sd, f"CV_{k}"] - rr.loc[sd, k]) for k in cvcols) for sd in SEEDS}
        row = dict(cfg_id=c["cfg_id"], v2_name=c["v2_cv_name"], percentile=50,
                   CV_macro_AUC_mean5=gs["CV_macro_AUC"].mean(), v2_macro_AUC_mean5=rr["macro_AUC"].mean(),
                   CV_macro_F1_mean5=gs["CV_macro_F1"].mean(), v2_macro_F1_mean5=rr["macro_F1"].mean(),
                   maxabs_diff_all_CV_metrics_all_seeds=max(diffs.values()),
                   seeds_differing_gt_1e9=";".join(str(s) for s, d in diffs.items() if d > 1e-9))
        hn = V2_HO_NAME.get(c["v2_cv_name"])
        if hn:
            h = gs.loc[13]
            pairs = [("gate_recall", "HO_gate"), ("primary_in_labels", "HO_prim"), ("within1", "HO_within1"),
                     ("mean_ordinal_dist", "HO_mdist"), ("macro_F1", "HO_f1"), ("macro_AUC", "HO_auc")]
            row["HO_seed13_maxabs_diff_vs_heldout_results_rounded"] = max(
                abs(round(float(h[b]), 3 if a != "mean_ordinal_dist" else 2) - float(repo_ho.loc[hn, a])) for a, b in pairs)
        rrows.append(row)
    for name in ("majority-level", "freq-random"):
        rr = repo_cv[repo_cv.model == name].set_index("seed")
        cols = ["macro_F1", "micro_F1", "subset_acc", "hamming"] + [f"F1_{l}" for l in LEVELS]
        mine = bl[(bl.baseline == name) & (bl.seed != "mean5")].copy()
        mine["seed"] = mine["seed"].astype(int)
        mine = mine.set_index("seed")
        d = max(abs(float(mine.loc[sd, k]) - float(rr.loc[sd, k])) for sd in rr.index for k in cols)
        rrows.append(dict(cfg_id=name, v2_name=name, maxabs_diff_all_CV_metrics_all_seeds=d))
    rep = pd.DataFrame(rrows)
    rep["note"] = ("round-2 cv_metrics.py encoded the verbs live in one batch; this grid uses the production per-word "
                   "vectors (all_embeddings.npz). The two differ by about 1e-7, which changes only SGD and "
                   "PassiveAggressive results; heldout_results.csv is rounded to 3 (mean distance 2) decimals")
    rep.to_csv(os.path.join(SEL_DIR, "v2_reproduction_check.csv"), index=False)
    meta["analysis"] = dict(runtime_s=time.time() - t0, bootstrap_check=boot_check, bootstrap_method=BOOT_METHOD,
                            structural_best=best)
    with open(os.path.join(SEL_DIR, "grid_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)
    print(f"analyze: done in {time.time() - t0:.1f}s; bootstrap check {boot_check}", flush=True)
    show = matched[["family", "variant", "cfg_id", "percentile", "PenalizedObj", "CV_macro_AUC_mean5",
                    "HO_within1_mean5", "HO_within1_all_mean5"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.6f}"), flush=True)


# ------------------------------------------------------------------------------------------
# step 3: encoder ablation
# ------------------------------------------------------------------------------------------
ENC_NPZ = os.path.join(SEL_DIR, "encoder_vectors.npz")
_XCACHE = {}


def encoder_X(key):
    """key = '<tag>|per_word' or '<tag>|batch'. mpnet|per_word = production all_embeddings.npz."""
    if key == "mpnet|per_word":
        return X_MPNET
    if key not in _XCACHE:
        z = np.load(ENC_NPZ)
        assert [str(v) for v in z["verbs"]] == list(VERBS)
        _XCACHE[key] = np.asarray(z[key.replace("|", "__")], dtype=float)
    return _XCACHE[key]


def build_encoder_vectors(reencode=False):
    if os.path.exists(ENC_NPZ) and not reencode:
        z = np.load(ENC_NPZ)
        if [str(v) for v in z["verbs"]] == list(VERBS):
            print(f"encoders: reusing {os.path.relpath(ENC_NPZ, REPO)}", flush=True)
            return json.loads(str(z["info"]))
    from sentence_transformers import SentenceTransformer
    arrays, info = {}, {}
    for tag, name in ENCODERS:
        t0 = time.time()
        enc = SentenceTransformer(name, local_files_only=True)
        # per word, exactly as retrain._embed_once (production vectors are built this way)
        pw = np.stack([np.asarray(enc.encode([v], normalize_embeddings=True, convert_to_numpy=True)[0], dtype=np.float32)
                       for v in VERBS])
        # one batch, exactly as round-2 cv_metrics.py main()
        bt_ = np.asarray(enc.encode(VERBS, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
        if tag != "mpnet":
            arrays[f"{tag}__per_word"] = pw
        arrays[f"{tag}__batch"] = bt_
        info[tag] = dict(model=name, dim=int(pw.shape[1]), seconds=time.time() - t0,
                         maxabs_per_word_vs_batch=float(np.abs(pw - bt_).max()))
        if tag == "mpnet":
            info[tag]["maxabs_live_per_word_vs_all_embeddings_npz"] = float(np.abs(pw - X_MPNET).max())
    import sentence_transformers, torch
    info["versions"] = dict(sentence_transformers=sentence_transformers.__version__, torch=torch.__version__)
    np.savez_compressed(ENC_NPZ, verbs=np.array(VERBS), info=np.array(json.dumps(info)), **arrays)
    print(f"encoders: wrote {os.path.relpath(ENC_NPZ, REPO)} {json.dumps(info)}", flush=True)
    return info


def run_encoder_job(job):
    key, fam, params_json, pct = job
    params = json.loads(params_json)
    cfg = dict(cfg_id=f"{key}::{config_id(fam, params)}::P{pct}", family=fam, params=params, variant="")
    return _run(cfg, encoder_X(key), [pct])


def step_encoders(workers, best_json=None, reencode=False):
    t0 = time.time()
    os.makedirs(SEL_DIR, exist_ok=True)
    info = build_encoder_vectors(reencode)
    best = structural_best(best_json)
    floor_fams = [f for f in OVR_FAMILIES if best[f]["PenalizedObj"] is not None and best[f]["PenalizedObj"] >= ADEQUACY_FLOOR]
    pj = lambda p: json.dumps(p, sort_keys=True)
    jobs = [(f"{tag}|per_word", fam, pj(best[fam]["params"]), best[fam]["percentile"]) for tag, _ in ENCODERS for fam in floor_fams]
    jobs += [(f"{tag}|batch", REF_FAMILY, pj(CFG_BY_ID[REF_ID]["params"]), REF_P) for tag, _ in ENCODERS]
    print(f"encoders: {len(floor_fams)} floor-passing families {floor_fams}; {len(jobs)} jobs", flush=True)
    res = pool_map(run_encoder_job, jobs, workers, "encoders")
    errors = [dict(job=str(j), **e) for j in jobs for e in res[j]["errors"]]
    bt = Boot(Y)
    rows = []
    for j in jobs:
        key, fam, params, pct = j
        params = json.loads(params)
        tag, encoding = key.split("|")
        r = res[j]
        df = pd.DataFrame(r["rows"]).set_index("seed")
        row = dict(encoder=dict(ENCODERS)[tag], encoder_tag=tag,
                   encoding=("per word (production; retrain._embed_once)" if encoding == "per_word"
                             else "one batch (round-2 cv_metrics.py)"),
                   family=fam, cfg_id=config_id(fam, params), percentile=pct,
                   PenalizedObj_mpnet_structural=best[fam]["PenalizedObj"] if fam in best else np.nan,
                   production_model=(config_id(fam, params) == REF_ID and pct == REF_P))
        for m in ["CV_macro_AUC", "CV_macro_F1", "CV_micro_F1", "HO_within1", "HO_within1_all", "HO_gate",
                  "HO_prim", "HO_mdist", "HO_f1", "HO_auc"]:
            row[f"{m}_mean5"] = float(df[m].mean()); row[f"{m}_sd5"] = float(df[m].std(ddof=1))
            row[f"{m}_seed13"] = float(df.loc[13, m])
        auc_b = [bt.auc(r["seeds"][sd]["S"]) for sd in SEEDS]
        f1_b = [bt.f1(r["seeds"][sd]["pred_cv"][0]) for sd in SEEDS]
        row["CV_macro_AUC_avg5_ci95_pctboot_lo"], row["CV_macro_AUC_avg5_ci95_pctboot_hi"] = ci(np.mean(auc_b, axis=0))
        row["CV_macro_F1_avg5_ci95_pctboot_lo"], row["CV_macro_F1_avg5_ci95_pctboot_hi"] = ci(np.mean(f1_b, axis=0))
        if row["production_model"]:
            row["v2_table7_CV_macro_AUC"] = V2_TABLE7_KNN21_AUC[tag]
            row["abs_diff_vs_v2_table7_after_rounding"] = abs(round(row["CV_macro_AUC_mean5"], 3) - V2_TABLE7_KNN21_AUC[tag])
        row["runtime_s"] = r["runtime_s"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out["bootstrap_method"] = BOOT_METHOD
    out["note"] = ("configurations and percentiles = each family's MPNet structural best (matched); "
                   "the outer partitions, calibration and thresholds are those of the main grid")
    out.to_csv(os.path.join(SEL_DIR, "encoder_ablation.csv"), index=False)
    # the MPNet per-word rows must equal the main grid at the matched configurations
    grid_path = os.path.join(SEL_DIR, "supervised_grid_per_seed.csv")
    consistency = None
    if os.path.exists(grid_path):
        gps = pd.read_csv(grid_path, float_precision="round_trip")
        dmax = 0.0
        for j in jobs:
            key, fam, params, pct = j
            if key != "mpnet|per_word":
                continue
            params = json.loads(params)
            a = gps[(gps.cfg_id == config_id(fam, params)) & (gps.percentile == pct)].set_index("seed")
            b = pd.DataFrame(res[j]["rows"]).set_index("seed")
            cols = [c for c in b.columns if c.startswith(("CV_", "HO_"))]
            dmax = max(dmax, float(np.nanmax(np.abs(a.loc[list(SEEDS), cols].values - b.loc[list(SEEDS), cols].values))))
        consistency = dmax
    meta = dict(script="revision_pipeline/supervised_eval.py encoders", workers=workers,
                runtime_total_s=time.time() - t0, encoders=info, floor_families=floor_fams,
                maxabs_mpnet_rows_vs_main_grid=consistency, errors=errors,
                warnings={str(j): res[j]["warnings"] for j in jobs if res[j]["warnings"]})
    with open(os.path.join(SEL_DIR, "encoder_ablation_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)
    show = out[["encoder_tag", "encoding", "family", "percentile", "CV_macro_AUC_mean5", "HO_within1_mean5"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.6f}"), flush=True)
    print(f"encoders: done in {time.time() - t0:.1f}s; errors={len(errors)}; "
          f"max |mpnet rows - main grid| = {consistency}", flush=True)
    if errors:
        raise SystemExit("encoder ablation finished WITH ERRORS; see encoder_ablation_meta.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", nargs="?", default="all", choices=["all", "grid", "analyze", "encoders"])
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--best-configs", default=None,
                    help="optional JSON with each family's structural best configuration (default: derived from "
                         "results/structural_scan_components.csv)")
    ap.add_argument("--reencode", action="store_true", help="re-encode the encoder-ablation vectors")
    a = ap.parse_args()
    t0 = time.time()
    if a.step in ("all", "grid"):
        step_grid(a.workers)
    if a.step in ("all", "analyze"):
        step_analyze(a.best_configs)
    if a.step in ("all", "encoders"):
        step_encoders(a.workers, a.best_configs, a.reencode)
    print(f"supervised_eval.py {a.step}: total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
