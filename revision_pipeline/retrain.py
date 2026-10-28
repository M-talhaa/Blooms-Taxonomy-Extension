#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
retrain.py - standalone implementation of the article's Algorithm 1 (training, calibration and
structural tuning of the model zoo), the model selection of Section 2.3.5 and Algorithm 2
(classification of the candidate verbs with the selected model), with export of the winner
bundle and the classified verb bank. It reproduces the training and classification code of the
original "BT 3 - Model training and classification.ipynb" (commit 2a87a33; the current notebook
runs this script instead). "The notebook", "BT3" and the cell numbers below refer to that original
version; every deviation from it is listed under DEVIATIONS.

USAGE (from the repository root; every path is relative)
  Stage 1, structural scan only (input of supervised_eval.py, select_model.py, robustness.py):
    python -I -B revision_pipeline/retrain.py --core-csv revision_pipeline/core_merged.csv --candidates revision_pipeline/candidates_final.txt --outdir . --scan-only
  Final stage, structural scan + model selection + export of the selected family:
    python -I -B revision_pipeline/retrain.py --core-csv revision_pipeline/core_merged.csv --candidates revision_pipeline/candidates_final.txt --outdir <out>
  Dry run without the encoder (only the scan is meaningful; never use for real runs):
    add --mock-embeddings --mock-dim 64 --scan-only

REPRODUCTION ORDER (round 3; commands and runtimes in results/selection/RUN_LOG.md)
  1. retrain.py --scan-only --outdir .   results/structural_scan_components.csv,
                                         model_comparison_table5.csv, model_comparison_pareto.csv
  2. supervised_eval.py all              results/selection/supervised_grid*.csv and related files
  3. select_model.py                     results/selection/selection_table.csv, selection_report.md
  4. robustness.py                       results/selection/robustness_*.csv, robustness_report.md
  5. retrain.py --outdir <out>           repeats the deterministic scan, applies the selection rule
                                         of select_model.py to it and exports the selected family:
                                         <out>/results/Verb List Classified by Model.csv, the winner
                                         bundle and winner_and_constants.json
  With --outdir . step 5 rewrites the committed results/ files in place. production_model/ is the
  packaged copy of the winner bundle (README.md and classify.py added).

INPUTS
  --core-csv     CSV with columns: verb, Kn, Cm, Ap, An, Sn, Ev (0/1 multi-hot), freq
                 (lower-case level names also accepted; 'weight'/'frequency' accepted for freq).
                 A long-format CSV (columns: verb, level, freq[, ...], one row per
                 verb-level pair) is auto-detected and pivoted to the wide layout:
                 Y[verb, level] = 1 for each pair, freq[verb] = max freq across its
                 rows (matching the notebook, whose freq dict took the max per-level
                 weight of each verb).
  --candidates   txt file, one candidate verb per line (a CSV with a verb column and
                 optional Kn..Ev trace columns is also accepted)
  Model selection (final stage) also reads results/selection/supervised_grid_per_seed.csv
  (supervised_eval.py) and, as a cross-check, results/selection/selection_table.csv
  (select_model.py), both located relative to this file.

PIPELINE (BT3 cells 5 to 11, with the deviations below)
  1. Embed with sentence-transformers/all-mpnet-base-v2, normalize_embeddings=True (L2); the
     vectors are served from revision_pipeline/all_embeddings.npz (per-word encodings).
  2. Row weights: w = 1 + log(1 + freq), clip to [0.5, 3.0], divide by the mean (ROW WEIGHTS).
  3. Model zoo MODEL_GRID trained one-vs-rest, plus the vote-based Legacy kNN (LEGACY_KNN_GRID).
  4. Per-level empirical-CDF calibration fitted on OUT-OF-FOLD core scores,
     KFold(n_splits=N_SPLITS_OOF, shuffle=True, random_state=RNG_SEED).
  5. Percentile threshold scan over P_GRID (row-weighted per-level quantiles of the positives'
     calibrated core scores).
  6. StructuralStats on the 1,650 candidates, SCS = 1*D - 1*M + 0.8*T + 0.3*C - 0.05*E, and the
     penalized objective (paper Eq. 19) = SCS minus the acceptance and entries-per-accepted band
     penalties.
  7. Per-family structural best: the largest penalized objective, ties by raw SCS, then by
     acceptance (tune_ovr_model for the one-vs-rest families, tune_legacy_knn for Legacy kNN, D11).
  8. Model selection (D12): select_model.SELECTION_RULE (currently "balanced_maximin", the
     function select_model.rule_balanced_maximin) over the per-family structural bests of this
     run, with the matched supervised criteria of supervised_eval.py. Missing supervised results
     stop the run with an error; there is no structural-only fallback.
  9. Export of the selected family: tuning thresholds, top-2 margin collapse (delta 0.060), soft
     tier at 0.85 x the hard thresholds.
  Outputs (under --outdir):
       results/structural_scan_components.csv           per-candidate SCS components (D4)
       results/model_comparison_table5.csv              per-family structural bests (Eq. 19; structural
                                                        column of Table 8)
       results/model_comparison_pareto.csv              Pareto front of the per-family bests (Eval_SCS, acceptance)
     final stage only:
       results/Verb List Classified by Model.csv        the classified bank (repo column layout)
       results/BT4_structural_best/BT4_PROD_TUNING_<winner>_<ts>.csv   (notebook-native name)
       results/BT4_structural_best/artifacts/<winner>_<ts>/            winner bundle
           model.pkl, calibration_cdfs.json, thresholds.json, row_weights.npy,
           S_core_calibrated.npy, S_eval_calibrated.npy, metadata.json, bt4_scorer.py
       results/winner_and_constants.json                winner, selection, constants, per-family bests
                                                        (key table5_model_comparison)

CONSTANTS (name = value: role; anything not listed is a literal inside a verbatim function)
  RNG_SEED = 13                 random_state of every seeded estimator in MODEL_GRID, of the OOF
                                KFold, of the PCA in front of LDA/QDA and of the bootstrap in
                                core_targets_from_labels
  USE_ROW_WEIGHTS = True        build row weights from the core frequencies (ROW WEIGHTS)
  USE_OOF = True                calibrate on out-of-fold core scores (in-sample only if an OOF
                                fit fails, which is printed as [warn])
  N_SPLITS_OOF = 5              folds of the OOF KFold (shuffle=True, random_state=RNG_SEED)
  P_GRID = [20, 25, ..., 60]    percentiles of the threshold scan
  MIN_ACCEPT = 0.01             a setting whose candidate acceptance is below this is not
                                admissible
  SCS weights (SCS defaults)    wD = 1.0, wM = 1.0, wT = 0.8, wC = 0.3, wE = 0.05; empty_min = 10
                                (a level with fewer than 10 entries counts as empty in E)
  SCS target_ratio = 1.51,      presentation only: they set the "overlap" and "acceptance" parts
  acc_window = (0.30, 0.85)     that SCS returns next to the score; they never enter the score
  _coherence_per_class          C = 0.7 * mean((silhouette + 1) / 2) + 0.3 * mean(1 / (1 + DB))
                                over the six one-vs-rest splits (inner weights 0.7 and 0.3)
  make_estimator (LDA, QDA)     whitened PCA (random_state=RNG_SEED) in front of the estimator,
                                n_components = _pca_dim_for_lda_qda(X, n_classes=2) = 2
  _make_calibrated_svc          SVC(kernel="linear", class_weight="balanced") with sigmoid
                                calibration, CalibratedClassifierCV(cv=3); C from MODEL_GRID
  SCS_EMPTY_LEVEL_PEN = 0.05    second subtraction of E in the one-vs-rest objective
                                (tune_ovr_model); E = 0 at every admissible one-vs-rest setting,
                                so it never changes a value used for selection
  core_targets_from_labels      bootstrap = 400 resamples, margin_acc = margin_enac = 0.08: the
                                data-driven acceptance band (ACC_BAND) and entries-per-accepted
                                band (ENAC_BAND); PEN_ACC = 5.0, PEN_ENAC = 2.5: quadratic penalty
                                coefficients of Eq. 19; SCS_TIE_EPS = 0.02 is still returned but
                                unused since round 3: it was the raw-SCS tie window of the
                                notebook's one-vs-rest pick within a setting (replaced in round 2,
                                D16) and of the Legacy kNN pick of the notebook and round 2 (D11)
  ACC_BAND_CLAMP = (0.25, 0.85) clamp applied to ACC_BAND after the bootstrap
  ADEQUACY_FLOOR = 0.5          selection: a family whose structural best has PenalizedObj below
                                this is not considered (equal to select_model.ADEQUACY_FLOOR;
                                checked at run time)
  LEGACY_KNN_GRID               Legacy kNN vote grid: K in (5, 7, 9, 11, 13, 15), cosine
                                threshold in (0.12, ..., 0.32), vote cutoff in (6, ..., 32);
                                unweighted and frequency-weighted variants
  MODEL_GRID                    the 11 one-vs-rest families and their 27 settings (verbatim)
  PREFER_PARITY_THRESH = True   export uses the tuning thresholds (D6)
  PREFER_PARITY_SCORES = True   export uses the tuning-calibrated candidate scores (D6)
  APPLY_TOP2_MARGIN = True      apply the top-2 margin collapse at export
  TOP2_MARGIN_DEFAULT = 0.060   top-2 margin delta: a multi-label verb keeps only its top level
                                when that level's calibrated score leads the second by at least
                                0.060 (D14)
  SUGGEST_ALPHA = 0.85          soft tier: a level is suggested at 0.85 x its hard threshold
  The supervised protocol (CV MultilabelStratifiedKFold seeds 13 to 17, inner KFold seed 13,
  held-out outer seeds 13 to 17) is fixed in supervised_eval.py; the selection rule, its switch
  SELECTION_RULE and its eligibility definition are in select_model.py.

ROW WEIGHTS (unweighted fit, weighted thresholds)
  The notebook calls OneVsRestClassifier.fit(X, Y, sample_weight=row_w) and falls back to the
  unweighted fit if that call raises. scikit-learn 1.9.0 (requirements.txt) rejects sample_weight
  (ValueError; releases before metadata routing raised TypeError), so EVERY one-vs-rest model,
  inside the OOF folds and on the full core, is fit UNWEIGHTED, as in the published run. The row
  weights reach only the weighted threshold percentiles (thresholds_from_percentile_weighted).
  _fit_ovr keeps the notebook's attempt so the code path and the results are unchanged, counts
  the outcomes in ROW_WEIGHT_FIT, prints them and writes them to winner_and_constants.json
  ("row_weight_fit"); a weighted fit that succeeds is reported as a warning, because it would
  change the models relative to the published run.

============================= DEVIATIONS ======================================
Everything below is copied verbatim from BT3 unless listed here. Deviations:

D1. I/O adaptation: core labels/freq come from --core-csv and
    candidates from --candidates instead of in-notebook Python lists / "newest CSV
    in results/". The CSV loader is the notebook's own fallback branch of
    load_core_from_memory_or_results() (coerce_bloom_columns / find_verb_col /
    groupby("verb_norm").max(), freq = groupby max of weight|freq|frequency),
    copied verbatim. The notebook's primary path derived freq as the max
    per-level WEIGHT of each verb; supplying that same value in the 'freq' column
    reproduces it exactly. A long-format core CSV (verb, level, freq) is accepted
    and pivoted to the wide layout before the verbatim loader logic runs.
    Candidate .txt gives no Kn..Ev trace columns, so the
    trailing trace columns of the export CSV are omitted (the notebook code is
    conditional on their presence; same code path, different input).
D2. All relative paths ("results/...") are rooted under --outdir instead of the
    notebook's CWD. Embedding cache: <outdir>/results/_embed_cache (notebook:
    results/_embed_cache).
D3. sklearn-version compat shims (behavior-preserving):
    (a) CalibratedClassifierCV(base_estimator=...) was renamed to estimator= in
        sklearn>=1.2 (removed >=1.4). _make_calibrated_svc() tries base_estimator
        first, falls back to estimator. Hyperparameters unchanged.
    (b) The notebook wraps OneVsRestClassifier.fit(X, Y, sample_weight=row_w) in
        try/except TypeError with an unweighted ovr.fit(X, Y) fallback. scikit-learn
        1.9.0 raises ValueError for the same call (metadata routing is off), so
        _fit_ovr catches (TypeError, ValueError): the weighted fit is attempted and
        the model is fit unweighted, which is the notebook's effective behavior
        (ROW WEIGHTS above).
D4. Instrumentation only: tune_ovr_model() and tune_legacy_knn() additionally
    append every evaluated (model, params, percentile) candidate's StructuralStats
    components (D, M, T, C, E, D_bar, W_bar, rho, acceptance, entry_rate,
    en_over_ac, SCS, PenalizedObj, thresholds) to a global SCAN_LOG that is saved
    to results/structural_scan_components.csv.
D5. --mock-embeddings replaces the SentenceTransformer with a deterministic
    per-word random unit vector (dry-run/testing only; never use for real runs).
D6. The committed notebook contains no cell that computes winner_thresholds_oof /
    "thresholds_oof_aligned" (the saved artifact's extra key must come from a cell
    removed before commit). Production in the notebook used the tuning thresholds
    anyway (PREFER_PARITY_THRESH=True), which this script reproduces; the bundle's
    thresholds.json therefore contains only "thresholds_tuning". Likewise
    S_eval_winner_tuning is undefined in the committed notebook, so scoring uses
    S_eval_winner (the notebook's own "CURRENT" fallback branch).
D7. The notebook imports spacy/wordnet and builds *_words_spacy docs in cells 0/4;
    none of that feeds the Algorithm-1 pipeline and it is omitted here.
D8. If the selected family were the Legacy kNN voter, the notebook's save cell would
    raise (winner_model is None -> "cannot be serialized"); the guard is kept, but
    Legacy kNN is not eligible for selection (D12), so it cannot be selected.
D9. display(...) calls replaced by print(...to_string()); prints kept.
D10. The notebook's params.json helper cell (set_active_params etc., cell 3) is
     legacy kNN bookkeeping unused by the pipeline and is omitted.
D11. Legacy kNN pick (round 3 bug fix). tune_legacy_knn ranks the admissible vote
     configurations by the penalized objective, ties by raw SCS, then acceptance,
     exactly like tune_ovr_model; the frequency-weighted variant replaces the
     unweighted one only if its (objective, SCS, acceptance) key is strictly larger
     (notebook rule, unchanged). The notebook and round 2 kept the configuration with
     the largest RAW SCS within SCS_TIE_EPS = 0.02, which ignored the guardrail
     penalty: the Legacy row of the round-2 model_comparison_table5.csv was
     frequency-weighted K=5/th=0.32/c=32 (acceptance 0.192, PenalizedObj -1.197);
     with the fix it is unweighted K=15/th=0.28/c=6 (PenalizedObj 1.0037). Legacy kNN
     is a structural reference row only (D12).
D12. Model selection (round 3). The notebook selected the largest penalized
     objective (structure only); round 2 added a two-stage joint rule fed by CV and
     held-out results of fixed default configurations at P50 (cv_metrics.py,
     methodology_upgrades.py). Round 3 delegates the selection to select_model.py:
     select_production_family() builds the menu from this run's per-family
     structural bests, attaches the supervised criteria of exactly those
     configurations and percentiles (results/selection/supervised_grid_per_seed.csv,
     supervised_eval.py) and applies select_model.SELECTION_RULE. Eligibility,
     adequacy floor, criteria and aggregation are defined there (select_model.py
     header). The run stops with an error if the supervised results are missing or
     do not cover a family the rule considers, if the core differs from the core of
     the supervised evaluation, if select_model's own re-derivation from the scan
     file disagrees with this run, or if results/selection/selection_table.csv
     selects a different family or score. The round-2 two-stage form on the same
     inputs is reported in winner_and_constants.json, not used.
D13. Stages: the scan outputs (structural_scan_components.csv,
     model_comparison_table5.csv, model_comparison_pareto.csv) are written before the
     selection, and --scan-only stops there. In winner_and_constants.json, bundle_dir is
     a POSIX path relative to --outdir, and the keys "selection", "reproduction_checks"
     (D15), "row_weight_fit" and "versions" are added.
D14. The committed notebook lacks the cell that defined the post-processing
     constants (its fallback top-2 delta was 0.12). TOP2_MARGIN_DEFAULT restores the
     published value 0.060 (paper Section 2.3.5), which produced the delivered bank
     and production_model/.
D15. Reproduction checks (reporting only): after the export, the winner bundle is compared
     with the packaged production_model/ (model.pkl, calibration_cdfs.json, thresholds.json,
     row_weights.npy, bt4_scorer.py: bytes, else the largest numerical difference) and the
     exported bank with the repository's results/"Verb List Classified by Model.csv" as it was
     before the export. The result is printed and stored under "reproduction_checks" in
     winner_and_constants.json; production_model/ is never written.
D16. One-vs-rest pick within a hyperparameter setting (round 2 bug fix). The notebook's
     tune_ovr_model kept the percentiles whose raw SCS lies within SCS_TIE_EPS = 0.02 of
     the largest raw SCS and took the largest penalized objective among them (ties by
     acceptance). That shortlist ignored the guardrail penalty, so tune_ovr_model ranks
     every admissible percentile by the penalized objective directly, ties by raw SCS,
     then acceptance (BUGFIX comment in tune_ovr_model). The comparison across settings
     is the notebook's.
===============================================================================
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings
import zlib
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

import numpy.linalg as npl

# ------------------------------
# Toggles & constants  (verbatim from BT3 cell 9 unless noted; roles in the CONSTANTS block above)
# ------------------------------
RNG_SEED = 13
USE_ROW_WEIGHTS = True
USE_OOF = True          # out-of-fold calibration
N_SPLITS_OOF = 5        # KFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
SCS_EMPTY_LEVEL_PEN = 0.05  # second empty-level subtraction in the one-vs-rest objective

# Percentile grid (verbatim): P_GRID = list(range(20, 61, 5))
P_GRID = list(range(20, 61, 5))          # [20, 25, 30, 35, 40, 45, 50, 55, 60]

# Post-processing (BT3 cell 11)
PREFER_PARITY_THRESH = True      # True -> tuning thresholds; False -> use OOF-aligned if available
PREFER_PARITY_SCORES = True      # True -> use tuning-calibrated scores if present
APPLY_TOP2_MARGIN    = True      # Post-process: collapse to single label if (top1 - top2) >= margin
TOP2_MARGIN_DEFAULT  = 0.060     # published production delta (D14)
SUGGEST_ALPHA        = 0.85      # "Soft suggestion" threshold = alpha * hard threshold (per class)

MIN_ACCEPT = 0.01                # min_accept guardrail used for every tuner call in BT3
ACC_BAND_CLAMP = (0.25, 0.85)    # clamp of the acceptance band (values verbatim from BT3 cell 9)
ADEQUACY_FLOOR = 0.5             # selection floor on PenalizedObj (= select_model.ADEQUACY_FLOOR, D12)

# Legacy kNN vote grid (verbatim defaults of tune_legacy_knn)
LEGACY_KNN_GRID = {
    "Ks": (5, 7, 9, 11, 13, 15),
    "Th": (0.12, 0.14, 0.16, 0.20, 0.24, 0.28, 0.32),
    "Ct": (6, 8, 10, 12, 16, 20, 24, 28, 32),
}

LEVELS = ["Kn", "Cm", "Ap", "An", "Sn", "Ev"]
LVL2IDX = {lv: i for i, lv in enumerate(LEVELS)}
LEVELS_LOWER = ["knowledge", "comprehension", "application", "analysis", "synthesis", "evaluation"]
LEVEL_NAMES = {"Kn": "knowledge", "Cm": "comprehension", "Ap": "application",
               "An": "analysis", "Sn": "synthesis", "Ev": "evaluation"}

# Warnings hygiene (verbatim)
warnings.filterwarnings("ignore", message="Variables are collinear")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Globals populated in run_pipeline() so the verbatim function bodies keep working.
X_core = None
Y_core = None
X_eval = None
core_words = None
eval_verbs = None
core_freq = None
df_eval = None
ROW_W = None
TARGETS = None
# Round-2 option (a cv_results_*.csv for the round-2 joint selection). Ignored since round 3
# (D12); run_pipeline prints a warning if it is set (BT 3 notebook compatibility).
CV_RESULTS_PATH = None
RESULTS_DIR = None
EMBED_CACHE_DIR = None

# D4: per-candidate structural-stats scan log (instrumentation only)
SCAN_LOG: List[Dict[str, Any]] = []

# Outcomes of the weighted one-vs-rest fit attempts (ROW WEIGHTS in the header)
ROW_WEIGHT_FIT = {"weighted_fit_ok": 0, "weighted_fit_fallback_unweighted": 0}

# ==============================================================================
# Embeddings  (verbatim from BT3 cell 5, cache rooted under outdir, D2;
#              mock backend, D5)
# ==============================================================================
_st = None            # SentenceTransformer, lazily created
MOCK_EMBED = False
MOCK_DIM = 768
_mock_cache: Dict[str, np.ndarray] = {}


def _mock_vec(word: str) -> np.ndarray:
    """D5: deterministic unit vector per word (dry-run only)."""
    if word in _mock_cache:
        return _mock_cache[word]
    seed = zlib.crc32(word.encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(MOCK_DIM).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-12)
    _mock_cache[word] = v
    return v


def _get_st():
    global _st
    if _st is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError("Please install: pip install sentence-transformers") from e
        # Deterministic, normalized embeddings (verbatim)
        _st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return _st


def _embed_once(text: str) -> np.ndarray:
    if MOCK_EMBED:
        return _mock_vec(text)
    v = _get_st().encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
    return v.astype(np.float32, copy=False)


# In-memory embedding store loaded from a consolidated all_embeddings.npz.
# It holds the vectors of every verb of the committed core and candidate lists, so a run reads
# one file instead of about 2,000 small .npy files; any other word falls back to the per-word
# cache (D2).
_NPZ_CACHE: dict = {}

def _load_npz_cache():
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_embeddings.npz")
    if os.path.exists(fp):
        z = np.load(fp)
        _NPZ_CACHE.update(zip([str(w) for w in z["words"]], z["vectors"].astype(np.float32)))
        print(f"[EMBED] loaded {len(_NPZ_CACHE)} vectors from all_embeddings.npz")


def embed_vec(word: str) -> np.ndarray:
    key = (word or "").strip().lower()
    if not key:
        key = "empty"
    if MOCK_EMBED:
        return _mock_vec(key)
    if key in _NPZ_CACHE:
        return _NPZ_CACHE[key]
    fp = os.path.join(EMBED_CACHE_DIR, f"{key.replace(' ', '_')}.npy")
    if os.path.exists(fp):
        return np.load(fp)
    v = _embed_once(key)
    np.save(fp, v)
    return v


def embed_batch(words: List[str]) -> np.ndarray:
    if MOCK_EMBED:
        return np.stack([_mock_vec((w or "").strip().lower() or "empty") for w in words], axis=0)
    paths, miss_idx, miss_words, vecs = [], [], [], []
    for i, w in enumerate(words):
        key = (w or "").strip().lower() or "empty"
        fp = os.path.join(EMBED_CACHE_DIR, f"{key.replace(' ', '_')}.npy")
        paths.append(fp)
        if key in _NPZ_CACHE:
            vecs.append(_NPZ_CACHE[key])
        elif os.path.exists(fp):
            vecs.append(np.load(fp))
        else:
            vecs.append(None); miss_idx.append(i); miss_words.append(key)
    if miss_words:
        V = _get_st().encode(miss_words, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        for j, i in enumerate(miss_idx):
            vecs[i] = V[j]; np.save(paths[i], V[j])
    return np.stack(vecs, axis=0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ==============================================================================
# Input loading  (verbatim helpers + the notebook's own CSV fallback branch, D1)
# ==============================================================================
def coerce_bloom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not all(c in df.columns for c in LEVELS):
        # allow lower-case names
        alt = dict(zip(LEVELS_LOWER, LEVELS))
        for lo, hi in alt.items():
            if lo in df.columns and hi not in df.columns:
                df[hi] = df[lo]
    if not all(c in df.columns for c in LEVELS):
        raise RuntimeError("Need 6 Bloom columns (Kn..Ev) or lowercased equivalents.")
    return df


def find_verb_col(df: pd.DataFrame) -> str:
    for c in ["verb", "Verb", "word", "token", "WORD", "VERB"]:
        if c in df.columns:
            return c
    raise RuntimeError("No verb/word/token column found.")


def _pivot_long_core(df: pd.DataFrame) -> pd.DataFrame:
    """D1: accept a long-format core CSV (verb, level, freq; one row per
    verb-level pair) by pivoting to the notebook's wide multi-hot layout.
    freq per verb = max across its rows (the notebook's freq dict likewise took
    the max per-level weight of each verb)."""
    vcol = find_verb_col(df)
    lvl_map = {**{lv: lv for lv in LEVELS},
               **{lv.lower(): lv for lv in LEVELS},
               **dict(zip(LEVELS_LOWER, LEVELS))}
    wide = df.copy()
    wide["_lv"] = wide["level"].astype(str).str.strip().map(
        lambda s: lvl_map.get(s, lvl_map.get(s.lower())))
    if wide["_lv"].isna().any():
        bad = sorted(set(wide.loc[wide["_lv"].isna(), "level"].astype(str)))
        raise RuntimeError(f"Unrecognized level values in core CSV: {bad}")
    out = wide.groupby(wide[vcol].astype(str).str.lower().str.strip())
    rows = []
    fcol = next((c for c in ["weight", "freq", "frequency"] if c in wide.columns), None)
    for verb, g in out:
        row = {vcol: verb}
        for lv in LEVELS:
            row[lv] = int((g["_lv"] == lv).any())
        if fcol is not None:
            row["freq"] = g[fcol].max()
        rows.append(row)
    return pd.DataFrame(rows)


def load_core_from_csv(path: str) -> Tuple[List[str], np.ndarray, Optional[Dict[str, int]]]:
    """Verbatim CSV fallback branch of BT3's load_core_from_memory_or_results().
    (Long-format input is first pivoted to the wide layout, D1.)"""
    df = pd.read_json(path) if str(path).endswith(".json") else pd.read_csv(path)
    if ("level" in df.columns) and not any(c in df.columns for c in LEVELS + LEVELS_LOWER):
        df = _pivot_long_core(df)
    df = coerce_bloom_columns(df)
    vcol = find_verb_col(df)
    df["verb_norm"] = df[vcol].astype(str).str.lower().str.strip()
    agg = df.groupby("verb_norm")[LEVELS].max().reset_index()
    core_words = agg["verb_norm"].tolist()
    Y = agg[LEVELS].astype(int).to_numpy()
    # Optional weights: look for a 'weight' or per-level freq column
    freq = None
    for c in ["weight", "freq", "frequency"]:
        if c in df.columns:
            freq = df.groupby("verb_norm")[c].max().to_dict()
            break
    return core_words, Y, freq


def load_candidates(path: str) -> Tuple[List[str], pd.DataFrame]:
    """D1: candidates from a txt (one verb/line) or a CSV with a verb column
    (optional Kn..Ev columns kept for traceability, as in the notebook's df_eval)."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        vcol = find_verb_col(df)
        verbs = df[vcol].astype(str).str.lower().str.strip().tolist()
        return verbs, df
    verbs = [l.strip().lower() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return verbs, pd.DataFrame({"verb": verbs})


# ==============================================================================
# Structural metrics + SCS  (verbatim from BT3 cell 8)
# ==============================================================================
def acceptance_entry(Y_bin: np.ndarray) -> dict:
    """
    Returns:
      acceptance = (# items with >=1 label) / N
      entry_rate = average #labels per item
      en_over_ac = entry_rate / acceptance  (~ avg #domains per accepted item)
      per_level  = counts per level (length 6)
    """
    n = Y_bin.shape[0]
    lbls = Y_bin.sum(axis=1)
    acc = float((lbls > 0).sum()) / max(n, 1)
    en = float(lbls.mean()) if n else 0.0
    ratio = (en / acc) if acc > 0 else 0.0
    per_level = Y_bin.sum(axis=0).astype(int).tolist()
    return dict(acceptance=acc, entry_rate=en, en_over_ac=ratio, per_level=per_level)


def _l2_normalize(M: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    nrm = np.linalg.norm(M, axis=axis, keepdims=True)
    nrm = np.where(nrm < eps, 1.0, nrm)
    return M / nrm


def _safe_spearman(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if ys.size == 0 or not np.all(np.isfinite(ys)):
        return 0.0
    # constant series -> undefined rho; treat as no trend (0.0)
    if len(set(np.round(ys, 12))) <= 1:
        return 0.0
    try:
        rho, _ = spearmanr(xs, ys)
        return float(rho) if np.isfinite(rho) else 0.0
    except Exception:
        return 0.0


def vector_centroid_summary(
    Y_bin: np.ndarray,
    words: List[str],
    V_all: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, dict, float, list, list]:
    """
    Build per-level centroids and summarize gap-wise distances & monotonicity.
    (See BT3 cell 8 for the full derivation; copied verbatim.)
    """
    n_items = Y_bin.shape[0]
    # Embeddings for ALL words (compute once per call)
    if V_all is None:
        V_all = embed_batch(words).astype(np.float64)
    V_all = _l2_normalize(V_all, axis=1)

    # Indices of accepted items per level (multi-label allowed)
    assign = [np.where(Y_bin[:, j] == 1)[0] for j in range(6)]
    counts = [int(len(idx)) for idx in assign]

    # Centroids per level
    d = V_all.shape[1] if n_items > 0 else len(embed_vec("probe"))
    C = np.zeros((6, d), dtype=np.float64)
    for j in range(6):
        idx = assign[j]
        if len(idx) == 0:
            continue
        c = V_all[idx].mean(axis=0)
        C[j] = _l2_normalize(c.reshape(1, -1), axis=1)[0]

    # Cosine distance matrix between centroids
    Dmat = np.zeros((6, 6), dtype=float)
    for i in range(6):
        for j in range(i + 1, 6):
            sim = float(np.dot(C[i], C[j]))
            dist = 1.0 - sim  # in [0,2]
            Dmat[i, j] = Dmat[j, i] = max(0.0, min(2.0, dist))

    # Summaries by gap k and monotonic trend
    D_summ, means_by_k = {}, []
    for k in range(1, 6):
        vals = [Dmat[i, i + k] for i in range(0, 6 - k)]
        m = float(np.mean(vals)) if len(vals) else float("nan")
        D_summ[k] = (m, float(np.max(vals)) if len(vals) else float("nan"),
                        float(np.min(vals)) if len(vals) else float("nan"))
        means_by_k.append(0.0 if math.isnan(m) else m)
    ks = np.arange(1, 6, dtype=float)
    rho = _safe_spearman(ks, means_by_k)
    return C, Dmat, D_summ, float(rho), counts, assign


def mutual_influence(Y_bin: np.ndarray):
    """
    Jaccard overlap between accepted sets:
      J(i,j) = |S_i ∩ S_j| / |S_i ∪ S_j|,  bounded in [0,1]
    We summarize gap-wise means:
      M_k = mean_i J(i, i+k)
    """
    sets = [set(np.where(Y_bin[:, j] == 1)[0].tolist()) for j in range(6)]
    J = np.zeros((6, 6), dtype=float)
    for i in range(6):
        for j in range(6):
            if i == j:
                J[i, j] = 1.0
            else:
                u = len(sets[i] | sets[j]); inter = len(sets[i] & sets[j])
                J[i, j] = (inter / u) if u > 0 else 0.0
    summ = {}
    for k in range(1, 6):
        vals = [J[i, i + k] for i in range(0, 6 - k)]
        summ[k] = (float(np.mean(vals)), float(np.max(vals)), float(np.min(vals))) if vals else (float('nan'),) * 3
    return J, summ


def _coherence_per_class(V_all: np.ndarray, Y_bin: np.ndarray, use_cosine: bool = True) -> Tuple[float, dict]:
    """
    For each level l, one-vs-rest Silhouette + Davies-Bouldin, normalized to [0,1]:
      C = 0.7*mean[(Sil+1)/2] + 0.3*mean[1/(1+DB)]
    (Copied verbatim.)
    """
    # Normalize embeddings if using cosine
    V = _l2_normalize(V_all, axis=1) if use_cosine else V_all

    S_norm, D_norm = [], []
    details = {"sil": [], "db": [], "S_tilde": [], "D_tilde": [],
               "valid_idx": [], "skipped": []}

    n = V.shape[0]
    metric = "cosine" if use_cosine else "euclidean"

    for lvl in range(6):
        y = (Y_bin[:, lvl] == 1).astype(int)
        n_pos = int(y.sum()); n_neg = int(n - n_pos)

        # Need at least 2 samples in each cluster for silhouette to be valid
        if n_pos < 2 or n_neg < 2:
            details["skipped"].append(lvl)
            continue

        # Silhouette
        try:
            sil = float(silhouette_score(V, y, metric=metric))
            Sil_t = (sil + 1.0) / 2.0
        except Exception:
            sil, Sil_t = float("nan"), float("nan")

        # Davies-Bouldin
        try:
            db = float(davies_bouldin_score(V, y))
            DB_t = 1.0 / (1.0 + db)
        except Exception:
            db, DB_t = float("nan"), float("nan")

        details["sil"].append(sil)
        details["db"].append(db)
        details["S_tilde"].append(Sil_t)
        details["D_tilde"].append(DB_t)
        details["valid_idx"].append(lvl)

        if not np.isnan(Sil_t): S_norm.append(Sil_t)
        if not np.isnan(DB_t):  D_norm.append(DB_t)

    if len(S_norm) == 0 and len(D_norm) == 0:
        details.update({"S_bar": float("nan"), "D_bar": float("nan"), "C": 0.0})
        return 0.0, details

    S_bar = float(np.mean(S_norm)) if len(S_norm) else 0.0
    D_bar = float(np.mean(D_norm)) if len(D_norm) else 0.0
    C = 0.7 * S_bar + 0.3 * D_bar
    details.update({"S_bar": S_bar, "D_bar": D_bar, "C": C})
    return float(C), details


def structural_eval(
    Y_bin: np.ndarray,
    words: List[str],
    V_all: Optional[np.ndarray] = None,
) -> dict:
    """Master evaluator: packages everything needed for SCS. (Verbatim.)"""
    acc = acceptance_entry(Y_bin)
    # Precompute embeddings once
    if V_all is None:
        V_all = embed_batch(words).astype(np.float64)
    V_all = _l2_normalize(V_all, axis=1)

    Cents, Dmat, D_summ, rho, counts, assign = vector_centroid_summary(Y_bin, words, V_all=V_all)
    J, J_summ = mutual_influence(Y_bin)
    C_peer, C_details = _coherence_per_class(V_all, Y_bin)

    return dict(
        acc=acc,
        centroids=Cents, V_all=V_all, assign_idx=assign,
        D_summ=D_summ, trend_rho=rho, level_counts=counts,
        M_summ=J_summ,
        C_peer=C_peer, C_details=C_details
    )


def SCS(
    out: dict,
    *,
    wD: float = 1.0, wM: float = 1.0, wT: float = 0.8, wC: float = 0.3, wE: float = 0.05,
    empty_min: int = 10,
    target_ratio: float = 1.51, acc_window=(0.30, 0.85)
):
    r"""
    Final score:
        SCS = 1*D  - 1*M  + 0.8*T  + 0.3*C  - 0.05*E
    (Component definitions as in BT3 cell 8; copied verbatim.)
    """
    eps = 1e-6

    # --- Degenerate guard: if no accepted items at all, hard-penalize ---
    counts = out.get("level_counts", [0] * 6)
    if sum(int(c) for c in counts) == 0:
        return -1e9, dict(D=0.0, M=1.0, trend=0.0, C=0.0, empty_levels=6,
                          acceptance=0.0, overlap=0.0, D_bar=0.0, W_bar=0.0, rho=0.0)

    # --- D: gap-aware separation with compactness (bounded) ---
    D_means = [out["D_summ"][k][0] for k in range(1, 6)]
    D_means = [0.0 if (m is None or not np.isfinite(m)) else float(m) for m in D_means]
    D_bar = float(np.average(D_means, weights=[1, 2, 3, 4, 5])) if np.any(np.isfinite(D_means)) else 0.0

    V = out["V_all"]; Cents = out["centroids"]; assign = out["assign_idx"]
    W_list, n_list = [], []
    for j in range(6):
        idx = assign[j]
        if len(idx) == 0:
            continue
        xj = V[idx]; cj = Cents[j].reshape(1, -1)
        Wj = float(np.mean(1.0 - np.clip(xj @ cj.T, -1.0, 1.0)))
        if np.isfinite(Wj):
            W_list.append(Wj); n_list.append(len(idx))
    W_bar = float(np.average(W_list, weights=n_list)) if n_list else 0.0

    D_term = D_bar / (D_bar + W_bar + eps)
    D_term = float(np.clip(D_term, 0.0, 1.0))

    # --- M: adjacent overlap (lower better) ---
    M_k1 = out["M_summ"][1][0]
    M_term = float(np.clip(0.0 if (M_k1 is None or not np.isfinite(M_k1)) else M_k1, 0.0, 1.0))

    # --- T: monotonic trend (rho -> [0,1]) ---
    rho = out.get("trend_rho", 0.0)
    if rho is None or not np.isfinite(rho): rho = 0.0
    T_term = float(np.clip((rho + 1.0) / 2.0, 0.0, 1.0))

    # --- C: per-class peer-reviewed coherence (0..1) ---
    C_val = out.get("C_peer", 0.0)
    if C_val is None or not np.isfinite(C_val): C_val = 0.0
    C_term = float(np.clip(C_val, 0.0, 1.0))

    # --- E: #near-empty levels (< empty_min) ---
    E = int(sum(1 for c in counts if c < int(empty_min)))

    score = (wD * D_term) - (wM * M_term) + (wT * T_term) + (wC * C_term) - (wE * E)

    # ---- Presentation-only (NOT part of score) ----
    acc = float(out["acc"].get("acceptance", 0.0) or 0.0)
    en_ac = float(out["acc"].get("en_over_ac", 0.0) or 0.0)

    overlap_term = 0.0
    if acc > 0:
        overlap_term = 1.0 - abs(en_ac - target_ratio) / max(target_ratio, 1e-6)
        overlap_term = float(np.clip(overlap_term, 0.0, 1.0))

    lo, hi = acc_window
    if acc < lo:       acc_term = max(0.0, 1.0 - (lo - acc) / max(lo, 1e-6))
    elif acc > hi:     acc_term = max(0.0, 1.0 - (acc - hi) / max(1.0 - hi, 1e-6))
    else:              acc_term = 1.0

    parts = dict(
        D=D_term, M=M_term, trend=T_term, C=C_term, empty_levels=E,
        acceptance=acc_term, overlap=overlap_term,
        D_bar=D_bar, W_bar=W_bar, rho=rho
    )
    return float(score), parts


# ==============================================================================
# Targets + penalized objective  (verbatim from BT3 cell 9)
# ==============================================================================
def core_targets_from_labels(Y_core, margin_acc=0.08, margin_enac=0.08, bootstrap=400, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = Y_core.shape[0]
    accs, enacs = [], []
    for _ in range(bootstrap):
        idx = rng.integers(0, n, n)
        Yb = Y_core[idx]
        lbls = Yb.sum(axis=1)
        acc = float((lbls > 0).mean())
        en = float(lbls.mean())
        enac = (en / acc) if acc > 0 else 0.0
        accs.append(acc); enacs.append(enac)
    acc_med, enac_med = float(np.median(accs)), float(np.median(enacs))
    ACC_BAND = (max(0.01, acc_med - margin_acc), min(0.99, acc_med + margin_acc))
    ENAC_BAND = (max(1.0, enac_med - margin_enac), enac_med + margin_enac)
    if ACC_BAND[1] <= ACC_BAND[0]: ACC_BAND = (ACC_BAND[0], ACC_BAND[0] + 1e-3)
    if ENAC_BAND[1] <= ENAC_BAND[0]: ENAC_BAND = (ENAC_BAND[0], ENAC_BAND[0] + 1e-3)
    return dict(ACC_BAND=ACC_BAND, ENAC_BAND=ENAC_BAND, SCS_TIE_EPS=0.02, PEN_ACC=5.0, PEN_ENAC=2.5)


def _penalty_from_out(out, ACC_BAND, ENAC_BAND, pen_acc=5.0, pen_enac=2.5):
    acc = float(out["acc"]["acceptance"])
    enac = float(out["acc"]["en_over_ac"])
    lo, hi = ACC_BAND
    loe, hie = ENAC_BAND
    p_acc = 0.0 if (lo <= acc <= hi) else min(abs(acc - lo), abs(acc - hi)) ** 2 * pen_acc
    p_enac = 0.0 if (loe <= enac <= hie) else min(abs(enac - loe), abs(enac - hie)) ** 2 * pen_enac
    return p_acc + p_enac


def _objective(scs, out, TARGETS):
    pen = _penalty_from_out(out, TARGETS["ACC_BAND"], TARGETS["ENAC_BAND"],
                            TARGETS["PEN_ACC"], TARGETS["PEN_ENAC"])
    return float(scs) - float(pen)


def _pareto_front(df, cols=[("Eval_SCS", False), ("Eval_Acceptance", False)]):
    X = df.copy()
    for c, asc in cols:
        if asc: X[c] = -X[c]
    keep, vals = [], X[[c for c, _ in cols]].to_numpy()
    for i in range(len(X)):
        v = vals[i]; dominated = False
        for j in range(len(X)):
            if i == j: continue
            w = vals[j]
            if np.all(w >= v) and np.any(w > v):
                dominated = True; break
        if not dominated: keep.append(i)
    return df.iloc[keep].copy()


# ==============================================================================
# Row weights  (verbatim: w = 1 + log1p(freq), clip [0.5, 3.0], / mean)
# ==============================================================================
def _build_core_row_weights() -> np.ndarray:
    n = X_core.shape[0]
    w = np.ones(n, dtype=float)
    if core_freq is not None:
        raw = np.array([float(core_freq.get(core_words[i], 0.0)) for i in range(n)], dtype=float)
        w = 1.0 + np.log1p(raw)
    w = np.nan_to_num(w, nan=1.0, posinf=3.0, neginf=0.5)
    w = np.clip(w, 0.5, 3.0)
    m = w.mean()
    if m > 0: w = w / m
    return w


# ==============================================================================
# Rank helpers for LDA/QDA  (verbatim)
# ==============================================================================
def _effective_rank(A: np.ndarray, eps: float = 1e-10) -> int:
    try:
        s = npl.svd(A, full_matrices=False, compute_uv=False)
        if s.size == 0: return 0
        tol = eps * max(A.shape) * (s[0] if np.isfinite(s[0]) else 1.0)
        return int(np.sum(s > tol))
    except Exception:
        return int(max(0, min(A.shape[0] - 1, A.shape[1])))


def _pca_dim_for_lda_qda(X: np.ndarray, n_classes: int = 2, hard_cap: int = 256) -> int:
    r = _effective_rank(X)
    if r <= 2: return 0
    return int(max(2, min(hard_cap, r, n_classes - 1 if n_classes > 1 else r)))


# ==============================================================================
# Scoring + robust CDF calibration  (verbatim)
# ==============================================================================
def scores_ovr(clf, X) -> np.ndarray:
    n = X.shape[0]
    if hasattr(clf, "decision_function"):
        try:
            S = np.asarray(clf.decision_function(X), dtype=float)
            return S.reshape(n, -1)
        except Exception:
            pass
    ests = getattr(clf, "estimators_", None)
    if not ests or len(ests) != 6:
        raise RuntimeError("OvR classifier is not fitted or missing .estimators_ (expected 6).")
    S_all = np.zeros((n, 6), dtype=float)
    for j, est in enumerate(ests):
        model = est.steps[-1][1] if hasattr(est, "steps") else est
        if hasattr(model, "decision_function"):
            s = est.decision_function(X); s = np.asarray(s, dtype=float)
            S_all[:, j] = s.ravel() if s.ndim > 1 else s
            continue
        if hasattr(model, "predict_proba"):
            P = np.asarray(est.predict_proba(X))
            classes_ = getattr(model, "classes_", getattr(est, "classes_", np.array([0, 1])))
            if P.ndim == 1 or (P.ndim == 2 and P.shape[1] == 1):
                only_cls = int(classes_[0]) if classes_.size else 0
                S_all[:, j] = 1.0 if only_cls == 1 else 0.0
            else:
                try:
                    idx_pos = int(np.where(classes_ == 1)[0][0])
                    S_all[:, j] = P[:, idx_pos]
                except Exception:
                    S_all[:, j] = P.max(axis=1)
            continue
        yhat = est.predict(X)
        S_all[:, j] = np.asarray(yhat, dtype=float)
    return S_all


def _fit_cdf_per_class(S_core: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    # robust CDF even for constant/NaN/Inf
    S_core = np.nan_to_num(S_core, nan=0.0, posinf=0.0, neginf=0.0)
    cdfs = []
    for j in range(S_core.shape[1]):
        x = np.asarray(S_core[:, j], dtype=float)
        mask = np.isfinite(x)
        if not mask.any():
            xs = np.array([0.0]); cdf = np.array([1.0]); cdfs.append((xs, cdf)); continue
        xs = np.sort(x[mask])
        if xs.size == 0:
            xs = np.array([0.0]); cdf = np.array([1.0]); cdfs.append((xs, cdf)); continue
        if np.allclose(xs[0], xs[-1]):
            v = xs[0]
            xs = np.array([v - 1e-9, v + 1e-9]); cdf = np.array([0.0, 1.0])
            cdfs.append((xs, cdf)); continue
        ranks = np.arange(1, len(xs) + 1, dtype=float)
        cdf = ranks / (len(xs) + 1.0)
        cdfs.append((xs, cdf))
    return cdfs


def _apply_cdf_per_class(S: np.ndarray, cdfs: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    S01 = np.zeros_like(S, dtype=float)
    for j in range(S.shape[1]):
        xs, cdf = cdfs[j]
        S01[:, j] = np.interp(S[:, j], xs, cdf, left=cdf[0], right=cdf[-1])
    return np.clip(S01, 0.0, 1.0)


# ==============================================================================
# Weighted thresholds & evaluation  (verbatim)
# ==============================================================================
def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    x = np.asarray(x, float); w = np.asarray(w, float)
    if x.size == 0: return 1.0
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    if x.size == 0: return 1.0
    ix = np.argsort(x); x, w = x[ix], w[ix]
    cw = np.cumsum(w)
    if cw[-1] <= 0: return float(x[-1])
    t = q * cw[-1]
    k = np.searchsorted(cw, t, side="right")
    k = min(max(k, 0), len(x) - 1)
    return float(x[k])


def thresholds_from_percentile_weighted(S_core01: np.ndarray, Y_core: np.ndarray, p: int, row_w: np.ndarray) -> np.ndarray:
    thr = np.zeros(6, dtype=float)
    q = p / 100.0
    for j in range(6):
        pos = np.where(Y_core[:, j] == 1)[0]
        if len(pos):
            thr[j] = _weighted_quantile(S_core01[pos, j], row_w[pos], q)
        else:
            thr[j] = _weighted_quantile(S_core01[:, j], row_w, q)
        thr[j] = float(np.clip(thr[j], 0.0, 1.0))
    return thr


def evaluate_with_thresholds(S_core01, S_eval01, thr, verbs_core, verbs_eval):
    Yb_core = (S_core01 >= thr.reshape(1, -1)).astype(int)
    Yb_eval = (S_eval01 >= thr.reshape(1, -1)).astype(int)
    out_c = structural_eval(Yb_core, verbs_core)
    out_e = structural_eval(Yb_eval, verbs_eval)
    scs_c, parts_c = SCS(out_c)
    scs_e, parts_e = SCS(out_e)
    return (Yb_core, Yb_eval, out_c, out_e, scs_c, scs_e, parts_c, parts_e)


# ==============================================================================
# Model-specific build + fit wrappers + OOF  (verbatim; D3 compat shims)
# ==============================================================================
def make_estimator(label: str, base, params: dict, X_fit: np.ndarray, y_fit: np.ndarray):
    if label in ("LDA", "QDA"):
        n_comp = _pca_dim_for_lda_qda(X_fit, n_classes=2, hard_cap=256)
        if n_comp < 2:
            raise RuntimeError(f"{label}: insufficient rank ({n_comp}) after analysis.")
        pca = PCA(n_components=n_comp, whiten=True, random_state=RNG_SEED)
        clf = base(**params)
        return Pipeline([("pca", pca), ("clf", clf)])
    return base(**params)


def _fit_ovr(ovr, X, Y, row_w):
    """Fit a OneVsRestClassifier as the notebook did (D3b, ROW WEIGHTS in the header): the
    weighted fit is attempted and, when the installed scikit-learn rejects sample_weight
    (1.9.0: ValueError; releases before metadata routing: TypeError), the model is fit
    UNWEIGHTED. Under the pinned scikit-learn every model is therefore fit unweighted and the
    row weights reach only the weighted threshold percentiles. Outcomes go to ROW_WEIGHT_FIT."""
    if row_w is None:
        ovr.fit(X, Y)
        return ovr
    try:
        ovr.fit(X, Y, sample_weight=row_w)
        ROW_WEIGHT_FIT["weighted_fit_ok"] += 1
    except (TypeError, ValueError):   # D3b: the notebook catches TypeError only
        ovr.fit(X, Y)
        ROW_WEIGHT_FIT["weighted_fit_fallback_unweighted"] += 1
    return ovr


def _row_weight_fit_report() -> Dict[str, Any]:
    d = dict(ROW_WEIGHT_FIT)
    print(f"[ROW-WEIGHTS] one-vs-rest fits: weighted fit accepted {d['weighted_fit_ok']}, "
          f"rejected by scikit-learn and fit unweighted {d['weighted_fit_fallback_unweighted']}; "
          "row weights enter the threshold percentiles")
    if d["weighted_fit_ok"]:
        print("[ROW-WEIGHTS] WARNING: this scikit-learn accepted sample_weight, so row weights reached "
              "the model fit. The published models were fit unweighted (scikit-learn 1.9.0); results "
              "will differ from the published run.")
    d["note"] = ("OneVsRestClassifier.fit(X, Y, sample_weight=w) is attempted as in the notebook; "
                 "scikit-learn 1.9.0 rejects it (ValueError) and the model is fit unweighted. "
                 "Row weights enter only the weighted threshold percentiles.")
    return d


def _fit_ovr_core(label, base, params, X, Y, row_w):
    est = make_estimator(label, base, params, X, Y)
    return _fit_ovr(OneVsRestClassifier(est), X, Y, row_w)


def _scores_in_sample(label, base, params, X_core, Y_core, X_eval, row_w):
    clf = _fit_ovr_core(label, base, params, X_core, Y_core, row_w)
    S_core_raw = scores_ovr(clf, X_core)
    S_eval_raw = scores_ovr(clf, X_eval)
    return S_core_raw, S_eval_raw, clf


def _scores_oof(label, base, params, X, Y, row_w, n_splits=5, seed=RNG_SEED):
    # Robust OOF; if any fold fails, raise to allow fallback
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    S_oof = np.zeros((X.shape[0], Y.shape[1]), dtype=float)
    for tr, va in kf.split(X):
        est = make_estimator(label, base, dict(params), X[tr], Y[tr])
        ovr = _fit_ovr(OneVsRestClassifier(est), X[tr], Y[tr],
                       row_w[tr] if row_w is not None else None)
        S_oof[va] = scores_ovr(ovr, X[va])
    clf_full = _fit_ovr_core(label, base, params, X, Y, row_w)
    return S_oof, clf_full


def _get_scores(label, base, params):
    # choose OOF with fallback to in-sample (prevents "no calibration found" situations)
    row_w = ROW_W if USE_ROW_WEIGHTS else None
    if USE_OOF:
        try:
            S_core_raw, clf = _scores_oof(label, base, params, X_core, Y_core, row_w, n_splits=N_SPLITS_OOF, seed=RNG_SEED)
            S_eval_raw = scores_ovr(clf, X_eval)
            return S_core_raw, S_eval_raw, clf, True
        except Exception as e:
            print(f"[warn] OOF failed for {label} {params}: {e}. Falling back to in-sample calibration.")
    S_core_raw, S_eval_raw, clf = _scores_in_sample(label, base, params, X_core, Y_core, X_eval, row_w)
    return S_core_raw, S_eval_raw, clf, False


# ==============================================================================
# Legacy kNN (vote) with guards  (verbatim)
# ==============================================================================
def _safe_cosine(S):
    return np.clip(S, -1.0, 1.0)


def legacy_knn_predict_eval(X_eval, X_core, Y_core, K, thresh, cutoff, freq=None, core_vocab=None):
    S = _safe_cosine(X_eval @ X_core.T)
    n = X_eval.shape[0]
    Yb = np.zeros((n, 6), dtype=int); Ys = np.zeros((n, 6), dtype=float)
    def _weights(top_idx):
        if freq is None: return np.ones(len(top_idx), float)
        if isinstance(freq, dict) and core_vocab is not None:
            return np.array([freq.get(core_vocab[j], 1.0) for j in top_idx], dtype=float)
        f = np.asarray(freq, dtype=float)
        return f[top_idx] if f.ndim == 1 else np.ones(len(top_idx), float)
    for i in range(n):
        s = S[i]; idx = np.where(s >= float(thresh))[0]
        if idx.size == 0: continue
        top = idx[np.argsort(s[idx])[::-1][:int(K)]]
        w = _weights(top).reshape(-1, 1)
        sc = (Y_core[top] * w).sum(axis=0)
        Ys[i] = sc; Yb[i] = (sc >= float(cutoff)).astype(int)
    return Yb, Ys


def legacy_knn_predict_core_loo(X_core, Y_core, K, thresh, cutoff, freq=None, core_vocab=None):
    S = _safe_cosine(X_core @ X_core.T); np.fill_diagonal(S, 0.0)
    n = X_core.shape[0]; Yb = np.zeros((n, 6), dtype=int)
    def _weights(top_idx):
        if freq is None: return np.ones(len(top_idx), float)
        if isinstance(freq, dict) and core_vocab is not None:
            return np.array([freq.get(core_vocab[j], 1.0) for j in top_idx], dtype=float)
        f = np.asarray(freq, dtype=float)
        return f[top_idx] if f.ndim == 1 else np.ones(len(top_idx), float)
    for i in range(n):
        s = S[i]; idx = np.where(s >= float(thresh))[0]
        if idx.size == 0: continue
        top = idx[np.argsort(s[idx])[::-1][:int(K)]]
        w = _weights(top).reshape(-1, 1)
        sc = (Y_core[top] * w).sum(axis=0)
        Yb[i] = (sc >= float(cutoff)).astype(int)
    return Yb


def _log_scan(model, params, mode, p, thr, scs_c, scs_e, obj_e, out_c, out_e, parts_c, parts_e,
              passed=True, used_oof=None):
    """D4: instrumentation only; record StructuralStats components per candidate.
    SCS re-aggregation under new weights: SCS' = wD*D - wM*M + wT*T + wC*C - wE*E
    per side; the notebook's PenalizedObj for OvR rows is
    (Eval_SCS - 0.05*Eval_E) - band_penalty, for Legacy kNN rows Eval_SCS - band_penalty."""
    row = dict(
        Model=model, Params=str(params), Mode=mode, Percentile=p,
        Thresholds=(json.dumps([round(float(t), 6) for t in thr]) if thr is not None else None),
        PassedMinAccept=bool(passed), UsedOOF=used_oof,
        SCS_core=scs_c, SCS_eval=scs_e, PenalizedObj_eval=obj_e,
    )
    for side, parts, out in (("Core", parts_c, out_c), ("Eval", parts_e, out_e)):
        if parts is None or out is None:
            for suffix in ["D", "M", "T", "C", "E", "D_bar", "W_bar", "rho",
                           "acceptance", "entry_rate", "en_over_ac", "per_level"]:
                row[f"{side}_{suffix}"] = None
            continue
        row[f"{side}_D"] = parts["D"]; row[f"{side}_M"] = parts["M"]
        row[f"{side}_T"] = parts["trend"]; row[f"{side}_C"] = parts["C"]
        row[f"{side}_E"] = parts["empty_levels"]
        row[f"{side}_D_bar"] = parts.get("D_bar"); row[f"{side}_W_bar"] = parts.get("W_bar")
        row[f"{side}_rho"] = parts.get("rho")
        row[f"{side}_acceptance"] = out["acc"]["acceptance"]
        row[f"{side}_entry_rate"] = out["acc"]["entry_rate"]
        row[f"{side}_en_over_ac"] = out["acc"]["en_over_ac"]
        row[f"{side}_per_level"] = json.dumps(out["acc"]["per_level"])
    SCAN_LOG.append(row)


def tune_legacy_knn(
    Ks=LEGACY_KNN_GRID["Ks"],
    Th=LEGACY_KNN_GRID["Th"],
    Ct=LEGACY_KNN_GRID["Ct"],
    use_freq=False,
    min_accept: float = MIN_ACCEPT
):
    freq = core_freq if use_freq else None
    vocab = core_words
    best, best_key = None, None
    for K, th, co in product(Ks, Th, Ct):
        Yb_eval, _ = legacy_knn_predict_eval(X_eval, X_core, Y_core, K, th, co, freq=freq, core_vocab=vocab)
        out_e = structural_eval(Yb_eval, eval_verbs)
        if out_e["acc"]["acceptance"] < min_accept:
            # D4: log the failing combo (eval side only; the notebook computes no
            # more for it, SCS here is derived for the log alone)
            _scs_e, _parts_e = SCS(out_e)
            _log_scan("Legacy kNN" + (" (freq)" if use_freq else ""), dict(K=K, thresh=th, cutoff=co),
                      f"K={K}/th={th}/c={co}", None, None, None, _scs_e, None,
                      None, out_e, None, _parts_e, passed=False)
            continue
        scs_e, parts_e = SCS(out_e)
        Yb_core = legacy_knn_predict_core_loo(X_core, Y_core, K, th, co, freq=freq, core_vocab=vocab)
        out_c = structural_eval(Yb_core, core_words); scs_c, parts_c = SCS(out_c)
        obj_e = _objective(scs_e, out_e, TARGETS)
        _log_scan("Legacy kNN" + (" (freq)" if use_freq else ""), dict(K=K, thresh=th, cutoff=co),
                  f"K={K}/th={th}/c={co}", None, None, scs_c, scs_e, obj_e, out_c, out_e, parts_c, parts_e)  # D4
        rec = dict(Model="Legacy kNN", Params=dict(K=K, thresh=th, cutoff=co, use_freq=bool(use_freq)),
                   Mode=f"K={K}/th={th}/c={co}", thr=None,
                   scs_c=scs_c, scs_e=scs_e, obj_e=obj_e,
                   out_c=out_c, out_e=out_e, parts_c=parts_c, parts_e=parts_e,
                   Yb_c=Yb_core, Yb_e=Yb_eval, clf=None, S_core=None, S_eval=None, cdfs=None, row_w=ROW_W)
        # BUGFIX (round 3, D11): rank by the penalized objective, as tune_ovr_model does; ties by
        # raw SCS, then acceptance; on a full tie the first configuration in grid order is kept.
        # The notebook and round 2 kept the largest RAW SCS within SCS_TIE_EPS = 0.02 (the larger
        # acceptance among near-ties), which ignored the guardrail penalty of the objective.
        key = (rec["obj_e"], rec["scs_e"], rec["out_e"]["acc"]["acceptance"])
        if best is None or key > best_key:
            best, best_key = rec, key
    return best


# ==============================================================================
# Model grids (family-specific)  (verbatim hyperparameters; D3a shim for CalibratedSVC)
# ==============================================================================
def _make_calibrated_svc(**kw):
    svc = SVC(kernel="linear", C=kw.get("C", 1.0), class_weight="balanced", probability=False)
    try:
        return CalibratedClassifierCV(base_estimator=svc, method="sigmoid", cv=3)
    except TypeError:  # sklearn >= 1.2 renamed base_estimator -> estimator
        return CalibratedClassifierCV(estimator=svc, method="sigmoid", cv=3)


MODEL_GRID = [
    ("LinearSVC", LinearSVC, [
        {"C": 0.1, "class_weight": "balanced", "max_iter": 5000, "tol": 1e-4, "random_state": RNG_SEED},
        {"C": 0.5, "class_weight": "balanced", "max_iter": 5000, "tol": 1e-4, "random_state": RNG_SEED},
        {"C": 1.0, "class_weight": "balanced", "max_iter": 5000, "tol": 1e-4, "random_state": RNG_SEED},
        {"C": 2.0, "class_weight": "balanced", "max_iter": 5000, "tol": 1e-4, "random_state": RNG_SEED},
    ]),
    ("CalibratedSVC", _make_calibrated_svc, [
        {"C": 0.5}, {"C": 1.0}, {"C": 2.0}
    ]),
    ("LogisticRegression", LogisticRegression, [
        {"C": 0.1, "solver": "liblinear", "penalty": "l2", "class_weight": "balanced", "max_iter": 4000, "random_state": RNG_SEED},
        {"C": 0.5, "solver": "liblinear", "penalty": "l2", "class_weight": "balanced", "max_iter": 4000, "random_state": RNG_SEED},
        {"C": 1.0, "solver": "liblinear", "penalty": "l2", "class_weight": "balanced", "max_iter": 4000, "random_state": RNG_SEED},
    ]),
    ("RidgeClassifier", RidgeClassifier, [
        {"alpha": 1.0}, {"alpha": 5.0}, {"alpha": 10.0},
    ]),
    ("SGDClassifier", SGDClassifier, [
        {"loss": "hinge",    "alpha": 1e-4, "class_weight": "balanced", "max_iter": 4000, "tol": 1e-4, "random_state": RNG_SEED},
        {"loss": "hinge",    "alpha": 1e-5, "class_weight": "balanced", "max_iter": 4000, "tol": 1e-4, "random_state": RNG_SEED},
        {"loss": "log_loss", "alpha": 1e-5, "class_weight": "balanced", "max_iter": 4000, "tol": 1e-4, "random_state": RNG_SEED},
    ]),
    ("PassiveAggressive", PassiveAggressiveClassifier, [
        {"C": 0.5, "class_weight": "balanced", "max_iter": 4000, "tol": 1e-4, "random_state": RNG_SEED},
        {"C": 1.0, "class_weight": "balanced", "max_iter": 4000, "tol": 1e-4, "random_state": RNG_SEED},
        {"C": 2.0, "class_weight": "balanced", "max_iter": 4000, "tol": 1e-4, "random_state": RNG_SEED},
    ]),
    ("LDA", LDA, [
        {"solver": "lsqr", "shrinkage": "auto"},
    ]),
    ("QDA", QDA, [
        {"reg_param": 0.05}, {"reg_param": 0.10},
    ]),
    ("GaussianNB", GaussianNB, [
        {"var_smoothing": 1e-9}, {"var_smoothing": 1e-8}
    ]),
    ("NearestCentroid", NearestCentroid, [{"metric": "euclidean"}]),
    ("KNeighborsClassifier", KNeighborsClassifier, [
        {"n_neighbors": 11, "metric": "cosine", "weights": "distance", "algorithm": "brute"},
        {"n_neighbors": 21, "metric": "cosine", "weights": "distance", "algorithm": "brute"},
    ]),
]


# ==============================================================================
# Tuning loop  (verbatim except the per-setting pick, D16: since round 2 it maximizes the
# penalized objective directly, see the BUGFIX comment below; D4 adds the _log_scan calls)
# ==============================================================================
def tune_ovr_model(label: str, base, param_grid: List[Dict[str, Any]], min_accept: float = 0.01):
    best = None
    for params in param_grid:
        try:
            S_core_raw, S_eval_raw, clf, used_oof = _get_scores(label, base, params)
        except Exception as e:
            print(f"[skip] {label} {params} -> score error: {e}")
            continue

        # Calibrate raw scores to [0,1] using (OOF if available) core scores
        cdfs = _fit_cdf_per_class(S_core_raw)
        S_core01 = _apply_cdf_per_class(S_core_raw, cdfs)
        S_eval01 = _apply_cdf_per_class(S_eval_raw, cdfs)

        cands = []
        for p in P_GRID:
            thr = thresholds_from_percentile_weighted(S_core01, Y_core, p, ROW_W if USE_ROW_WEIGHTS else np.ones_like(ROW_W))
            Yb_c, Yb_e, out_c, out_e, scs_c, scs_e, parts_c, parts_e = evaluate_with_thresholds(
                S_core01, S_eval01, thr, core_words, eval_verbs
            )
            if out_e["acc"]["acceptance"] < min_accept:
                _log_scan(label, params, f"P{p}", p, thr, scs_c, scs_e, None, out_c, out_e, parts_c, parts_e,
                          passed=False, used_oof=used_oof)  # D4
                continue

            obj_e = _objective(scs_e - SCS_EMPTY_LEVEL_PEN * parts_e["empty_levels"], out_e, TARGETS)
            _log_scan(label, params, f"P{p}", p, thr, scs_c, scs_e, obj_e, out_c, out_e, parts_c, parts_e,
                      passed=True, used_oof=used_oof)  # D4

            cands.append({
                "p": p, "thr": thr, "scs_c": scs_c, "scs_e": scs_e, "obj_e": obj_e,
                "out_c": out_c, "out_e": out_e, "parts_c": parts_c, "parts_e": parts_e,
                "Yb_c": Yb_c, "Yb_e": Yb_e, "clf": clf,
                "S_core": S_core01, "S_eval": S_eval01, "params": params,
                "cdfs": cdfs, "row_w": ROW_W if USE_ROW_WEIGHTS else np.ones_like(ROW_W),
                "Model": label, "Mode": f"P{p}",   # <-- ensure presence for table
            })

        if not cands:
            continue

        # BUGFIX (round 2): the original code shortlisted by RAW SCS within
        # SCS_TIE_EPS and only then maximized the penalized objective. Because the
        # shortlist ignored the guardrail penalty, a family could be represented by a
        # configuration with a far worse objective than its own optimum (e.g.
        # LogisticRegression was reported at objective 0.020 when its best setting
        # scores 0.857, and the SGD family was represented by P50 at 0.945 rather than
        # P55 at 0.983). The penalized objective is the criterion the paper states, so
        # it is now maximized directly, with raw SCS as the tie-break.
        cands.sort(key=lambda d: (d["obj_e"], d["scs_e"], d["out_e"]["acc"]["acceptance"]),
                   reverse=True)
        sel = cands[0]

        if (best is None) or (sel["obj_e"] > best["obj_e"]) or (
            abs(sel["obj_e"] - best["obj_e"]) < 1e-12 and sel["scs_e"] > best["scs_e"]
        ):
            best = sel
    return best


def _mean_norm(A):
    n = np.linalg.norm(A, axis=1) + 1e-12
    return float(np.mean(n))


def _row(rec):
    oc, oe = rec["out_c"], rec["out_e"]
    pc, pe = rec["parts_c"], rec["parts_e"]
    return dict(
        Model=f"{rec['Model']} ({rec['Mode']})",
        Params=str(rec["params"] if "params" in rec else rec.get("Params", "{}")),
        Core_SCS=rec["scs_c"], Core_Acceptance=oc["acc"]["acceptance"], Core_EntryRate=oc["acc"]["entry_rate"], Core_en_over_ac=oc["acc"]["en_over_ac"],
        Core_D=pc["D"], Core_M=pc["M"], Core_trend=pc["trend"], Core_EmptyLevels=pc["empty_levels"],
        Eval_SCS=rec["scs_e"], Eval_Acceptance=oe["acc"]["acceptance"], Eval_EntryRate=oe["acc"]["entry_rate"], Eval_en_over_ac=oe["acc"]["en_over_ac"],
        Eval_D=pe["D"], Eval_M=pe["M"], Eval_trend=pe["trend"], Eval_EmptyLevels=pe["empty_levels"],
        PenalizedObj=rec.get("obj_e", _objective(rec["scs_e"], rec["out_e"], TARGETS))
    )


# ==============================================================================
# Production post-processing helpers  (verbatim from BT3 cell 11)
# ==============================================================================
def _accepted_pattern(bits: np.ndarray) -> str:
    return "".join(str(int(b)) for b in bits.tolist())


def _choose_primary(scores01: np.ndarray, mask_bin: np.ndarray) -> list:
    """Primary = argmax of scores among 1s in mask; '' if none."""
    prim = []
    for i in range(scores01.shape[0]):
        if mask_bin[i].sum() == 0:
            prim.append("")
        else:
            idx = np.argmax(scores01[i] * mask_bin[i])
            prim.append(LEVELS[idx])
    return prim


def _other_domains(mask_bin: np.ndarray, primary: list) -> list:
    out = []
    prim_idx = {i: (LEVELS.index(p) if p in LEVELS else -1) for i, p in enumerate(primary)}
    for i in range(mask_bin.shape[0]):
        if mask_bin[i].sum() <= 1:
            out.append("")
            continue
        others = [LEVELS[j] for j in range(6) if mask_bin[i, j] == 1 and j != prim_idx[i]]
        out.append(", ".join(others))
    return out


def _ensure_eval_verb_col(df_eval):
    for c in ["verb", "Verb", "word", "token", "WORD", "VERB"]:
        if c in df_eval.columns: return c
    raise RuntimeError("No verb column found in df_eval.")


def _apply_top2_margin(S01, Yb, delta=TOP2_MARGIN_DEFAULT):
    """If >=2 labels accepted and (top1 - top2) >= delta, collapse to single label (delta 0.060, D14)."""
    Yb2 = Yb.copy()
    for i in range(S01.shape[0]):
        idx = np.where(Yb2[i] == 1)[0]
        if idx.size >= 2:
            s = S01[i, idx]
            o = np.argsort(s)[::-1]
            top, second = idx[o[0]], idx[o[1]]
            if (S01[i, top] - S01[i, second]) >= delta:
                Yb2[i, :] = 0
                Yb2[i, top] = 1
    return Yb2


# Scorer helper written into the bundle (verbatim from BT3 cell 10)
SCORER_CODE = '''
import json, numpy as np, joblib
from pathlib import Path

LEVELS = ["Kn","Cm","Ap","An","Sn","Ev"]

def _load_cdfs(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cdfs = []
    for d in raw:
        xs = np.asarray(d["xs"], dtype=float)
        cdf = np.asarray(d["cdf"], dtype=float)
        cdfs.append((xs, cdf))
    return cdfs

def apply_cdfs(S, cdfs):
    S = np.nan_to_num(np.asarray(S, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    S01 = np.zeros_like(S, dtype=float)
    for j in range(S.shape[1]):
        xs, cdf = cdfs[j]
        S01[:, j] = np.interp(S[:, j], xs, cdf, left=cdf[0], right=cdf[-1])
    return np.clip(S01, 0.0, 1.0)

class BT4Scorer:
    def __init__(self, bundle_dir, use_oof_thresholds=False):
        bundle = Path(bundle_dir)
        self.model = joblib.load(bundle / "model.pkl")
        self.cdfs  = _load_cdfs(bundle / "calibration_cdfs.json")
        with open(bundle / "thresholds.json", "r", encoding="utf-8") as f:
            thr = json.load(f)
        if use_oof_thresholds and "thresholds_oof_aligned" in thr:
            self.thresholds = np.asarray(thr["thresholds_oof_aligned"], dtype=float)
            self.thr_source = "OOF_ALIGNED"
        else:
            self.thresholds = np.asarray(thr["thresholds_tuning"], dtype=float)
            self.thr_source = "TUNING"

    def score_raw(self, X):
        # Uses decision_function if available; else predict_proba; else predict.
        m = self.model
        if hasattr(m, "decision_function"):
            S = m.decision_function(X)
        elif hasattr(m, "predict_proba"):
            P = m.predict_proba(X)
            # OneVsRest: list of (n,2) arrays -> take prob of class 1
            if isinstance(P, list):
                S = np.column_stack([p[:,1] if p.ndim == 2 and p.shape[1] >= 2 else p.ravel() for p in P])
            else:
                S = P
        else:
            S = m.predict(X).astype(float)
        S = np.asarray(S, dtype=float)
        if S.ndim == 1: S = S.reshape(-1, len(LEVELS))
        return S

    def score_calibrated(self, X):
        S = self.score_raw(X)
        return apply_cdfs(S, self.cdfs)

    def predict_binary(self, X):
        S01 = self.score_calibrated(X)
        Yb = (S01 >= self.thresholds.reshape(1, -1)).astype(int)
        return Yb, S01

    def suggest_binary(self, X, alpha=0.85):
        S01 = self.score_calibrated(X)
        thr_soft = self.thresholds * float(alpha)
        Yb = (S01 >= thr_soft.reshape(1, -1)).astype(int)
        return Yb, S01
'''.strip("\n")


# ==============================================================================
# Model selection (round 3, D12): select_model.py rule with matched supervised criteria
# ==============================================================================
def _finite_or_none(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_production_family(records: List[Dict[str, Any]], core_csv: str) -> Dict[str, Any]:
    """Applies select_model.SELECTION_RULE to the per-family structural bests of this run, with
    the supervised criteria of exactly those configurations and percentiles
    (results/selection/supervised_grid_per_seed.csv, written by supervised_eval.py). Never falls
    back to a structural-only selection: every missing or inconsistent input raises."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)          # python -I does not put the script directory on sys.path
    import select_model as sm             # imported here: select_model imports retrain lazily

    if sm.ADEQUACY_FLOOR != ADEQUACY_FLOOR:
        raise RuntimeError(f"ADEQUACY_FLOOR = {ADEQUACY_FLOOR} differs from select_model.ADEQUACY_FLOOR = "
                           f"{sm.ADEQUACY_FLOOR}")
    sm.check_constants()                  # the constants select_model mirrors equal this file's

    per_seed = os.path.join(sm.SEL_DIR, "supervised_grid_per_seed.csv")
    if not os.path.exists(per_seed):
        raise RuntimeError(
            f"Model selection needs the supervised evaluation results, which are missing: {per_seed}\n"
            "Run first:  python -I -B revision_pipeline/supervised_eval.py all\n"
            "retrain.py never falls back to a structural-only selection; use --scan-only to write the "
            "structural scan alone.")
    sm_words, sm_Y, _ = load_core_from_csv(sm.CORE_CSV)
    if list(sm_words) != list(core_words) or not np.array_equal(np.asarray(sm_Y), np.asarray(Y_core)):
        raise RuntimeError(f"The core of this run ({core_csv}) differs from the core of the supervised "
                           f"evaluation ({sm.CORE_CSV}); the supervised criteria would not match.")
    if MOCK_EMBED:
        print("[MOCK] the supervised criteria were computed with the real embeddings; this selection is a "
              "dry run only.")
    sup = sm.SupervisedTables(sm.SEL_DIR, require=True)

    rows = []
    for r in records:
        if r["Model"] == sm.LEGACY:
            p = r["Params"]
            cfg = sm.legacy_config_id("freq-weighted" if p["use_freq"] else "unweighted",
                                      p["K"], p["thresh"], p["cutoff"])
            pct = -1
        else:
            cfg = sm.config_id(r["Model"], r["params"])
            pct = int(r["p"])
        rows.append(dict(family=r["Model"], label=f"{r['Model']} ({r['Mode']})", cfg_id=cfg, percentile=pct,
                         SCS=float(r["scs_e"]), PenalizedObj=float(r["obj_e"]),
                         acceptance=float(r["out_e"]["acc"]["acceptance"]),
                         e_over_a=float(r["out_e"]["acc"]["en_over_ac"])))
    menu = sm.menu_from_records(rows, sup)

    # (a) the supervised criteria exist for every family the rule considers
    crit = ["CV_macro_AUC_mean5", "HO_within1_mean5"]
    pool = menu[menu["eligible"] & (menu["PenalizedObj"] >= ADEQUACY_FLOOR)]
    missing = pool[pool[crit].isna().any(axis=1)]
    if len(missing):
        raise RuntimeError(
            "Supervised results are missing for " +
            ", ".join(f"{c} P{p}" for c, p in zip(missing["cfg_id"], missing["percentile"])) +
            f" in {per_seed}. Rerun supervised_eval.py after the scan.")
    if len(pool) < 2:
        raise RuntimeError(f"Only {len(pool)} eligible families reach the adequacy floor {ADEQUACY_FLOOR}; "
                           "the selection rule needs at least two.")

    # (b) select_model's own re-derivation from the scan file written by this run must agree
    scan_csv = str(RESULTS_DIR / "structural_scan_components.csv")
    ref = sm.build_menu(sm.StructuralScan(scan_csv)).set_index("family")
    bad = [] if len(ref) == len(menu) else [f"{len(ref)} families there, {len(menu)} here"]
    for r in menu.itertuples():
        if r.family not in ref.index:
            bad.append(f"{r.family}: missing"); continue
        q = ref.loc[r.family]
        mine = (r.cfg_id, int(r.percentile), float(r.PenalizedObj), float(r.SCS))
        theirs = (q["cfg_id"], int(q["percentile"]), float(q["PenalizedObj"]), float(q["SCS"]))
        if mine != theirs:
            bad.append(f"{r.family}: {mine} vs {theirs}")
    if bad:
        raise RuntimeError("select_model's re-derivation from " + scan_csv + " disagrees with this run: " +
                           "; ".join(bad))

    # (c) the rule (select_model.SELECTION_RULE); the round-2 two-stage form is reported, not used
    ranked, info = sm.select(menu, floor=ADEQUACY_FLOOR)
    if info["winner"] is None:
        raise RuntimeError("the selection rule returned no family")
    win = ranked.iloc[0]
    if not bool(win["eligible"]):
        raise RuntimeError(f"selected family {win['family']} is not eligible")
    _, info_v2 = sm.rule_v2_two_stage(menu, floor=ADEQUACY_FLOOR, ho_col="HO_within1_seed13")

    # (d) agreement with select_model.py's committed output
    st_path = os.path.join(sm.SEL_DIR, "selection_table.csv")
    if MOCK_EMBED:
        st_status = "skipped (mock embeddings)"
    elif not os.path.exists(st_path):
        st_status = "selection_table.csv not found (select_model.py not run)"
        print(f"[warn] {st_status}")
    else:
        st = pd.read_csv(st_path, float_precision="round_trip")
        chosen = st.loc[st["selected"].astype(bool)]
        theirs = (list(chosen["family"]), [float(x) for x in chosen["score"]], list(pd.unique(st["selection_rule"])))
        mine = ([info["winner"]], [float(info["score"])], [info["rule"]])
        if theirs != mine:
            raise RuntimeError(f"{st_path} (select_model.py) gives {theirs}, this run gives {mine}. Rerun "
                               "select_model.py or check the inputs.")
        st_status = "same family, rule and score"

    table = menu[["family", "label", "eligible", "cfg_id", "percentile", "PenalizedObj", "CV_macro_AUC_mean5",
                  "HO_within1_mean5", "HO_within1_seed13"]].merge(
        ranked[["family", "n_obj", "n_auc", "n_ho", "score", "limiting_criterion", "rank"]], on="family", how="left")
    table.insert(3, "floor_pass", table["PenalizedObj"] >= ADEQUACY_FLOOR)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n[SELECTION] select_model." + sm.RULES[info["rule"]].__name__ +
              f" (SELECTION_RULE = {info['rule']!r}, adequacy floor {ADEQUACY_FLOOR}):")
        print(table.drop(columns=["cfg_id", "HO_within1_seed13"]).to_string(index=False))
    print(f"[SELECTION] selected {info['winner']} ({win['label']}), score {info['score']:.10f}; runner-up "
          f"{info['runner_up']} {info['runner_up_score']:.10f}; margin {info['margin']:.10f}")
    print(f"[SELECTION] round-2 two-stage form on the same inputs (reported only): {info_v2['winner']} "
          f"{info_v2['score']:.10f}")
    print(f"[SELECTION] cross-checks: select_model re-derivation identical; selection_table.csv: {st_status}")

    rel = lambda path: os.path.relpath(path, sm.REPO).replace(os.sep, "/")
    return dict(
        rule=info["rule"],
        rule_function=sm.RULES[info["rule"]].__name__,
        adequacy_floor=ADEQUACY_FLOOR,
        criteria=info.get("criteria"),
        eligibility="one-vs-rest families with per-level OOF-ECDF calibrated scores; Legacy kNN is a "
                    "structural reference row only (select_model.ELIGIBILITY_NOTE)",
        winner_family=str(info["winner"]),
        winner_label=str(win["label"]),
        winner_cfg_id=str(win["cfg_id"]),
        winner_percentile=int(win["percentile"]),
        score=_finite_or_none(info["score"]),
        runner_up=info["runner_up"],
        runner_up_score=_finite_or_none(info["runner_up_score"]),
        margin=_finite_or_none(info["margin"]),
        v2_two_stage_same_inputs=dict(winner=info_v2["winner"], score=_finite_or_none(info_v2["score"]),
                                      note="round-2 two-stage rule (held-out seed 13) on the same matched "
                                           "inputs; reported only, not used"),
        supervised_inputs=dict(path=rel(per_seed), sha256=_sha256(per_seed)),
        cross_checks=dict(select_model_rederivation="identical configurations, PenalizedObj and SCS",
                          selection_table=st_status,
                          supervised_coverage="complete for every eligible family above the floor"),
        table=json.loads(table.to_json(orient="records")),
    )


def _reproduction_checks(bundle_dir: Path, bank_path: Path, bank_before: Optional[bytes]) -> Dict[str, Any]:
    """D15: compares the exported winner bundle with the packaged production_model/ and the exported
    bank with the repository's results/"Verb List Classified by Model.csv" (read before the export).
    Reports only; nothing in production_model/ or in the repository bank is written here."""
    repo = Path(__file__).resolve().parent.parent
    prod = repo / "production_model"
    out: Dict[str, Any] = {}
    if prod.is_dir():
        files = {}
        for name in ("model.pkl", "calibration_cdfs.json", "thresholds.json", "row_weights.npy", "bt4_scorer.py"):
            a, b = bundle_dir / name, prod / name
            if not (a.exists() and b.exists()):
                files[name] = "missing"
                continue
            if a.read_bytes() == b.read_bytes():
                files[name] = "byte-identical"
                continue
            try:
                if name == "model.pkl":
                    ma, mb = joblib.load(a), joblib.load(b)
                    d = max(float(np.max(np.abs(scores_ovr(ma, X) - scores_ovr(mb, X)))) for X in (X_core, X_eval))
                    what = "raw scores on the core and the candidates"
                elif name == "row_weights.npy":
                    d = float(np.max(np.abs(np.load(a) - np.load(b))))
                    what = "values"
                elif name == "bt4_scorer.py":
                    d, what = float("nan"), "source text"
                else:
                    ja, jb = (json.loads(p.read_text(encoding="utf-8")) for p in (a, b))
                    flat = ((lambda j: np.concatenate([np.r_[c["xs"], c["cdf"]] for c in j]))
                            if name == "calibration_cdfs.json" else (lambda j: np.asarray(j["thresholds_tuning"])))
                    d = float(np.max(np.abs(flat(ja) - flat(jb))))
                    what = "values"
                files[name] = f"differs in bytes; max abs difference of {what}: {d:.3e}"
            except Exception as e:   # shapes differ, unreadable file
                files[name] = f"differs ({type(e).__name__}: {e})"
        out["production_model"] = files
    if bank_before is not None:
        new = bank_path.read_bytes()
        if new == bank_before:
            out["bank"] = "byte-identical to the repository bank"
        else:
            from io import BytesIO
            a = pd.read_csv(BytesIO(bank_before), keep_default_na=False)
            b = pd.read_csv(BytesIO(new), keep_default_na=False)
            cols = ["Verb", "AcceptedPattern", "PrimaryDomain", "OtherDomains", "SoftPattern", "SoftPrimaryDomain"]
            same = (len(a) == len(b)) and bool((a[cols].astype(str).to_numpy() == b[cols].astype(str).to_numpy()).all())
            out["bank"] = ("differs in bytes; hard labels, primaries and soft patterns identical for all "
                           f"{len(b)} verbs" if same else "DIFFERS in labels, primaries or soft patterns")
    for k, v in out.items():
        print(f"[CHECK] {k}: {v}")
    flat_status = json.dumps(out)
    if "differs" in flat_status.lower() or "missing" in flat_status:
        print("[CHECK] WARNING: the export is not byte-identical to the packaged production model or the "
              "repository bank (details above).")
    return out


# ==============================================================================
# Main pipeline
# ==============================================================================
def run_pipeline(core_csv: str, candidates_path: str, outdir: str, scan_only: bool = False) -> None:
    global X_core, Y_core, X_eval, core_words, eval_verbs, core_freq, df_eval
    global ROW_W, TARGETS, RESULTS_DIR, EMBED_CACHE_DIR

    if CV_RESULTS_PATH:
        print("[warn] retrain.CV_RESULTS_PATH is a round-2 option and is ignored: the selection uses "
              "select_model.py with the supervised_eval.py results (D12).")
    SCAN_LOG.clear()
    ROW_WEIGHT_FIT.update(weighted_fit_ok=0, weighted_fit_fallback_unweighted=0)

    outdir = Path(outdir)
    RESULTS_DIR = outdir / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_CACHE_DIR = str(RESULTS_DIR / "_embed_cache")   # D2 (notebook: results/_embed_cache)
    _load_npz_cache()
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)

    # ---- 1) Load data ----
    core_words, Y_core, core_freq = load_core_from_csv(core_csv)
    eval_verbs, df_eval = load_candidates(candidates_path)

    X_core = embed_batch(core_words)
    X_eval = embed_batch(eval_verbs)
    print(f"[DATA] core={len(core_words)} | eval={len(eval_verbs)} | embed_dim={X_core.shape[1]}")

    # ---- Targets (verbatim, incl. clamp) ----
    TARGETS = core_targets_from_labels(Y_core)
    print("Data-driven TARGETS:", TARGETS)
    lo, hi = TARGETS["ACC_BAND"]
    lo = max(ACC_BAND_CLAMP[0], min(lo, ACC_BAND_CLAMP[1]))
    hi = max(lo + 1e-3, min(ACC_BAND_CLAMP[1], hi))
    TARGETS["ACC_BAND"] = (lo, hi)
    print("Adjusted TARGETS:", TARGETS)

    # ---- Row weights ----
    ROW_W = _build_core_row_weights() if USE_ROW_WEIGHTS else np.ones(X_core.shape[0], dtype=float)
    print(f"[weights] use={USE_ROW_WEIGHTS} | min={ROW_W.min():.3f} mean={ROW_W.mean():.3f} max={ROW_W.max():.3f}")
    print("effective rank X_core:", _effective_rank(X_core))

    # sanity: norms for cosine pipelines (verbatim)
    if not (0.95 <= _mean_norm(X_core) <= 1.05 and 0.95 <= _mean_norm(X_eval) <= 1.05):
        print("[warn] Embeddings seem not L2-normalized; cosine-based pieces may degrade.")

    # ---- Tuning loop (verbatim) ----
    records: List[Dict[str, Any]] = []

    for label, base, grid in MODEL_GRID:
        print(f"=== Tuning {label} ===")
        rec = tune_ovr_model(label, base, grid, min_accept=MIN_ACCEPT)
        if rec is None:
            print(f"[warn] No valid configuration for {label}")
            continue
        records.append(rec)

    print("=== Tuning Legacy kNN (unweighted) ===")
    rec_knn = tune_legacy_knn(use_freq=False, min_accept=MIN_ACCEPT)
    if rec_knn is not None:
        records.append(rec_knn)

    if core_freq is not None:
        print("=== Tuning Legacy kNN (freq-weighted) ===")
        rec_knn_w = tune_legacy_knn(use_freq=True, min_accept=MIN_ACCEPT)
        if rec_knn_w is not None:
            # prefer weighted if better by obj/SCS/acceptance
            def _key(r): return (r["obj_e"], r["scs_e"], r["out_e"]["acc"]["acceptance"])
            if rec_knn is None or _key(rec_knn_w) > _key(rec_knn):
                if rec_knn is None: records.append(rec_knn_w)
                else: records[-1] = rec_knn_w

    # Validate/clean records to avoid KeyError: 'Model'
    records = [r for r in records if isinstance(r, dict) and "Model" in r and "Mode" in r]

    if not records:
        raise RuntimeError("No valid model configuration produced a usable candidate. Consider widening TARGET bands or P_GRID.")

    df_models = pd.DataFrame([_row(r) for r in records])
    print(df_models.sort_values(["PenalizedObj", "Eval_SCS", "Eval_Acceptance"], ascending=[False, False, False]).to_string(index=False))

    df_pareto = _pareto_front(df_models, cols=[("Eval_SCS", False), ("Eval_Acceptance", False)])
    print("\nPareto-front (Eval_SCS vs Acceptance):")
    print(df_pareto.sort_values(["Eval_SCS", "Eval_Acceptance"], ascending=[False, False]).to_string(index=False))

    # ---- Save scan components (D4), the per-family bests and the Pareto table, before the selection (D13) ----
    pd.DataFrame(SCAN_LOG).to_csv(RESULTS_DIR / "structural_scan_components.csv", index=False)
    df_models.sort_values(["PenalizedObj", "Eval_SCS", "Eval_Acceptance"], ascending=[False, False, False]) \
        .to_csv(RESULTS_DIR / "model_comparison_table5.csv", index=False)
    df_pareto.sort_values(["Eval_SCS", "Eval_Acceptance"], ascending=[False, False]) \
        .to_csv(RESULTS_DIR / "model_comparison_pareto.csv", index=False)
    print("Saved:", str(RESULTS_DIR / "structural_scan_components.csv"))
    print("Saved:", str(RESULTS_DIR / "model_comparison_table5.csv"))
    print("Saved:", str(RESULTS_DIR / "model_comparison_pareto.csv"))
    row_weight_fit = _row_weight_fit_report()
    if scan_only:
        print("\n[scan-only] structural scan written; no model selection and no export (D13).")
        return

    # ---- Model selection (D12): select_model.SELECTION_RULE with matched supervised criteria ----
    selection = select_production_family(records, core_csv)
    winner_name = selection["winner_label"]
    winner_row = df_models.loc[df_models["Model"] == winner_name].iloc[0]
    print(f"\n>> Selected family (select_model.{selection['rule_function']}):")
    print(winner_row.to_frame().T.to_string())

    winner_name = winner_row["Model"]
    _w = next(r for r in records if f"{r['Model']} ({r['Mode']})" == winner_name)

    winner_model       = _w.get("clf", None)          # None for Legacy kNN (vote)
    winner_params      = _w.get("params", _w.get("Params", {}))
    winner_thresholds  = _w.get("thr", None)          # thresholds in [0,1] for OvR models; None for kNN
    Ybin_winner_core   = _w["Yb_c"]
    Ybin_winner_eval   = _w["Yb_e"]
    S_core_winner      = _w.get("S_core", None)       # calibrated scores (OvR) or None (kNN)
    S_eval_winner      = _w.get("S_eval", None)       # calibrated scores (OvR) or None (kNN)
    winner_cdfs        = _w.get("cdfs", None)         # apply to raw scores before thresholds (OvR)
    winner_row_weights = _w.get("row_w", ROW_W)

    print(f"\nArtifacts ready for production:\n  {winner_name} | params={winner_params} | thr={winner_thresholds}")

    # D15: the repository bank as it is before this export (compared with the export below)
    repo_bank = Path(__file__).resolve().parent.parent / "results" / "Verb List Classified by Model.csv"
    bank_before = repo_bank.read_bytes() if (repo_bank.exists() and not MOCK_EMBED) else None

    # ---- SAVE WINNER BUNDLE (verbatim from cell 10; guard D8) ----
    if winner_model is None:
        raise RuntimeError("winner_model is None (Legacy kNN cannot be serialized as a scoring model in this path).")

    thr_oof = None  # D6: the committed notebook has no cell producing winner_thresholds_oof

    base_dir = RESULTS_DIR / "BT4_structural_best" / "artifacts"
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    bundle_name = f"{winner_name.replace(' ', '_').replace('(', '').replace(')', '')}_{ts}"
    bundle_dir = base_dir / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    model_path = bundle_dir / "model.pkl"
    joblib.dump(winner_model, model_path)

    cdfs_serializable = []
    for (xs, cdf) in winner_cdfs if winner_cdfs is not None else []:
        xs = np.asarray(xs, dtype=float)
        cdf = np.asarray(cdf, dtype=float)
        cdfs_serializable.append({"xs": xs.tolist(), "cdf": cdf.tolist()})
    cdfs_path = bundle_dir / "calibration_cdfs.json"
    with open(cdfs_path, "w", encoding="utf-8") as f:
        json.dump(cdfs_serializable, f, ensure_ascii=False)

    thr_obj = {
        "levels": LEVELS,
        "thresholds_tuning": np.asarray(winner_thresholds, dtype=float).tolist(),
    }
    if thr_oof is not None:
        thr_obj["thresholds_oof_aligned"] = np.asarray(thr_oof, dtype=float).tolist()
    thr_path = bundle_dir / "thresholds.json"
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(thr_obj, f, ensure_ascii=False, indent=2)

    if winner_row_weights is not None:
        np.save(bundle_dir / "row_weights.npy", np.asarray(winner_row_weights, dtype=float))
    if S_core_winner is not None:
        np.save(bundle_dir / "S_core_calibrated.npy", np.asarray(S_core_winner, dtype=float))
    if S_eval_winner is not None:
        np.save(bundle_dir / "S_eval_calibrated.npy", np.asarray(S_eval_winner, dtype=float))

    meta = {
        "created_at": ts,
        "winner_name": winner_name,
        "winner_params": winner_params,
        "levels": LEVELS,
        "level_names": LEVEL_NAMES,
        "use_row_weights": winner_row_weights is not None,
        "has_oof_thresholds": thr_oof is not None,
        "calibration": {
            "type": "rank_cdf_per_class",
            "note": "Apply CDFs to raw decision scores to map to [0,1] before thresholding.",
            "cdf_count": len(cdfs_serializable) if winner_cdfs is not None else 0
        },
        "thresholds": {
            "tuning_source": str(thr_path),
            "oof_present": thr_oof is not None
        },
        "artifact_paths": {
            "model": str(model_path),
            "cdfs": str(cdfs_path),
            "thresholds": str(thr_path),
            "row_weights": str(bundle_dir / "row_weights.npy") if winner_row_weights is not None else None,
            "S_core_calibrated": str(bundle_dir / "S_core_calibrated.npy") if S_core_winner is not None else None,
            "S_eval_calibrated": str(bundle_dir / "S_eval_calibrated.npy") if S_eval_winner is not None else None,
        }
    }
    with open(bundle_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(bundle_dir / "bt4_scorer.py", "w", encoding="utf-8") as f:
        f.write(SCORER_CODE)

    try:
        latest_link = base_dir / "LATEST"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(bundle_dir.name)  # relative link
        print("Updated symlink:", str(latest_link), "->", bundle_dir.name)
    except Exception:
        pass
    print("Winner bundle saved:", str(bundle_dir))

    # ---- PRODUCTION (verbatim from cell 11; D6 fallback branches noted) ----
    vcol_eval = _ensure_eval_verb_col(df_eval)
    verbs = [str(v).strip().lower() for v in df_eval[vcol_eval].tolist()]

    if S_eval_winner is None:
        raise RuntimeError("Missing S_eval_winner (and S_eval_winner_tuning). Re-run tuning/parity or OOF cell.")
    S01 = np.asarray(S_eval_winner, dtype=float)   # notebook's "CURRENT" branch (D6)
    score_src = "CURRENT"

    thr_vec = np.asarray(winner_thresholds, dtype=float)
    chosen = f"{winner_name} | calibrated thresholds (tuning)"
    print("[prod] Using parity/tuning thresholds")

    Ybin_hard = (S01 >= thr_vec.reshape(1, -1)).astype(int)

    if APPLY_TOP2_MARGIN:
        delta = TOP2_MARGIN_DEFAULT
        Ybin_hard = _apply_top2_margin(S01, Ybin_hard, delta=delta)
        chosen += f" + top2-margin({delta:.3f})"

    soft_thr = thr_vec * float(SUGGEST_ALPHA)
    Ybin_soft = (S01 >= soft_thr.reshape(1, -1)).astype(int)

    S01r = np.round(S01, 3)
    export = pd.DataFrame({"Verb": verbs})

    for j, lv in enumerate(LEVELS):
        export[f"score_{lv}"]      = S01r[:, j]
        export[f"accept_{lv}"]     = Ybin_hard[:, j].astype(int)
        export[f"cal_score_{lv}"]  = np.round(S01[:, j], 6)
        export[f"thr_{lv}"]        = np.round(thr_vec[j], 6)
        export[f"meets_thr_{lv}"]  = (S01[:, j] >= thr_vec[j]).astype(int)

    export["AcceptedCount"]   = Ybin_hard.sum(axis=1).astype(int)
    export["Accepted"]        = export["AcceptedCount"] > 0
    export["AcceptedPattern"] = [_accepted_pattern(Ybin_hard[i]) for i in range(Ybin_hard.shape[0])]
    hard_primary              = _choose_primary(S01, Ybin_hard)
    export["PrimaryDomain"]   = hard_primary
    export["OtherDomains"]    = _other_domains(Ybin_hard, hard_primary)

    export["SoftAcceptedCount"] = Ybin_soft.sum(axis=1).astype(int)
    export["SoftAccepted"]      = export["SoftAcceptedCount"] > 0
    export["SoftPattern"]       = [_accepted_pattern(Ybin_soft[i]) for i in range(Ybin_soft.shape[0])]
    soft_primary                = _choose_primary(S01, Ybin_soft)
    export["SoftPrimaryDomain"] = soft_primary
    export["SoftOtherDomains"]  = _other_domains(Ybin_soft, soft_primary)

    # keep original labels (if present) for traceability
    trace_cols = [c for c in df_eval.columns if c in LEVELS]
    if trace_cols:
        export = export.merge(
            df_eval[[vcol_eval] + trace_cols].rename(columns={vcol_eval: "Verb"}),
            on="Verb", how="left"
        )

    suggested_only = (export["AcceptedCount"] == 0) & export["SoftAccepted"]
    suggested_only_count = int(suggested_only.sum())

    n = len(export)
    accepted_words   = int(export["Accepted"].sum())
    rejected_words   = int((~export["Accepted"]).sum())
    fully_accepted   = int((export["AcceptedCount"] == 6).sum())
    single_label     = int((export["AcceptedCount"] == 1).sum())
    multi_label      = int((export["AcceptedCount"] >= 2).sum())
    acc_rate         = float(export["Accepted"].mean()) if n else 0.0
    entry_rate       = float(export["AcceptedCount"].mean()) if n else 0.0
    hist             = export["AcceptedCount"].value_counts().reindex(range(0, 7), fill_value=0).astype(int).to_dict()
    per_domain_totals = {lv: int(export[f"accept_{lv}"].sum()) for lv in LEVELS}

    print(f"Chosen setting: {chosen}")
    print(f"[debug] score source: {score_src} | thr source: TUNING | thr min/max: {thr_vec.min():.4f}/{thr_vec.max():.4f}")
    print(f"Acceptance rate: {acc_rate:.6f}   (accepted {accepted_words} / {n}, rejected {rejected_words})")
    print(f"Entry rate:      {entry_rate:.6f}")
    print(f"Fully accepted (all 6): {fully_accepted}")
    print(f"Single-label items:     {single_label}")
    print(f"Multi-label items:      {multi_label}")
    print(f"Suggested-only (no hard accept but soft suggestions present @ {SUGGEST_ALPHA:.2f}x thr): {suggested_only_count}")
    for k in range(0, 7):
        print(f"# of words with {k} entries: {hist.get(k, 0)}")
    for lv in LEVELS:
        print(f"# of entries acquired by domain {LEVEL_NAMES[lv]}: {per_domain_totals[lv]}")

    agg_cols    = ["Verb", "Accepted", "AcceptedCount", "AcceptedPattern", "PrimaryDomain", "OtherDomains",
                   "SoftAccepted", "SoftAcceptedCount", "SoftPattern", "SoftPrimaryDomain", "SoftOtherDomains"]
    score_cols  = [f"score_{lv}" for lv in LEVELS]
    debug_cols  = sum([[f"cal_score_{lv}", f"thr_{lv}", f"meets_thr_{lv}"] for lv in LEVELS], [])
    accept_cols = [f"accept_{lv}" for lv in LEVELS]
    orig_label_cols = [c for c in df_eval.columns if c in LEVELS]
    label_cols_in_export = [c for c in orig_label_cols if c in export.columns]

    cols_final = [c for c in (agg_cols + score_cols + debug_cols + accept_cols + label_cols_in_export) if c in export.columns]
    export = export[cols_final]

    out_dir = RESULTS_DIR / "BT4_structural_best"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_winner = winner_name.replace(" ", "_").replace("(", "").replace(")", "")
    out_path = out_dir / f"BT4_PROD_TUNING_{safe_winner}_{ts}.csv"
    export.to_csv(out_path, index=False)
    # Repo-style copy with the canonical name (same column layout as
    # results/"Verb List Classified by Model.csv")
    export.to_csv(RESULTS_DIR / "Verb List Classified by Model.csv", index=False)
    print("\nSaved:", str(out_path))
    print("Saved:", str(RESULTS_DIR / "Verb List Classified by Model.csv"))
    checks = ({"skipped": "mock embeddings"} if MOCK_EMBED else
              _reproduction_checks(bundle_dir, RESULTS_DIR / "Verb List Classified by Model.csv", bank_before))

    # ---- Winner, constants and per-family bests (winner_and_constants.json) ----
    summary = {
        "winner": {
            "name": winner_name,
            "params": winner_params,
            "mode": _w.get("Mode"),
            "percentile": _w.get("p"),
            "thresholds_tuning": np.asarray(winner_thresholds, dtype=float).tolist(),
            "Eval_SCS": float(winner_row["Eval_SCS"]),
            "PenalizedObj": float(winner_row["PenalizedObj"]),
            "Eval_Acceptance": float(winner_row["Eval_Acceptance"]),
            "bundle_dir": os.path.relpath(bundle_dir, outdir).replace(os.sep, "/"),
            "selected_by": f"select_model.{selection['rule_function']} (SELECTION_RULE = {selection['rule']!r})",
        },
        "selection": selection,
        "reproduction_checks": checks,
        "row_weight_fit": row_weight_fit,
        "constants": {
            "embedding_model": ("MOCK (deterministic random unit vectors)" if MOCK_EMBED
                                 else "sentence-transformers/all-mpnet-base-v2"),
            "l2_normalized_embeddings": True,
            "random_state": RNG_SEED,
            "use_row_weights": USE_ROW_WEIGHTS,
            "row_weight_rule": "w = 1 + log(1 + freq); clip [0.5, 3.0]; divide by mean",
            "use_oof": USE_OOF,
            "oof_folds_K": N_SPLITS_OOF,
            "oof_splitter": f"KFold(n_splits={N_SPLITS_OOF}, shuffle=True, random_state={RNG_SEED})",
            "calibration": "per-level empirical CDF on OOF core scores; cdf = rank/(n+1); interp with clamped tails",
            "percentile_grid": P_GRID,
            "scs_weights": {"wD": 1.0, "wM": 1.0, "wT": 0.8, "wC": 0.3, "wE": 0.05},
            "scs_empty_min": 10,
            "scs_target_ratio": 1.51,
            "scs_acc_window": [0.30, 0.85],
            "scs_empty_level_pen_in_objective": SCS_EMPTY_LEVEL_PEN,
            "targets": {
                "ACC_BAND": list(TARGETS["ACC_BAND"]),
                "ENAC_BAND": list(TARGETS["ENAC_BAND"]),
                "SCS_TIE_EPS": TARGETS["SCS_TIE_EPS"],
                "PEN_ACC": TARGETS["PEN_ACC"],
                "PEN_ENAC": TARGETS["PEN_ENAC"],
                "bootstrap": 400, "margin_acc": 0.08, "margin_enac": 0.08,
                "acc_band_clamp": list(ACC_BAND_CLAMP),
            },
            "min_accept": MIN_ACCEPT,
            "top2_margin_delta": TOP2_MARGIN_DEFAULT,
            "suggest_alpha": SUGGEST_ALPHA,
            "adequacy_floor": ADEQUACY_FLOOR,
            "selection_rule": selection["rule"],
            "row_weights_in_fit": ("attempted as in the notebook; rejected by scikit-learn 1.9.0, so every "
                                   "one-vs-rest model is fit unweighted; row weights enter only the "
                                   "threshold percentiles"),
            "legacy_knn_grid": {k: list(v) for k, v in LEGACY_KNN_GRID.items()},
            "legacy_knn_pick": ("per variant: largest PenalizedObj, ties raw SCS then acceptance (D11); the "
                                "freq-weighted variant replaces the unweighted one only if strictly better"),
            "levels": LEVELS,
        },
        "model_zoo": [
            {"model": label, "param_grid": grid} for label, _, grid in MODEL_GRID
        ] + [{"model": "Legacy kNN", "param_grid": "vote grid (Ks x Th x Ct above), unweighted and freq-weighted"}],
        "table5_model_comparison": json.loads(
            df_models.sort_values(["PenalizedObj", "Eval_SCS", "Eval_Acceptance"],
                                  ascending=[False, False, False]).to_json(orient="records")
        ),
        "production_summary": {
            "acceptance_rate": acc_rate, "entry_rate": entry_rate,
            "accepted_words": accepted_words, "rejected_words": rejected_words,
            "fully_accepted": fully_accepted, "single_label": single_label,
            "multi_label": multi_label, "suggested_only": suggested_only_count,
            "hist_accepted_count": {str(k): int(hist.get(k, 0)) for k in range(0, 7)},
            "per_domain_totals": per_domain_totals,
        },
        "inputs": {"core_csv": str(core_csv), "candidates": str(candidates_path),
                    "n_core": len(core_words), "n_candidates": len(eval_verbs)},
        "mock_embeddings": MOCK_EMBED,
        "versions": _versions(),
    }
    with open(RESULTS_DIR / "winner_and_constants.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Saved:", str(RESULTS_DIR / "winner_and_constants.json"))


def _versions() -> Dict[str, str]:
    import scipy
    import sklearn
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__}


def main():
    global MOCK_EMBED, MOCK_DIM
    ap = argparse.ArgumentParser(description="Standalone Algorithms 1 and 2: structural scan, model selection "
                                             "(select_model.py) and export of the selected family.")
    ap.add_argument("--core-csv", required=True, help="Core CSV: verb, Kn..Ev (0/1), freq")
    ap.add_argument("--candidates", required=True, help="Candidate verb list (.txt, one verb per line; or .csv)")
    ap.add_argument("--outdir", default="retrain_out", help="Output directory (default: retrain_out)")
    ap.add_argument("--scan-only", action="store_true",
                    help="stage 1: write the structural scan, the per-family bests and the Pareto table, then stop "
                         "(no model selection, no export)")
    ap.add_argument("--mock-embeddings", action="store_true",
                    help="D5: deterministic random unit vectors instead of MPNet (dry-run/testing ONLY)")
    ap.add_argument("--mock-dim", type=int, default=768, help="Dimension for --mock-embeddings (default 768)")
    args = ap.parse_args()

    MOCK_EMBED = bool(args.mock_embeddings)
    MOCK_DIM = int(args.mock_dim)
    if MOCK_EMBED:
        print(f"[MOCK] Using deterministic mock embeddings (dim={MOCK_DIM}). Results are NOT meaningful.")

    t0 = time.time()
    run_pipeline(args.core_csv, args.candidates, args.outdir, scan_only=args.scan_only)
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
