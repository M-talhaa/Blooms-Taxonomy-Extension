#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
select_model.py - model selection for the round-3 revision (structure + generalization).

WHAT THIS FILE DOES
  1. Re-aggregates the structural scan written by retrain.py
     (results/structural_scan_components.csv) into SCS and the penalized objective
     (Eq. 19) with exact scalar replicas of retrain.SCS, retrain._penalty_from_out and
     retrain._objective, and picks each family's structural best configuration with the
     bug-fixed rule (maximize the penalized objective; ties: raw SCS, then acceptance).
  2. Attaches MATCHED supervised numbers to every family: the CV and held-out results of
     exactly that configuration and percentile, from supervised_eval.py
     (results/selection/supervised_grid*.csv).
  3. Applies the selection rule. The rule is ONE named function, chosen by SELECTION_RULE;
     every alternative used in the robustness analysis is a named function too, so a
     different rule can be switched on without touching the inputs.

SELECTION RULE IN USE (SELECTION_RULE = "balanced_maximin", rule_balanced_maximin)
  Pending the authors' final confirmation (round-3 decision: keep kNN-21, replace the
  two-stage rule by a simpler balanced rule over the same three criteria).
  (1) Eligibility: families that produce per-level scores calibrated by the out-of-fold
      empirical CDF, and can therefore run the production pipeline (per-level
      thresholds, top-2 margin collapse, soft tier): every one-vs-rest family of the
      model zoo. The vote-based "Legacy kNN" is NOT eligible (one global vote cutoff, no
      per-level calibrated scores, retrain.py cannot export it); it is reported as a
      structural reference row only.
  (2) Adequacy floor: PenalizedObj >= ADEQUACY_FLOOR (0.5).
  (3) Three criteria, each at the family's structural best configuration and percentile:
      PenalizedObj (Eq. 19, 1,650 unlabeled candidates), CV macro AUC (repeated
      stratified 5x5 CV, mean of seeds 13 to 17) and held-out within-one-level accuracy
      (full production pipeline, fold mean per outer seed, mean of outer seeds 13 to 17).
  (4) Aggregation: min-max normalize each criterion across the floor-passing eligible
      families; score = minimum of the three normalized values (one-stage maximin);
      select the highest score; ties broken by PenalizedObj, then by menu order.

OTHER NAMED RULES (robustness only): rule_v2_two_stage (exact replica of the round-2
  code, including its fallbacks), rule_stage_one_maximin, rule_rank_sum,
  rule_equal_weight_sum, rule_weighted_sum, rule_epsilon_auc_given_obj,
  rule_epsilon_obj_given_auc, rule_structural_only; pareto_front.
  robustness.py applies all of them (and the sensitivity analyses) to the same inputs.

INPUTS
  results/structural_scan_components.csv  (retrain.py structural scan, 1,650 candidates)
  results/selection/supervised_grid_per_seed.csv and supervised_grid_bootstrap.csv
      (supervised_eval.py grid / analyze: every configuration at P20 to P60)
  revision_pipeline/cv_results_mpnet.csv and heldout_results.csv (round-2 outputs; used only
      for the round-2 reference file)
  The Legacy kNN row carries structural numbers only. Its supervised numbers exist only under
  an evaluation-only construction and are computed by robustness.py.

USAGE (run from the repository root; all paths are relative to this file)
  python -I -B revision_pipeline/select_model.py                  # default rule, writes outputs
  python -I -B revision_pipeline/select_model.py --rule v2_two_stage --no-write   # print only

OUTPUTS (results/selection/)
  selection_table.csv, selection_report.md, v2_default_config_reference.csv
"""
import argparse
import ast
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results")
SEL_DIR = os.path.join(RESULTS, "selection")
SCAN_CSV = os.path.join(RESULTS, "structural_scan_components.csv")
TABLE5_CSV = os.path.join(RESULTS, "model_comparison_table5.csv")
WINNER_JSON = os.path.join(RESULTS, "winner_and_constants.json")
CORE_CSV = os.path.join(HERE, "core_merged.csv")

LEVELS = ["Kn", "Cm", "Ap", "An", "Sn", "Ev"]
SEEDS = (13, 14, 15, 16, 17)

# ---------------------------------------------------------------------------------------
# Constants (each mirrors retrain.py; asserted against it in check_constants())
# ---------------------------------------------------------------------------------------
BASE_WEIGHTS = {"wD": 1.0, "wM": 1.0, "wT": 0.8, "wC": 0.3, "wE": 0.05}   # retrain.SCS defaults
SCS_EMPTY_LEVEL_PEN = 0.05      # retrain.SCS_EMPTY_LEVEL_PEN (second E subtraction, OvR rows only)
PEN_ACC, PEN_ENAC = 5.0, 2.5    # retrain.core_targets_from_labels penalty coefficients
BAND_MARGIN = 0.08              # retrain.core_targets_from_labels margin_acc = margin_enac
ACC_CAP = 0.85                  # retrain.run_pipeline acceptance clamp ceiling
ADEQUACY_FLOOR = 0.5            # families below this PenalizedObj are structurally inadequate
TIE_EPS_OBJ = 1e-12             # retrain.tune_ovr_model cross-setting tie tolerance

SELECTION_RULE = "balanced_maximin"   # <- the switch (see RULES at the bottom of this block)

# Families of the one-vs-rest model zoo (retrain.MODEL_GRID labels, in grid order).
OVR_FAMILIES = ["LinearSVC", "CalibratedSVC", "LogisticRegression", "RidgeClassifier",
                "SGDClassifier", "PassiveAggressive", "LDA", "QDA", "GaussianNB",
                "NearestCentroid", "KNeighborsClassifier"]
LEGACY = "Legacy kNN"
ELIGIBILITY_NOTE = {
    **{f: "eligible: one-vs-rest, per-level scores calibrated by the out-of-fold empirical CDF"
       for f in OVR_FAMILIES},
    LEGACY: ("reference only: vote rule with one global cutoff, no per-level calibrated "
             "scores, cannot run the production pipeline or be exported by retrain.py"),
}

# Production model (fixed by the authors; used as the McNemar / bootstrap reference).
PRODUCTION_FAMILY = "KNeighborsClassifier"
PRODUCTION_PARAMS = {"n_neighbors": 21, "metric": "cosine", "weights": "distance", "algorithm": "brute"}
PRODUCTION_PERCENTILE = 50

# Hyperparameters that are constant within a family (dropped from the short configuration id).
CONST_KEYS = {"class_weight", "max_iter", "tol", "random_state", "solver", "penalty",
              "metric", "weights", "algorithm"}


def short_params(params):
    return json.dumps({k: v for k, v in sorted(params.items()) if k not in CONST_KEYS})


def config_id(family, params, variant=""):
    """Configuration id shared by supervised_eval.py, retrain.py and this file."""
    return f"{family}|{short_params(params)}" + (f"|{variant}" if variant else "")


def legacy_config_id(variant, K, thresh, cutoff):
    return f"{LEGACY}|{variant}|K={int(K)}/th={float(thresh)}/c={int(cutoff)}"


PRODUCTION_CFG = config_id(PRODUCTION_FAMILY, PRODUCTION_PARAMS)


# ---------------------------------------------------------------------------------------
# Structural side: scan loading, re-aggregation, per-family bests
# ---------------------------------------------------------------------------------------
def load_scan(path=SCAN_CSV):
    """structural_scan_components.csv (written by retrain.py) as a tidy frame, scan order kept."""
    raw = pd.read_csv(path, float_precision="round_trip")
    rows = []
    for i, r in enumerate(raw.itertuples(index=False)):
        label = r.Model
        params = ast.literal_eval(r.Params) if isinstance(r.Params, str) else {}
        is_leg = label.startswith(LEGACY)
        if is_leg:
            variant = "freq-weighted" if "(freq)" in label else "unweighted"
            cid = legacy_config_id(variant, params["K"], params["thresh"], params["cutoff"])
            pct = -1
        else:
            variant = ""
            cid = config_id(label, params)
            pct = int(r.Percentile)
        per_level = json.loads(r.Eval_per_level)
        rows.append(dict(
            row=i, family=(LEGACY if is_leg else label), label=label, variant=variant,
            params=params, params_str=r.Params, cfg_id=cid, percentile=pct, mode=r.Mode,
            group=(f"{label}|{r.Mode}" if is_leg else f"{label}|{r.Params}"),
            admissible=bool(r.PassedMinAccept),
            D=float(r.Eval_D), M=float(r.Eval_M), T=float(r.Eval_T), C=float(r.Eval_C), E=float(r.Eval_E),
            n_entries=int(sum(per_level)), per_level=r.Eval_per_level,
            acceptance=float(r.Eval_acceptance), e_over_a=float(r.Eval_en_over_ac),
            SCS_stored=float(r.SCS_eval),
            Obj_stored=(float(r.PenalizedObj_eval) if pd.notna(r.PenalizedObj_eval) else np.nan),
            thresholds=(r.Thresholds if isinstance(r.Thresholds, str) else ""),
        ))
    return pd.DataFrame(rows)


def scs_vec(scan, w=BASE_WEIGHTS):
    """retrain.SCS line by line: (wD*D) - (wM*M) + (wT*T) + (wC*C) - (wE*E); -1e9 if no entry."""
    v = (w["wD"] * scan["D"].to_numpy(float)) - (w["wM"] * scan["M"].to_numpy(float)) \
        + (w["wT"] * scan["T"].to_numpy(float)) + (w["wC"] * scan["C"].to_numpy(float)) \
        - (w["wE"] * scan["E"].to_numpy(float))
    return np.where(scan["n_entries"].to_numpy() > 0, v, -1e9)


def penalty_vec(scan, T):
    """Scalar replica of retrain._penalty_from_out (Python float pow, same operation order)."""
    lo, hi = T["ACC_BAND"]
    loe, hie = T["ENAC_BAND"]
    out = np.zeros(len(scan))
    for i, (acc, enac) in enumerate(zip(scan["acceptance"].to_numpy(float), scan["e_over_a"].to_numpy(float))):
        acc = float(acc); enac = float(enac)
        p_acc = 0.0 if (lo <= acc <= hi) else min(abs(acc - lo), abs(acc - hi)) ** 2 * T["PEN_ACC"]
        p_enac = 0.0 if (loe <= enac <= hie) else min(abs(enac - loe), abs(enac - hie)) ** 2 * T["PEN_ENAC"]
        out[i] = p_acc + p_enac
    return out


def objective_vec(scan, scs, pen):
    """retrain objective: OvR rows _objective(scs - 0.05*E) (E subtracted a second time,
    retrain.tune_ovr_model), Legacy kNN rows _objective(scs) (retrain.tune_legacy_knn).
    E = 0 at every OvR row that passes min_accept, so the asymmetry never changes a value
    used for selection. NaN for rows below min_accept (never candidates)."""
    E = scan["E"].to_numpy(float)
    is_ovr = (scan["family"] != LEGACY).to_numpy()
    s = np.where(is_ovr, scs - SCS_EMPTY_LEVEL_PEN * E, scs)
    return np.where(scan["admissible"].to_numpy(), s - pen, np.nan)


def compute_targets(margin=BAND_MARGIN, acc_cap=ACC_CAP, ea_shift=0.0, pen_scale=1.0, core_csv=CORE_CSV):
    """retrain.core_targets_from_labels(Y_core, margin, margin) + the run_pipeline clamp,
    with optional perturbations (robustness): acc_cap replaces the 0.85 ceiling, ea_shift
    moves the e/a band, pen_scale multiplies both penalty coefficients."""
    import retrain as rt
    _, Y, _ = rt.load_core_from_csv(core_csv)
    T = rt.core_targets_from_labels(Y, margin_acc=margin, margin_enac=margin)
    lo, hi = T["ACC_BAND"]
    lo = max(0.25, min(lo, acc_cap))
    hi = max(lo + 1e-3, min(acc_cap, hi))
    loe, hie = T["ENAC_BAND"]
    return dict(ACC_BAND=(lo, hi), ENAC_BAND=(loe + ea_shift, hie + ea_shift),
                PEN_ACC=float(T["PEN_ACC"]) * pen_scale, PEN_ENAC=float(T["PEN_ENAC"]) * pen_scale)


class StructuralScan:
    """Scan + index structure for fast per-family picks under many weight / band settings."""

    def __init__(self, path=SCAN_CSV):
        self.path = path
        self.df = load_scan(path)
        df = self.df
        self.acc = df["acceptance"].to_numpy(float)
        self.groups = {}
        for fam in OVR_FAMILIES:
            d = df[df.family == fam]
            self.groups[fam] = [d.index[d["group"] == g].to_numpy() for g in pd.unique(d["group"])]
        self.legacy_rows = {v: df.index[(df.family == LEGACY) & (df.variant == v)].to_numpy()
                            for v in ("unweighted", "freq-weighted")}

    def evaluate(self, w=BASE_WEIGHTS, T=None):
        T = T or compute_targets()
        scs = scs_vec(self.df, w)
        pen = penalty_vec(self.df, T)
        return scs, pen, objective_vec(self.df, scs, pen)

    def _best_in_setting(self, idx, obj, scs):
        """retrain.tune_ovr_model inside one hyperparameter setting: stable sort by
        (obj, scs, acceptance) descending among admissible percentiles."""
        idx = idx[~np.isnan(obj[idx])]
        if len(idx) == 0:
            return None
        order = np.lexsort((-self.acc[idx], -scs[idx], -obj[idx]))
        return int(idx[order[0]])

    @staticmethod
    def _cross_setting(best, sel, obj, scs):
        """retrain.tune_ovr_model across settings (grid order): replace if the objective is
        larger, or equal within 1e-12 and raw SCS larger."""
        if sel is None:
            return best
        if best is None or obj[sel] > obj[best] or (abs(obj[sel] - obj[best]) < TIE_EPS_OBJ and scs[sel] > scs[best]):
            return sel
        return best

    def family_bests(self, obj, scs, include_legacy=True):
        """dict family -> scan row index of the family's structural best configuration."""
        out = {}
        for fam, groups in self.groups.items():
            best = None
            for idx in groups:
                best = self._cross_setting(best, self._best_in_setting(idx, obj, scs), obj, scs)
            if best is not None:
                out[fam] = best
        if include_legacy:
            # Bug-fixed rule for the vote family (round 2 kept the largest raw SCS with a 0.02
            # tie window, retrain.tune_legacy_knn): per variant, maximize the objective among
            # admissible vote configurations; ties by raw SCS, then acceptance; scan order last.
            picks = {v: self._best_in_setting(idx, obj, scs) for v, idx in self.legacy_rows.items()}
            u, f = picks["unweighted"], picks["freq-weighted"]
            pick = u
            # retrain.run_pipeline: the freq-weighted variant replaces the unweighted one
            # only if its (objective, SCS, acceptance) key is strictly larger.
            if f is not None and (u is None or (obj[f], scs[f], self.acc[f]) > (obj[u], scs[u], self.acc[u])):
                pick = f
            if pick is not None:
                out[LEGACY] = pick
        return out

    def menu(self, bests, obj, scs):
        """Structural part of the selection menu (one row per family, OVR_FAMILIES order, Legacy last)."""
        rows = []
        order = [f for f in OVR_FAMILIES if f in bests] + ([LEGACY] if LEGACY in bests else [])
        for fam in order:
            i = bests[fam]
            r = self.df.loc[i]
            rows.append(dict(
                family=fam, eligible=(fam != LEGACY), eligibility_note=ELIGIBILITY_NOTE[fam],
                cfg_id=r["cfg_id"], percentile=int(r["percentile"]),
                config=(f"{r['cfg_id']} P{int(r['percentile'])}" if fam != LEGACY else r["cfg_id"]),
                params_full=json.dumps(r["params"], sort_keys=True), scan_row=int(r["row"]),
                SCS=float(scs[i]), PenalizedObj=float(obj[i]), acceptance=float(r["acceptance"]),
                e_over_a=float(r["e_over_a"]), E=float(r["E"]),
            ))
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Supervised side (outputs of supervised_eval.py)
# ---------------------------------------------------------------------------------------
SUP_METRICS = ["CV_macro_AUC", "CV_macro_F1", "CV_micro_F1", "HO_within1", "HO_within1_all",
               "HO_gate", "HO_prim", "HO_mdist", "HO_f1", "HO_auc"]


class SupervisedTables:
    """Lookups (cfg_id, percentile) -> per-seed and mean values of the supervised metrics.
    OvR: results/selection/supervised_grid_per_seed.csv (supervised_eval.py).
    Legacy kNN: empty unless set_legacy() is called (robustness.py does this with its
    evaluation-only construction), so the selection itself never depends on it."""

    def __init__(self, sel_dir=SEL_DIR, require=True):
        self.sel_dir = sel_dir
        p = os.path.join(sel_dir, "supervised_grid_per_seed.csv")
        if not os.path.exists(p):
            if require:
                raise FileNotFoundError(
                    f"Supervised results not found: {p}\n"
                    "Run first:  python -I -B revision_pipeline/supervised_eval.py all")
            self.ovr = {}
            self.per_seed = None
        else:
            self.per_seed = pd.read_csv(p, float_precision="round_trip")
            self.ovr = self._index(self.per_seed, with_pct=True)
        self.legacy = {}
        bp = os.path.join(sel_dir, "supervised_grid_bootstrap.csv")
        self.boot = (pd.read_csv(bp, float_precision="round_trip").set_index(["cfg_id", "percentile"])
                     if os.path.exists(bp) else None)

    @staticmethod
    def _index(df, with_pct):
        out = {}
        for key, g in df.groupby(["cfg_id", "percentile"] if with_pct else ["cfg_id"]):
            k = (key[0], int(key[1])) if with_pct else (key[0] if isinstance(key, tuple) else key, -1)
            g = g.sort_values("seed")
            d = {}
            for m in SUP_METRICS:
                if m in g.columns:
                    vals = dict(zip(g.seed.astype(int), g[m].astype(float)))
                    d[m] = vals
            out[k] = d
        return out

    def set_legacy(self, per_seed):
        """per_seed: frame with cfg_id (legacy_config_id), seed and SUP_METRICS columns."""
        self.legacy = self._index(per_seed, with_pct=False)

    def get(self, cfg_id, percentile):
        if cfg_id.startswith(LEGACY):
            return self.legacy.get((cfg_id, -1))
        return self.ovr.get((cfg_id, int(percentile)))


def attach_supervised(menu, sup):
    """Adds <metric>_mean5, <metric>_sd5 and <metric>_seed<s> columns to a menu."""
    menu = menu.copy()
    cols = {}
    for i, r in menu.iterrows():
        d = sup.get(r["cfg_id"], r["percentile"])
        for m in SUP_METRICS:
            vals = d.get(m) if d else None
            if not vals or len(vals) != len(SEEDS):
                cols.setdefault(f"{m}_mean5", {})[i] = np.nan
                cols.setdefault(f"{m}_sd5", {})[i] = np.nan
                for s in SEEDS:
                    cols.setdefault(f"{m}_seed{s}", {})[i] = np.nan
                continue
            arr = np.array([vals[s] for s in SEEDS], dtype=float)
            cols.setdefault(f"{m}_mean5", {})[i] = float(np.mean(arr))
            cols.setdefault(f"{m}_sd5", {})[i] = float(np.std(arr, ddof=1))
            for s in SEEDS:
                cols.setdefault(f"{m}_seed{s}", {})[i] = float(vals[s])
    for c, v in cols.items():
        menu[c] = pd.Series(v)
    return menu


# ---------------------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------------------
def minmax(x):
    """Min-max normalization as in round 2 (range 0 -> every value 1.0)."""
    x = np.asarray(x, dtype=float)
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 0 else x * 0 + 1.0


def _pool(menu, floor, eligible_only=True, need=()):
    m = menu
    if eligible_only:
        m = m[m["eligible"]]
    m = m[m["PenalizedObj"] >= floor]
    for c in need:
        m = m[m[c].notna()]
    return m.copy()


def _rank(m, score_col, tie_cols=("PenalizedObj",)):
    """Stable ranking: score descending, then tie columns descending, then menu order."""
    keys = [-m[c].to_numpy(float) for c in reversed(tie_cols)] + [-m[score_col].to_numpy(float)]
    order = np.lexsort(keys)
    m = m.iloc[order].copy()
    m["rank"] = np.arange(1, len(m) + 1)
    return m


def _info(rule, ranked, score_col, extra=None):
    info = dict(rule=rule, n_considered=len(ranked), winner=None, score=np.nan, runner_up=None,
                runner_up_score=np.nan, margin=np.nan, tie_broken_by_obj=False)
    if len(ranked):
        info["winner"] = ranked.iloc[0]["family"]; info["score"] = float(ranked.iloc[0][score_col])
    if len(ranked) > 1:
        info["runner_up"] = ranked.iloc[1]["family"]; info["runner_up_score"] = float(ranked.iloc[1][score_col])
        info["margin"] = info["score"] - info["runner_up_score"]
        info["tie_broken_by_obj"] = bool(info["margin"] == 0)
    if extra:
        info.update(extra)
    return info


def rule_balanced_maximin(menu, floor=ADEQUACY_FLOOR, obj_col="PenalizedObj", auc_col="CV_macro_AUC_mean5",
                          ho_col="HO_within1_mean5", eligible_only=True):
    """PROPOSED RULE (default). One-stage maximin over three min-max normalized criteria
    among the floor-passing eligible families; ties broken by PenalizedObj."""
    m = _pool(menu, floor, eligible_only, need=(auc_col, ho_col))
    if len(m) == 0:
        return m, _info("balanced_maximin", m, "score")
    m["n_obj"] = minmax(m[obj_col]); m["n_auc"] = minmax(m[auc_col]); m["n_ho"] = minmax(m[ho_col])
    m["score"] = np.minimum(np.minimum(m["n_obj"], m["n_auc"]), m["n_ho"])
    m["limiting_criterion"] = m[["n_obj", "n_auc", "n_ho"]].idxmin(axis=1).str.replace("n_", "", regex=False)
    ranked = _rank(m, "score", (obj_col,))
    return ranked, _info("balanced_maximin", ranked, "score",
                         dict(criteria=f"{obj_col}, {auc_col}, {ho_col}", floor=floor))


def rule_v2_two_stage(menu, floor=ADEQUACY_FLOOR, cutoff=0.5, obj_col="PenalizedObj",
                      auc_col="CV_macro_AUC_mean5", ho_col="HO_within1_seed13", eligible_only=True):
    """Exact replica of the round-2 retrain.run_pipeline joint selection, including its fallbacks:
    stage one = min(n_obj, n_auc) over floor-passing families; shortlist = stage one >= cutoff;
    stage two (only with >= 2 shortlisted and held-out values for all of them) = min of the three
    criteria re-normalized inside the shortlist; otherwise the score is the stage-one score over all
    floor-passing families. Fewer than 2 floor-passing families: the code kept the purely structural
    pick (max PenalizedObj)."""
    cand = menu[menu["eligible"]] if eligible_only else menu
    cand = cand[cand[auc_col].notna()]
    cand = cand[cand[obj_col] >= floor].copy()
    if len(cand) < 2:
        allf = _pool(menu, -np.inf, eligible_only)
        ranked = _rank(allf.assign(score=allf[obj_col]), "score", ("SCS", "acceptance"))
        return ranked, _info("v2_two_stage", ranked, "score",
                             dict(mechanism="no joint rule (fewer than 2 floor-passing families)",
                                  stage2_ran=False, shortlist=""))
    cand["n_obj"] = minmax(cand[obj_col]); cand["n_auc"] = minmax(cand[auc_col])
    cand["Stage1"] = np.minimum(cand["n_obj"], cand["n_auc"])
    short = cand[cand["Stage1"] >= cutoff].copy()
    if len(short) >= 2 and short[ho_col].notna().all():
        short["s_obj"] = minmax(short[obj_col]); short["s_auc"] = minmax(short[auc_col])
        short["s_ho"] = minmax(short[ho_col])
        short["score"] = np.minimum(np.minimum(short["s_obj"], short["s_auc"]), short["s_ho"])
        ranked = _rank(short, "score", (obj_col,))
        rest = cand[~cand.index.isin(short.index)]
        ranked_all = pd.concat([ranked, rest.assign(score=np.nan)], ignore_index=False)
        info = _info("v2_two_stage", ranked, "score")
        info.update(stage2_ran=True, shortlist=";".join(short["family"]),
                    mechanism=("stage two, positive score" if info["score"] > 0 else
                               "stage two, all scores 0, tie broken by PenalizedObj"))
        return ranked_all, info
    cand["score"] = cand["Stage1"]
    ranked = _rank(cand, "score", (obj_col,))
    info = _info("v2_two_stage", ranked, "score")
    info.update(stage2_ran=False, shortlist=";".join(short["family"]),
                mechanism=f"stage-one fallback (shortlist of {len(short)})")
    return ranked, info


def rule_stage_one_maximin(menu, floor=ADEQUACY_FLOOR, obj_col="PenalizedObj", auc_col="CV_macro_AUC_mean5",
                           eligible_only=True, **_):
    """Two-criterion maximin (structure, CV AUC) over floor-passing eligible families."""
    m = _pool(menu, floor, eligible_only, need=(auc_col,))
    m["n_obj"] = minmax(m[obj_col]); m["n_auc"] = minmax(m[auc_col])
    m["score"] = np.minimum(m["n_obj"], m["n_auc"])
    ranked = _rank(m, "score", (obj_col,))
    return ranked, _info("stage_one_maximin", ranked, "score")


def _avg_ranks(x):
    from scipy.stats import rankdata
    return rankdata(-np.asarray(x, dtype=float), method="average")   # 1 = best


def rule_rank_sum(menu, criteria=("PenalizedObj", "CV_macro_AUC_mean5", "HO_within1_mean5"),
                  floor=ADEQUACY_FLOOR, eligible_only=True, **_):
    """Sum of per-criterion ranks (1 = best, average ranks for ties); lowest sum wins."""
    m = _pool(menu, floor, eligible_only, need=criteria)
    m["rank_sum"] = sum(_avg_ranks(m[c]) for c in criteria)
    m["score"] = -m["rank_sum"]
    ranked = _rank(m, "score", ("PenalizedObj",))
    info = _info("rank_sum", ranked, "score")
    info["tied_best"] = ";".join(ranked[ranked["rank_sum"] == ranked["rank_sum"].min()]["family"])
    return ranked, info


def rule_weighted_sum(menu, weights, criteria=("PenalizedObj", "CV_macro_AUC_mean5", "HO_within1_mean5"),
                      floor=ADEQUACY_FLOOR, eligible_only=True, **_):
    """Weighted sum of min-max normalized criteria."""
    m = _pool(menu, floor, eligible_only, need=criteria)
    m["score"] = sum(float(w) * minmax(m[c]) for w, c in zip(weights, criteria))
    ranked = _rank(m, "score", ("PenalizedObj",))
    return ranked, _info("weighted_sum", ranked, "score", dict(weights=list(map(float, weights))))


def rule_equal_weight_sum(menu, criteria=("PenalizedObj", "CV_macro_AUC_mean5", "HO_within1_mean5"), **kw):
    ranked, info = rule_weighted_sum(menu, [1.0 / len(criteria)] * len(criteria), criteria, **kw)
    info["rule"] = "equal_weight_sum"
    return ranked, info


def rule_epsilon_auc_given_obj(menu, eps=0.075, obj_col="PenalizedObj", auc_col="CV_macro_AUC_mean5",
                               floor=ADEQUACY_FLOOR, eligible_only=True, **_):
    """Best CV AUC among families whose objective is within eps of the best objective."""
    m = _pool(menu, floor, eligible_only, need=(auc_col,))
    m = m[m[obj_col] >= m[obj_col].max() - eps].copy()
    m["score"] = m[auc_col]
    ranked = _rank(m, "score", (obj_col,))
    return ranked, _info("epsilon_auc_given_obj", ranked, "score", dict(eps=eps))


def rule_epsilon_obj_given_auc(menu, eps=0.008, obj_col="PenalizedObj", auc_col="CV_macro_AUC_mean5",
                               floor=ADEQUACY_FLOOR, eligible_only=True, **_):
    """Best objective among families whose CV AUC is within eps of the best CV AUC."""
    m = _pool(menu, floor, eligible_only, need=(auc_col,))
    m = m[m[auc_col] >= m[auc_col].max() - eps].copy()
    m["score"] = m[obj_col]
    ranked = _rank(m, "score", (auc_col,))
    return ranked, _info("epsilon_obj_given_auc", ranked, "score", dict(eps=eps))


def rule_structural_only(menu, eligible_only=True, floor=-np.inf, **_):
    """Round-1 style: maximize PenalizedObj (ties: SCS, acceptance), no supervised evidence."""
    m = _pool(menu, floor, eligible_only)
    m["score"] = m["PenalizedObj"]
    ranked = _rank(m, "score", ("SCS", "acceptance"))
    return ranked, _info("structural_only", ranked, "score")


def pareto_front(menu, criteria=("PenalizedObj", "CV_macro_AUC_mean5", "HO_within1_mean5"),
                 floor=ADEQUACY_FLOOR, eligible_only=True):
    """Families not dominated on the given criteria (all to be maximized)."""
    m = _pool(menu, floor, eligible_only, need=criteria)
    V = m[list(criteria)].to_numpy(float)
    keep = []
    for i in range(len(m)):
        dom = any(np.all(V[j] >= V[i]) and np.any(V[j] > V[i]) for j in range(len(m)) if j != i)
        if not dom:
            keep.append(m.iloc[i]["family"])
    return keep


RULES = {
    "balanced_maximin": rule_balanced_maximin,     # proposed rule (default)
    "v2_two_stage": rule_v2_two_stage,
    "stage_one_maximin": rule_stage_one_maximin,
    "rank_sum": rule_rank_sum,
    "equal_weight_sum": rule_equal_weight_sum,
    "structural_only": rule_structural_only,
}


def select(menu, rule=None, **kw):
    """Apply the selection rule (SELECTION_RULE unless another name is given)."""
    name = rule or SELECTION_RULE
    if name not in RULES:
        raise KeyError(f"unknown rule {name!r}; available: {sorted(RULES)}")
    return RULES[name](menu, **kw)


# ---------------------------------------------------------------------------------------
# Building the menu in one call (used by retrain.py, robustness.py and main())
# ---------------------------------------------------------------------------------------
def build_menu(scan=None, w=BASE_WEIGHTS, T=None, sup=None, include_legacy=True):
    scan = scan or StructuralScan()
    scs, pen, obj = scan.evaluate(w, T)
    bests = scan.family_bests(obj, scs, include_legacy=include_legacy)
    menu = scan.menu(bests, obj, scs)
    if sup is not None:
        menu = attach_supervised(menu, sup)
    return menu


def menu_from_records(rows, sup):
    """retrain.py entry point: rows = list of dicts (family, cfg_id, percentile, SCS,
    PenalizedObj, acceptance, e_over_a, label) built from retrain's per-family records."""
    menu = pd.DataFrame(rows)
    menu["eligible"] = menu["family"] != LEGACY
    menu["eligibility_note"] = menu["family"].map(ELIGIBILITY_NOTE)
    return attach_supervised(menu, sup)


# ---------------------------------------------------------------------------------------
# Checks and reports
# ---------------------------------------------------------------------------------------
def check_constants():
    """Asserts that the constants mirrored here equal retrain.py's (a named constant when
    retrain.py defines one, otherwise the literal in its source)."""
    import inspect
    import retrain as rt
    sig = inspect.signature(rt.SCS).parameters
    assert {k: sig[k].default for k in BASE_WEIGHTS} == BASE_WEIGHTS, "SCS weights differ from retrain.SCS"
    assert rt.SCS_EMPTY_LEVEL_PEN == SCS_EMPTY_LEVEL_PEN
    assert [lab for lab, _, _ in rt.MODEL_GRID] == OVR_FAMILIES, "OVR_FAMILIES differ from retrain.MODEL_GRID"
    T = rt.core_targets_from_labels(rt.load_core_from_csv(CORE_CSV)[1])
    assert (T["PEN_ACC"], T["PEN_ENAC"]) == (PEN_ACC, PEN_ENAC)
    tsig = inspect.signature(rt.core_targets_from_labels).parameters
    assert tsig["margin_acc"].default == BAND_MARGIN == tsig["margin_enac"].default
    src_all = inspect.getsource(rt)
    src = inspect.getsource(rt.run_pipeline)
    if hasattr(rt, "ACC_BAND_CLAMP"):
        assert tuple(rt.ACC_BAND_CLAMP) == (0.25, ACC_CAP)
    else:
        assert f"lo = max(0.25, min(lo, {ACC_CAP}))" in src and f"min({ACC_CAP}, hi)" in src, "acceptance clamp"
    if hasattr(rt, "ADEQUACY_FLOOR"):
        assert rt.ADEQUACY_FLOOR == ADEQUACY_FLOOR
    else:
        assert f'cand["PenalizedObj"] >= {ADEQUACY_FLOOR}' in src, "adequacy floor"
    assert f'"{LEGACY}"' in src_all
    return True


def verify_against_table5(menu, table5=TABLE5_CSV):
    """The OvR per-family picks re-derived here must equal retrain.py's
    model_comparison_table5.csv. The Legacy kNN row is compared separately in the report: the
    regenerated table uses the bug-fixed vote rule, while round 2 kept the largest raw SCS within a
    0.02 tie window."""
    if not os.path.exists(table5):
        return "table5 not found"
    t5 = pd.read_csv(table5, float_precision="round_trip")
    t5["family"] = t5["Model"].str.replace(r"\s*\(.*\)$", "", regex=True)
    bad = []
    for r in menu[menu.family != LEGACY].itertuples():
        t = t5[t5.family == r.family]
        if t.empty:
            bad.append(f"{r.family}: missing in table5"); continue
        t = t.iloc[0]
        if not (t["PenalizedObj"] == r.PenalizedObj and t["Eval_SCS"] == r.SCS):
            bad.append(f"{r.family}: table5 {t['PenalizedObj']!r}/{t['Eval_SCS']!r} vs {r.PenalizedObj!r}/{r.SCS!r}")
    return "identical" if not bad else "; ".join(bad)


def verify_against_supervised_eval(menu, sel_dir=SEL_DIR, tol=1e-12):
    """supervised_eval.py derives the same per-family structural bests on its own
    (matched_supervised.csv); configurations must be identical and values agree within tol."""
    p = os.path.join(sel_dir, "matched_supervised.csv")
    if not os.path.exists(p):
        return "matched_supervised.csv not found"
    m = pd.read_csv(p, float_precision="round_trip")
    m = m[~m["supplementary"].astype(bool)].set_index("family")
    bad, md = [], 0.0
    for r in menu[menu.family != LEGACY].itertuples():
        if r.family not in m.index:
            bad.append(f"{r.family}: missing"); continue
        q = m.loc[r.family]
        if q["cfg_id"] != r.cfg_id or int(q["percentile"]) != int(r.percentile):
            bad.append(f"{r.family}: {q['cfg_id']} P{q['percentile']} vs {r.cfg_id} P{r.percentile}")
        for c in ("PenalizedObj", "CV_macro_AUC_mean5", "HO_within1_mean5"):
            d = abs(float(q[c]) - float(getattr(r, c)))
            md = max(md, d)
            if d > tol:
                bad.append(f"{r.family} {c}: {q[c]!r} vs {getattr(r, c)!r}")
    return (f"identical configurations, values within {md:.1e}" if not bad else "; ".join(bad))


def fmt(v, nd=6):
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def md_table(df, cols, nd=6):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r.get(c, ""), nd) for c in cols) + " |")
    return "\n".join(lines)


# Round-2 default configurations (retrain.CV_NAME_MAP; cv_metrics.py / methodology_upgrades.py
# evaluated each family only here, at P50).
V2_CV_NAME = {"SGDClassifier": "SGD-hinge-1e4", "LogisticRegression": "LogReg-C1", "RidgeClassifier": "Ridge-a1",
              "LinearSVC": "LinearSVC-C1", "KNeighborsClassifier": "kNN-21", "NearestCentroid": "NearestCentroid",
              "PassiveAggressive": "PassiveAggressive-C1", "GaussianNB": "GaussianNB", "LDA": "LDA", "QDA": "QDA",
              "CalibratedSVC": "CalibratedSVC"}
V2_DEFAULT_CFG = {
    "SGDClassifier": 'SGDClassifier|{"alpha": 0.0001, "loss": "hinge"}',
    "LogisticRegression": 'LogisticRegression|{"C": 1.0}',
    "RidgeClassifier": 'RidgeClassifier|{"alpha": 1.0}',
    "LinearSVC": 'LinearSVC|{"C": 1.0}',
    "KNeighborsClassifier": 'KNeighborsClassifier|{"n_neighbors": 21}',
    "NearestCentroid": "NearestCentroid|{}",
    "PassiveAggressive": 'PassiveAggressive|{"C": 1.0}',
    "GaussianNB": 'GaussianNB|{"var_smoothing": 1e-09}',
    "LDA": 'LDA|{"shrinkage": "auto"}|PCA24',
    "QDA": 'QDA|{"reg_param": 0.05}|PCA24',
    "CalibratedSVC": 'CalibratedSVC|{"C": 0.5}',
}
# Families methodology_upgrades.py ran (held-out protocol), and the rows the round-2 code read
# from heldout_results.csv (retrain.run_pipeline HO_MAP: LinearSVC missing, so stage two never ran).
V2_HO_FAMILIES = ["KNeighborsClassifier", "CalibratedSVC", "LogisticRegression", "NearestCentroid", "LinearSVC"]
V2_HO_MAP_CODE = {"KNeighborsClassifier": "kNN-21", "LogisticRegression": "LogReg-C1",
                  "NearestCentroid": "NearestCentroid", "CalibratedSVC": "CalibratedSVC"}


def v2_inputs(scan_menu, sup):
    """Round-2 (mismatched) supervised inputs next to the structural best configurations:
    CV macro AUC = mean of the five repeats in cv_results_mpnet.csv (round-2 cv_metrics.py output,
    default configuration at P50); held-out within-one seed 13 = the default configuration at P50
    and outer seed 13 of supervised_eval.py, which is bit-identical to round-2 methodology_upgrades.py
    before rounding (heldout_results.csv holds the 3-decimal values; its LinearSVC-C1 row was added in
    round 3 when methodology_upgrades.py became its producing script, and the round-2 code path below
    still leaves LinearSVC out, as the committed round-2 code did)."""
    cvp = os.path.join(HERE, "cv_results_mpnet.csv")
    if not os.path.exists(cvp):
        raise FileNotFoundError(f"{cvp} missing (round-2 cv_metrics.py output)")
    cv = pd.read_csv(cvp, float_precision="round_trip").groupby("model")["macro_AUC"].mean()
    base = scan_menu[scan_menu.eligible].copy()
    base["v2_cv_config"] = base.family.map(lambda f: V2_CV_NAME[f] + " P50")
    base["v2_default_cfg_id"] = base.family.map(V2_DEFAULT_CFG)
    base["CV_macro_AUC_mean5"] = base.family.map(lambda f: float(cv[V2_CV_NAME[f]]))

    def ho13(f):
        if f not in V2_HO_FAMILIES:
            return np.nan
        d = sup.get(V2_DEFAULT_CFG[f], 50)
        return float(d["HO_within1"][13])
    base["HO_within1_seed13"] = base.family.map(ho13)
    return base.reset_index(drop=True)


def v2_reference(scan_menu, sup, out_csv):
    """Round-2 inputs (default configurations at P50) and the exact round-2 two-stage score, in
    three variants: exact inputs; inputs rounded to 3 decimals as printed in Tables 8a/8b; and the
    path the committed round-2 code actually took (its HO_MAP had no LinearSVC entry, so stage two
    did not run; V2_HO_MAP_CODE reproduces that map)."""
    hop = os.path.join(HERE, "heldout_results.csv")
    if not os.path.exists(hop):
        return None, "round-2 reference input heldout_results.csv missing"
    base = v2_inputs(scan_menu, sup)
    ho = pd.read_csv(hop, float_precision="round_trip").set_index("model")
    # the exact seed-13 values must round to the published 3-decimal values
    for f, tag in V2_HO_MAP_CODE.items():
        v = float(base.loc[base.family == f, "HO_within1_seed13"].iloc[0])
        assert abs(round(v, 3) - float(ho.loc[tag, "within1"])) < 1e-12, (f, v, ho.loc[tag, "within1"])
    out = []
    variants = {
        "v2_exact": base,
        "v2_rounded_3dp_as_printed": base.assign(**{c: base[c].round(3) for c in
                                                    ["PenalizedObj", "CV_macro_AUC_mean5", "HO_within1_seed13"]}),
        "v2_committed_code_path": base.assign(HO_within1_seed13=base.family.map(
            lambda f: float(ho.loc[V2_HO_MAP_CODE[f], "within1"]) if f in V2_HO_MAP_CODE else np.nan)),
    }
    for tag, menu in variants.items():
        ranked, info = rule_v2_two_stage(menu)
        t = menu.merge(ranked[["family"] + [c for c in ["n_obj", "n_auc", "Stage1", "s_obj", "s_auc", "s_ho", "score"]
                                            if c in ranked.columns]], on="family", how="left")
        t.insert(0, "variant", tag)
        t["winner"] = info["winner"]; t["winner_score"] = info["score"]
        t["stage2_ran"] = info.get("stage2_ran"); t["mechanism"] = info.get("mechanism")
        out.append(t)
    ref = pd.concat(out, ignore_index=True)
    keep = ["variant", "family", "config", "v2_cv_config", "v2_default_cfg_id", "PenalizedObj", "CV_macro_AUC_mean5", "HO_within1_seed13",
            "n_obj", "n_auc", "Stage1", "s_obj", "s_auc", "s_ho", "score", "winner", "winner_score", "stage2_ran",
            "mechanism"]
    ref = ref[[c for c in keep if c in ref.columns]].rename(columns={
        "config": "structural_best_config", "CV_macro_AUC_mean5": "v2_CV_macro_AUC_mean5_P50",
        "HO_within1_seed13": "v2_HO_within1_seed13_P50", "score": "v2_two_stage_score"})
    if out_csv:
        ref.to_csv(out_csv, index=False)
    return ref, "ok"


BOOT_COLS = ["CV_macro_AUC_avg5_ci95_pctboot_lo", "CV_macro_AUC_avg5_ci95_pctboot_hi",
             "dCV_macro_AUC_vs_kNN21P50_avg5_ci95_pctboot_lo", "dCV_macro_AUC_vs_kNN21P50_avg5_ci95_pctboot_hi",
             "HO_within1_pooled_avg5_ci95_pctboot_lo", "HO_within1_pooled_avg5_ci95_pctboot_hi",
             "dHO_within1_pooled_vs_kNN21P50_avg5_ci95_pctboot_lo",
             "dHO_within1_pooled_vs_kNN21P50_avg5_ci95_pctboot_hi",
             "CV_macro_F1_avg5_ci95_pctboot_lo", "CV_macro_F1_avg5_ci95_pctboot_hi"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scan", default=SCAN_CSV)
    ap.add_argument("--sel-dir", default=SEL_DIR)
    ap.add_argument("--rule", default=SELECTION_RULE, choices=sorted(RULES))
    ap.add_argument("--no-write", action="store_true", help="print only, write nothing")
    a = ap.parse_args()
    check_constants()
    sup = SupervisedTables(a.sel_dir)
    scan = StructuralScan(a.scan)
    T = compute_targets()
    menu = build_menu(scan, BASE_WEIGHTS, T, sup)
    t5_check = verify_against_table5(menu)
    se_check = verify_against_supervised_eval(menu, a.sel_dir)
    ranked, info = select(menu, a.rule)
    ranked_v2, info_v2 = rule_v2_two_stage(menu, ho_col="HO_within1_seed13")
    ranked_v2m, info_v2m = rule_v2_two_stage(menu, ho_col="HO_within1_mean5")
    # selection table: every family, incl. Legacy kNN reference row and LDA/QDA
    tab = menu.copy()
    tab["floor_pass"] = tab["PenalizedObj"] >= ADEQUACY_FLOOR
    sc_cols = [c for c in ["n_obj", "n_auc", "n_ho", "score", "limiting_criterion", "rank"] if c in ranked.columns]
    tab = tab.merge(ranked[["family"] + sc_cols], on="family", how="left")
    tab["selected"] = tab["family"] == info["winner"]
    tab["selection_rule"] = info["rule"]
    tab = tab.merge(ranked_v2[["family", "score"]].rename(columns={"score": "v2_two_stage_score_HOseed13"}),
                    on="family", how="left")
    if sup.boot is not None:
        for c in BOOT_COLS:
            tab[c] = [float(sup.boot.loc[(r.cfg_id, r.percentile), c]) if (r.cfg_id, r.percentile) in sup.boot.index
                      else np.nan for r in tab.itertuples()]
    lead = ["family", "eligible", "floor_pass", "config", "SCS", "PenalizedObj", "acceptance", "e_over_a",
            "CV_macro_AUC_mean5", "CV_macro_AUC_sd5", "CV_macro_AUC_avg5_ci95_pctboot_lo",
            "CV_macro_AUC_avg5_ci95_pctboot_hi", "CV_macro_F1_mean5", "HO_within1_mean5", "HO_within1_sd5",
            "HO_within1_seed13", "HO_within1_all_mean5", "HO_gate_mean5", "HO_gate_seed13",
            "n_obj", "n_auc", "n_ho", "score", "limiting_criterion", "rank", "selected", "selection_rule",
            "v2_two_stage_score_HOseed13", "eligibility_note", "cfg_id", "percentile", "params_full", "scan_row"]
    tab = tab[[c for c in lead if c in tab.columns] + [c for c in tab.columns if c not in lead]]
    print(tab[["family", "eligible", "floor_pass", "PenalizedObj", "CV_macro_AUC_mean5", "HO_within1_mean5",
               "score", "selected"]].to_string(index=False))
    print(f"\n[{info['rule']}] winner {info['winner']} score {info['score']!r}; runner-up {info['runner_up']} "
          f"{info['runner_up_score']!r}; margin {info['margin']!r}")
    print(f"[v2_two_stage, HO seed 13] winner {info_v2['winner']} {info_v2['score']!r} ({info_v2.get('mechanism')})")
    print(f"[table5 check] {t5_check}")
    print(f"[supervised_eval matched_supervised.csv check] {se_check}")
    if a.no_write:
        return
    os.makedirs(a.sel_dir, exist_ok=True)
    tab.to_csv(os.path.join(a.sel_dir, "selection_table.csv"), index=False)
    ref, status = v2_reference(menu, sup, os.path.join(a.sel_dir, "v2_default_config_reference.csv"))
    write_report(a, tab, ranked, info, info_v2, info_v2m, ref, status, t5_check + "; supervised_eval.py "
                 "matched_supervised.csv: " + se_check, T)


def write_report(a, tab, ranked, info, info_v2, info_v2m, ref, ref_status, t5_check, T):
    L = []
    w = L.append
    w("# Model selection (generated by select_model.py; do not edit by hand)")
    w("")
    w(f"Rule in use: `{info['rule']}` (SELECTION_RULE in select_model.py). Status: proposed, pending the "
      "authors' final confirmation.")
    w("")
    w("Rule, in words:")
    w("1. Eligible families: every one-vs-rest family of the model zoo, because each produces per-level scores "
      "calibrated by the out-of-fold empirical CDF and can therefore run the production pipeline (per-level "
      "thresholds, top-2 margin collapse, soft tier). Legacy kNN (a vote rule with one global cutoff) is not "
      "eligible and is shown as a structural reference row.")
    w(f"2. Adequacy floor: PenalizedObj >= {ADEQUACY_FLOOR}.")
    w("3. Criteria, each at the family's structural best configuration and percentile: PenalizedObj (Eq. 19, on "
      "the 1,650 unlabeled candidates), CV macro AUC (repeated stratified 5x5 cross-validation, mean of seeds 13 "
      "to 17) and held-out within-one-level accuracy (full production pipeline, mean of outer seeds 13 to 17).")
    w("4. Score = minimum of the three min-max normalized criteria (normalized over the floor-passing eligible "
      "families). Highest score wins; ties broken by PenalizedObj.")
    w("")
    w(f"Structural inputs: `{os.path.relpath(a.scan, REPO).replace(os.sep, '/')}` re-aggregated at the baseline weights "
      f"{BASE_WEIGHTS} and bands ACC {tuple(round(x, 6) for x in T['ACC_BAND'])}, "
      f"e/a {tuple(round(x, 6) for x in T['ENAC_BAND'])}. Cross-checks of the one-vs-rest picks: "
      f"results/model_comparison_table5.csv: {t5_check}.")
    w("Supervised inputs: `results/selection/supervised_grid_per_seed.csv` (supervised_eval.py), taken at exactly "
      "the configuration and percentile in the config column. The Legacy kNN row carries no supervised numbers here; "
      "robustness.py evaluates it under an evaluation-only construction (votes against the outer-training verbs, vote "
      "sums calibrated by an inner-fold empirical CDF).")
    w("")
    w("## Selection table")
    w("")
    cols = ["family", "eligible", "floor_pass", "config", "PenalizedObj", "CV_macro_AUC_mean5", "HO_within1_mean5",
            "HO_within1_seed13", "HO_gate_mean5", "n_obj", "n_auc", "n_ho", "score", "selected"]
    t = tab.sort_values(["score", "PenalizedObj"], ascending=False, na_position="last")
    w(md_table(t, cols))
    w("")
    w(f"**Selected: {info['winner']}** (score {info['score']:.10f}); runner-up {info['runner_up']} "
      f"({info['runner_up_score']:.10f}); margin {info['margin']:.10f}.")
    win = ranked.iloc[0]
    w(f"The winner's score is limited by its normalized {win['limiting_criterion']} "
      f"(n_obj {win['n_obj']:.6f}, n_auc {win['n_auc']:.6f}, n_ho {win['n_ho']:.6f}).")
    w("")
    if "dCV_macro_AUC_vs_kNN21P50_avg5_ci95_pctboot_lo" in tab.columns:
        w("## How far apart the candidates are (paired percentile bootstrap, supervised_eval.py)")
        w("")
        w("Differences to the production model kNN-21 P50 (other minus kNN-21), 95% intervals from 10,000 paired "
          "resamples of the 381 verbs, statistic averaged over the five repeats (CV) or outer seeds (held-out, "
          "within-one pooled over accepted verbs).")
        w("")
        tt = tab[tab["eligible"] & tab["floor_pass"]].sort_values("score", ascending=False)
        w(md_table(tt, ["family", "config", "CV_macro_AUC_mean5", "dCV_macro_AUC_vs_kNN21P50_avg5_ci95_pctboot_lo",
                        "dCV_macro_AUC_vs_kNN21P50_avg5_ci95_pctboot_hi", "HO_within1_mean5",
                        "dHO_within1_pooled_vs_kNN21P50_avg5_ci95_pctboot_lo",
                        "dHO_within1_pooled_vs_kNN21P50_avg5_ci95_pctboot_hi"]))
        w("")
    leg = tab[tab["family"] == LEGACY]
    if len(leg):
        r = leg.iloc[0]
        w("## Legacy kNN (structural reference row, not eligible)")
        w("")
        w(f"Bug-fixed structural best: `{r['config']}`, PenalizedObj {r['PenalizedObj']:.10f}, SCS {r['SCS']:.10f}, "
          f"acceptance {r['acceptance']:.6f}. It is above the adequacy floor but is not eligible: "
          f"{ELIGIBILITY_NOTE[LEGACY]}.")
        if os.path.exists(TABLE5_CSV):
            t5 = pd.read_csv(TABLE5_CSV, float_precision="round_trip")
            t5l = t5[t5["Model"].str.startswith(LEGACY)]
            if len(t5l):
                q = t5l.iloc[0]
                w(f"The regenerated table (results/model_comparison_table5.csv) applies the same bug-fixed rule and "
                  f"shows `{q['Model']}` with PenalizedObj {float(q['PenalizedObj']):.10f}. The round-2 table showed "
                  "`Legacy kNN (K=5/th=0.32/c=32)` (frequency-weighted) with PenalizedObj -1.1967902490, because the "
                  "round-2 vote rule kept the largest raw SCS within a 0.02 tie window instead of the largest "
                  "penalized objective.")
        w("Its supervised numbers (evaluation-only construction) and the effect of admitting it are reported in "
          "robustness_report.md.")
        w("")
    w("## The round-2 two-stage rule on the same matched inputs (for disclosure)")
    w("")
    w(f"- Held-out seed 13 (as in round 2): winner **{info_v2['winner']}** with {info_v2['score']:.10f} "
      f"({info_v2.get('mechanism')}; shortlist {info_v2.get('shortlist')}).")
    w(f"- Held-out mean of seeds 13 to 17: winner **{info_v2m['winner']}** with {info_v2m['score']:.10f} "
      f"({info_v2m.get('mechanism')}).")
    w("- Under the two-stage form the production family scores 0 because its CV AUC is the lowest of the "
      "shortlist after re-normalization inside the shortlist; see robustness_report.md for how small the AUC "
      "differences are.")
    w("")
    if ref is not None:
        w("## Round-2 reference (default configurations at P50, as printed in round 2)")
        w("")
        for v in ref["variant"].unique():
            r = ref[ref.variant == v].iloc[0]
            w(f"- {v}: winner {r['winner']} with {r['winner_score']:.10f} (stage two ran: {r['stage2_ran']}; "
              f"{r['mechanism']})")
        w("")
        w("Values: `results/selection/v2_default_config_reference.csv`.")
    else:
        w(f"Round-2 reference not written: {ref_status}")
    w("")
    w("Robustness of this choice: `results/selection/robustness_report.md` (robustness.py).")
    txt = "\n".join(L)
    for bad in (chr(0x2013), chr(0x2014)):
        assert bad not in txt
    with open(os.path.join(a.sel_dir, "selection_report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
