#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
robustness.py - how much the round-3 model selection depends on the choices behind it.

Every analysis uses the same inputs and the same functions as select_model.py (structural
re-aggregation of results/structural_scan_components.csv, per-family structural best with the
bug-fixed rule, MATCHED supervised numbers from results/selection/supervised_grid_per_seed.csv,
rules as named functions). Nothing here changes the selection; it only reports how stable it is.

ANALYSES (outputs in results/selection/)
  1. Aggregation alternatives (robustness_aggregation.csv): on six family sets (round-2 inputs
     with the 7-family menu of the round-2 version of paper_results.py and the 9 floor-passing
     families of round-2 Table 8a; matched inputs without and with Legacy kNN, held-out seed 13
     and 5-seed mean): Spearman(PenalizedObj, CV AUC); two-criterion weighted sum (exact switch
     points), rank-sum, maximin (= round-2 stage one), Pareto front, epsilon-constraint rules at
     the round-2 epsilons and as functions of epsilon; three-criterion weighted-sum simplex
     (share of the simplex won), equal weights, rank-sum, one-stage maximin, Pareto front; the
     round-2 two-stage rule; the proposed rule.
  2. Held-out variants (robustness_heldout_variants.csv): proposed rule and round-2 two-stage
     rule with held-out within-one from outer seed 13 alone, each seed 14 to 17 alone, the
     5-seed mean, and within-one over ALL held-out verbs (rejected verbs count as misses).
  3. Legacy kNN admitted (robustness_legacy_knn_supervised.csv): the vote family evaluated
     under the same partitions with an EVALUATION-ONLY construction (votes against the
     outer-training verbs; per-level vote sums calibrated by an empirical CDF fitted on
     inner-fold vote sums, inner KFold(5, shuffle, seed 13); hard labels = the vote rule itself,
     no percentile thresholds, no soft tier). Production cannot use this model (retrain.py cannot
     export it); the numbers only show what admitting it would do.
  4. SCS weights (robustness_scs_weights.csv, robustness_scs_config_reach.csv): 500 joint draws
     (numpy default_rng(13); each weight times U(0.5, 1.5), drawn in the order wD, wM, wT, wC, wE,
     the same stream as the round-2 version of paper_results.py) and single-weight x0.5, x0.75,
     x1.25, x1.5. For every setting the objective is recomputed (bands and penalties unchanged),
     each family's best configuration is re-selected, its matched supervised numbers are pulled,
     and the proposed rule, the structural-only rule and the round-2 two-stage rule are applied.
  5. Guardrails and penalties (robustness_guardrails.csv, robustness_guardrail_family_picks.csv):
     penalty coefficients x0, x0.5, x2, x4; acceptance target 0.80 / 0.85 / 0.90; e/a band shifted
     by -0.05 / +0.05; band margin 0.04 / 0.08 / 0.12; adequacy floor 0.4 / 0.5 / 0.6; and, for the
     two-stage rule only, the stage-one cutoff 0.4 / 0.5 / 0.6. One change at a time.
  6. Decision margins (robustness_decision_margins.csv): the smallest one-at-a-time change to one
     family's criterion value that changes the proposed rule's winner.
  Report: robustness_report.md (generated).

USAGE (from the repository root)
  python -I -B revision_pipeline/robustness.py            # everything (about one to two minutes)
  python -I -B revision_pipeline/robustness.py --only weights --no-write
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import select_model as sm  # noqa: E402

REPO = sm.REPO
SEEDS = sm.SEEDS
LEGACY = sm.LEGACY
KNN = sm.PRODUCTION_FAMILY
BASE = dict(sm.BASE_WEIGHTS)
EMB_NPZ = os.path.join(HERE, "all_embeddings.npz")
TOP2_MARGIN = 0.060              # methodology_upgrades.py / retrain.TOP2_MARGIN_DEFAULT
N_DRAWS, DRAW_SEED = 500, 13     # round-2 SCS weight sensitivity protocol
SINGLE_FACTORS = (0.5, 0.75, 1.25, 1.5)
V2_EPS_OBJ, V2_EPS_AUC = 0.075, 0.008     # epsilons quoted in the round-2 robustness paragraph
CV7 = ["SGDClassifier", "LogisticRegression", "RidgeClassifier", "LinearSVC", "KNeighborsClassifier",
       "NearestCentroid", "PassiveAggressive"]   # CVMAP menu of the round-2 version of paper_results.py
SHORT = {"KNeighborsClassifier": "kNN-21", "CalibratedSVC": "CalSVC", "LogisticRegression": "LogReg",
         "NearestCentroid": "NCentroid", "LinearSVC": "LinSVC", "RidgeClassifier": "Ridge",
         "PassiveAggressive": "PA", "SGDClassifier": "SGD", "GaussianNB": "GNB", "LDA": "LDA", "QDA": "QDA",
         LEGACY: "LegacykNN"}
OBJ, AUC, HO5, HO13 = "PenalizedObj", "CV_macro_AUC_mean5", "HO_within1_mean5", "HO_within1_seed13"


# =========================================================================================
# Legacy kNN under the evaluation-only construction
# =========================================================================================
class LegacyEvaluator:
    """CV (5 repeats) and held-out (outer seeds 13 to 17) for one vote configuration, on the
    same MultilabelStratifiedKFold partitions as supervised_eval.py. Per outer fold: hard labels
    = retrain.legacy_knn_predict_eval against the outer-training verbs; scores = vote sums mapped
    through per-level empirical CDFs (retrain._fit_cdf_per_class) fitted on inner out-of-fold vote
    sums (KFold(5, shuffle=True, random_state=13) over the outer-training verbs). Held-out: top-2
    margin collapse at 0.060 on the calibrated scores, primary = highest calibrated score among
    the assigned levels (methodology_upgrades.py evaluate), metrics averaged over the 5 folds."""

    def __init__(self):
        import cv_metrics as cvm
        import retrain as rt
        self.rt = rt
        assert abs(rt.TOP2_MARGIN_DEFAULT - TOP2_MARGIN) < 1e-12, "top-2 margin differs from retrain.py"
        verbs, Y, freq = cvm.load_core(sm.CORE_CSV)
        z = np.load(EMB_NPZ)
        vec = dict(zip([str(x) for x in z["words"]], z["vectors"]))
        self.X = np.array([vec[v] for v in verbs], dtype=np.float64)
        self.Y, self.freq = Y, freq
        self.cache = {}

    def _votes(self, Xq, Xr, Yr, K, th, co, fr):
        return self.rt.legacy_knn_predict_eval(Xq, Xr, Yr, K, th, co,
                                               freq=(None if fr is None else np.asarray(fr, float)),
                                               core_vocab=None)

    def _member(self, tr, te, K, th, co, use_freq):
        from sklearn.model_selection import KFold
        X, Y, freq = self.X, self.Y, self.freq
        f_tr = freq[tr] if use_freq else None
        hard_te, votes_te = self._votes(X[te], X[tr], Y[tr], K, th, co, f_tr)
        S_oof = np.zeros((len(tr), 6))
        for itr, iva in KFold(n_splits=5, shuffle=True, random_state=13).split(X[tr]):
            _, v = self._votes(X[tr][iva], X[tr][itr], Y[tr][itr], K, th, co, (f_tr[itr] if use_freq else None))
            S_oof[iva] = v
        cdfs = self.rt._fit_cdf_per_class(S_oof)
        return hard_te, self.rt._apply_cdf_per_class(votes_te, cdfs)

    @staticmethod
    def _top2_collapse(S01, hard, delta=TOP2_MARGIN):
        hard = hard.copy()
        for i in range(len(hard)):
            on = np.where(hard[i] == 1)[0]
            if len(on) >= 2:
                s_on = S01[i, on]
                o = np.argsort(s_on)[::-1]
                if s_on[o[0]] - s_on[o[1]] >= delta:
                    hard[i] = 0
                    hard[i, on[o[0]]] = 1
        return hard

    @staticmethod
    def _evaluate(S01, hard, Y_te):
        from sklearn.metrics import f1_score, roc_auc_score
        accepted = hard.sum(1) > 0
        prim = np.where(accepted, np.argmax(S01 * hard, axis=1), -1)
        idx = np.where(accepted)[0]
        prim_in = np.mean([Y_te[i, prim[i]] == 1 for i in idx]) if len(idx) else 0.0
        true_ranks = [np.where(Y_te[i] == 1)[0] for i in idx]
        dists = [min(abs(prim[i] - t) for t in true_ranks[j]) for j, i in enumerate(idx)]
        within1 = np.mean([d <= 1 for d in dists]) if dists else 0.0
        within1_all = float(np.sum([d <= 1 for d in dists])) / len(Y_te)
        mdist = np.mean(dists) if dists else np.nan
        f1 = f1_score(Y_te, hard, average="macro", zero_division=0)
        try:
            auc = float(np.mean([roc_auc_score(Y_te[:, j], S01[:, j]) for j in range(6)]))
        except ValueError:
            auc = np.nan
        return dict(gate=accepted.mean(), prim=prim_in, within1=within1, within1_all=within1_all,
                    mdist=mdist, f1=f1, auc=auc)

    def per_seed(self, cfg_id, variant, K, thresh, cutoff):
        key = cfg_id
        if key in self.cache:
            return self.cache[key]
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        from sklearn.metrics import f1_score, roc_auc_score
        X, Y = self.X, self.Y
        K, th, co, uf = int(K), float(thresh), int(cutoff), (variant == "freq-weighted")
        rows = []
        for sd in SEEDS:
            pred = np.zeros_like(Y)
            S = np.zeros(Y.shape)
            folds = []
            for tr, te in MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=sd).split(X, Y):
                hard, S_te = self._member(tr, te, K, th, co, uf)
                pred[te] = hard
                S[te] = S_te
                folds.append(self._evaluate(S_te, self._top2_collapse(S_te, hard), Y[te]))
            fm = pd.DataFrame(folds).mean(numeric_only=True)
            rows.append(dict(cfg_id=cfg_id, family=LEGACY, variant=variant, K=K, thresh=th, cutoff=co,
                             score_mode="cdf_kfold (evaluation-only)", seed=sd,
                             CV_macro_AUC=float(np.mean([roc_auc_score(Y[:, j], S[:, j]) for j in range(6)])),
                             CV_macro_F1=f1_score(Y, pred, average="macro", zero_division=0),
                             CV_micro_F1=f1_score(Y, pred, average="micro", zero_division=0),
                             oof_acceptance=float((pred.sum(1) > 0).mean()),
                             HO_gate=float(fm["gate"]), HO_within1=float(fm["within1"]),
                             HO_within1_all=float(fm["within1_all"]), HO_prim=float(fm["prim"]),
                             HO_mdist=float(fm["mdist"]), HO_f1=float(fm["f1"]), HO_auc=float(fm["auc"])))
        self.cache[key] = rows
        return rows

    def frame(self):
        return pd.DataFrame([r for rows in self.cache.values() for r in rows])


# =========================================================================================
# Context: inputs shared by every analysis
# =========================================================================================
class Ctx:
    def __init__(self, scan_path=sm.SCAN_CSV, sel_dir=sm.SEL_DIR):
        sm.check_constants()
        self.scan = sm.StructuralScan(scan_path)
        self.sup = sm.SupervisedTables(sel_dir)
        self.T0 = sm.compute_targets()
        self.leg = LegacyEvaluator()
        self.sel_dir = sel_dir

    def menu(self, w=None, T=None):
        """Selection menu (11 one-vs-rest families + Legacy kNN) with matched supervised numbers;
        Legacy kNN configurations are evaluated on demand (evaluation-only construction)."""
        w = w or BASE
        T = T or self.T0
        scs, pen, obj = self.scan.evaluate(w, T)
        bests = self.scan.family_bests(obj, scs, include_legacy=True)
        if LEGACY in bests:
            r = self.scan.df.loc[bests[LEGACY]]
            p = r["params"]
            if r["cfg_id"] not in self.leg.cache:
                self.leg.per_seed(r["cfg_id"], r["variant"], p["K"], p["thresh"], p["cutoff"])
                self.sup.set_legacy(self.leg.frame())
        menu = self.scan.menu(bests, obj, scs)
        return sm.attach_supervised(menu, self.sup), bests, obj, scs


# =========================================================================================
# Small helpers for the robustness analyses
# =========================================================================================
def nrm(x):
    return sm.minmax(x)


def spearman(a, b):
    from scipy import stats
    r = stats.spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


def ranks_best1(x):
    from scipy import stats
    return stats.rankdata(-np.asarray(x, float), method="average")


def ws_intervals(names, a, b, step=1e-4):
    """Weighted sum w*a + (1-w)*b on a 1e-4 grid of w (argmax, first family on ties), plus the
    exact switch points between consecutive winners."""
    ws = np.round(np.arange(0, 1 + step / 2, step), 10)
    win = [names[int(np.argmax(w * a + (1 - w) * b))] for w in ws]
    out, start = [], 0
    for k in range(1, len(ws) + 1):
        if k == len(ws) or win[k] != win[start]:
            out.append([ws[start], ws[k - 1], win[start]])
            start = k
    exact = []
    for (lo1, hi1, f), (lo2, hi2, g) in zip(out[:-1], out[1:]):
        i, j = names.index(f), names.index(g)
        den = (a[i] - a[j]) - (b[i] - b[j])
        exact.append((f, g, float((b[j] - b[i]) / den) if den != 0 else np.nan))
    return out, exact


def pareto(names, M):
    keep = []
    for i in range(len(names)):
        dom = any(all(M[j] >= M[i]) and any(M[j] > M[i]) for j in range(len(names)) if j != i)
        if not dom:
            keep.append(names[i])
    return keep


def crit_set(menu, ho_col, include_legacy, floor=sm.ADEQUACY_FLOOR):
    """Floor-passing families of a menu with columns family, PenalizedObj, CV_AUC, HO_within1."""
    m = menu if include_legacy else menu[menu["eligible"]]
    m = m[m[OBJ] >= floor]
    return pd.DataFrame(dict(family=m["family"].to_numpy(), PenalizedObj=m[OBJ].to_numpy(float),
                             CV_AUC=m[AUC].to_numpy(float), HO_within1=m[ho_col].to_numpy(float)))


def two_stage_info(menu, ho_col, include_legacy, floor=sm.ADEQUACY_FLOOR, cutoff=0.5):
    ranked, info = sm.rule_v2_two_stage(menu, floor=floor, cutoff=cutoff, ho_col=ho_col,
                                        eligible_only=not include_legacy)
    pool = menu if include_legacy else menu[menu["eligible"]]
    n_floor = int(((pool[OBJ] >= floor) & pool[AUC].notna()).sum())
    if n_floor < 2:
        mech = "no joint rule (structural pick)"
    elif not info.get("stage2_ran"):
        mech = "stage-one fallback (shortlist of %d)" % (len(info["shortlist"].split(";")) if info["shortlist"] else 0)
    elif info["score"] > 0:
        mech = "stage two, positive score"
    else:
        mech = "stage two, all scores 0, tie broken by PenalizedObj"
    info = dict(info, n_floor=n_floor, mechanism=mech)
    return ranked, info


def proposed_info(menu, ho_col=HO5, include_legacy=False, floor=sm.ADEQUACY_FLOOR):
    ranked, info = sm.rule_balanced_maximin(menu, floor=floor, ho_col=ho_col, eligible_only=not include_legacy)
    info = dict(info, n_floor=len(ranked))
    if len(ranked):
        k = ranked[ranked.family == KNN]
        info["kNN_score"] = float(k["score"].iloc[0]) if len(k) else np.nan
        info["limiting"] = ranked.iloc[0]["limiting_criterion"]
    return ranked, info


def structural_info(menu, include_legacy):
    return sm.rule_structural_only(menu, eligible_only=not include_legacy)[1]


# =========================================================================================
# 1. Aggregation alternatives
# =========================================================================================
def analysis_aggregation(ctx, menu):
    v2 = sm.v2_inputs(menu, ctx.sup)
    v2c = pd.DataFrame(dict(family=v2["family"], PenalizedObj=v2[OBJ].astype(float),
                            CV_AUC=v2[AUC].astype(float), HO_within1=v2[HO13].astype(float)))
    sets = {"v2_7fam": v2c[v2c.family.isin(CV7)], "v2_9fam": v2c[v2c.PenalizedObj >= sm.ADEQUACY_FLOOR]}
    for hm, col in [("seed13", HO13), ("mean5", HO5)]:
        sets[f"A_matched_HO{hm}"] = crit_set(menu, col, include_legacy=False)
        sets[f"B_matched_HO{hm}"] = crit_set(menu, col, include_legacy=True)
    rows = []

    def rec(setname, rule, winner, detail="", score=np.nan):
        rows.append(dict(set=setname, rule=rule, winner=winner, detail=detail, score=score))

    for setname, fams in sets.items():
        fams = fams.reset_index(drop=True)
        names = list(fams.family)
        o = fams.PenalizedObj.to_numpy(float)
        u = fams.CV_AUC.to_numpy(float)
        no, nu = nrm(o), nrm(u)
        rho, p = spearman(o, u)
        rec(setname, "Spearman(obj, AUC)", "", f"rho {rho:+.4f} p {p:.4f}", rho)
        iv, ex = ws_intervals(names, no, nu)
        for lo, hi, f in iv:
            rec(setname, "2C weighted sum", f, f"w in [{lo:.4f}, {hi:.4f}]")
        for f, g, wx in ex:
            rec(setname, "2C weighted sum exact switch", f"{f}->{g}", f"w = {wx:.6f}", wx)
        rec(setname, "2C weighted sum w=0.5", names[int(np.argmax(0.5 * no + 0.5 * nu))])
        rs = ranks_best1(o) + ranks_best1(u)
        rec(setname, "2C rank-sum", "/".join(names[k] for k in range(len(names)) if rs[k] == rs.min()),
            f"min rank sum {rs.min():.1f}", float(rs.min()))
        mm = np.minimum(no, nu)
        kmm = int(np.argmax(mm))
        rec(setname, "2C maximin (stage one)", names[kmm], f"{mm[kmm]:.10f}", float(mm[kmm]))
        pf = pareto(names, np.c_[o, u])
        rec(setname, "2C Pareto front", ";".join(pf), "kNN on front: " + str(KNN in pf))
        e1 = [k for k in range(len(names)) if o[k] >= o.max() - V2_EPS_OBJ]
        e2 = [k for k in range(len(names)) if u[k] >= u.max() - V2_EPS_AUC]
        rec(setname, "2C eps1 (best AUC within 0.075 of best obj)", names[max(e1, key=lambda k: u[k])])
        rec(setname, "2C eps2 (best obj within 0.008 of best AUC)", names[max(e2, key=lambda k: o[k])])

        def sweep(cond, target, epss):
            out, prev = [], None
            for e in epss:
                adm = [k for k in range(len(names)) if cond[k] >= cond.max() - e]
                wn = names[max(adm, key=lambda k: target[k])]
                if wn != prev:
                    out.append([e, wn])
                    prev = wn
            return out
        s1 = sweep(o, u, np.round(np.arange(0, 0.30001, 1e-5), 6))
        s2 = sweep(u, o, np.round(np.arange(0, 0.06001, 1e-6), 7))
        rec(setname, "2C eps1 sweep", "", "; ".join(f"{e:.5f}: {wn}" for e, wn in s1))
        rec(setname, "2C eps2 sweep", "", "; ".join(f"{e:.6f}: {wn}" for e, wn in s2))
        if fams.HO_within1.notna().all():
            h = fams.HO_within1.to_numpy(float)
            nh = nrm(h)
            cnt = {}
            grid = [(a / 100, b / 100) for a in range(101) for b in range(101 - a)]
            for a, b in grid:
                k = int(np.argmax(a * no + b * nu + (1 - a - b) * nh))
                cnt[names[k]] = cnt.get(names[k], 0) + 1
            share = {k: v / len(grid) for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])}
            rec(setname, "3C weighted-sum simplex share", ";".join(f"{k} {v:.3f}" for k, v in share.items()),
                f"{len(grid)} points, step 0.01", share.get(KNN, 0.0))
            eq = (no + nu + nh) / 3
            rec(setname, "3C equal weights", names[int(np.argmax(eq))], f"{eq.max():.10f}", float(eq.max()))
            rs3 = ranks_best1(o) + ranks_best1(u) + ranks_best1(h)
            rec(setname, "3C rank-sum", "/".join(names[k] for k in range(len(names)) if rs3[k] == rs3.min()),
                f"{rs3.min():.1f}", float(rs3.min()))
            m3 = np.minimum(np.minimum(no, nu), nh)
            k3 = int(np.argmax(m3))
            rec(setname, "3C one-stage maximin", names[k3], f"{m3[k3]:.10f}", float(m3[k3]))
            pf3 = pareto(names, np.c_[o, u, h])
            rec(setname, "3C Pareto front", ";".join(pf3), "kNN on front: " + str(KNN in pf3))
        ts = sm.rule_v2_two_stage(fams.assign(eligible=True, SCS=np.nan, acceptance=np.nan),
                                  auc_col="CV_AUC", ho_col="HO_within1")[1]
        rec(setname, "published two-stage", ts["winner"], f"{ts['score']:.10f}", ts["score"])
    # the proposed rule itself (select_model.rule_balanced_maximin) on the matched menus
    for hm, col in [("seed13", HO13), ("mean5", HO5)]:
        for tag, inc in [("A", False), ("B", True)]:
            _, info = proposed_info(menu, col, include_legacy=inc)
            rec(f"{tag}_matched_HO{hm}", "proposed rule (select_model.rule_balanced_maximin)", info["winner"],
                f"{info['score']:.10f}; runner-up {info['runner_up']} {info['runner_up_score']:.10f}", info["score"])
    return pd.DataFrame(rows)


# =========================================================================================
# 2. Held-out variants (seed 13 alone, each seed, 5-seed mean; accepted vs all verbs)
# =========================================================================================
HO_VARIANTS = [("seed13", "HO_within1_seed13"), ("mean5", "HO_within1_mean5")] + \
              [(f"seed{s}", f"HO_within1_seed{s}") for s in (14, 15, 16, 17)] + \
              [("all_verbs_mean5", "HO_within1_all_mean5"), ("all_verbs_seed13", "HO_within1_all_seed13")]


def analysis_heldout(menu):
    rows = []
    for tag, inc in [("A", False), ("B", True)]:
        for hv, col in HO_VARIANTS:
            for rule in ("proposed", "two_stage"):
                if rule == "proposed":
                    _, info = proposed_info(menu, col, include_legacy=inc)
                else:
                    _, info = two_stage_info(menu, col, include_legacy=inc)
                rows.append(dict(menu=tag, ho=hv, ho_column=col, rule=rule, winner=info["winner"],
                                 score=info["score"], runner_up=info["runner_up"],
                                 runner_up_score=info["runner_up_score"], margin=info["margin"],
                                 shortlist=info.get("shortlist", ""), stage2_ran=info.get("stage2_ran", ""),
                                 kNN_score=info.get("kNN_score", np.nan)))
    return pd.DataFrame(rows)


# =========================================================================================
# 4. SCS weights
# =========================================================================================
def weight_settings(n_draws=N_DRAWS, seed=DRAW_SEED):
    settings = [("baseline", dict(BASE))]
    rng = np.random.default_rng(seed)
    for d in range(n_draws):
        settings.append((f"joint{d:03d}", {k: v * rng.uniform(0.5, 1.5) for k, v in BASE.items()}))
    for k in BASE:
        for f in SINGLE_FACTORS:
            w = dict(BASE)
            w[k] = BASE[k] * f
            settings.append((f"single_{k}_x{f}", w))
    return settings


def setting_record(ctx, menu, rec, floor=sm.ADEQUACY_FLOOR, cutoff=0.5):
    """Winners of the proposed, structural-only and two-stage rules for one setting."""
    for tag, inc in [("A", False), ("B", True)]:
        _, p = proposed_info(menu, HO5, include_legacy=inc, floor=floor)
        rec[f"proposed_{tag}_winner"] = p["winner"]
        rec[f"proposed_{tag}_score"] = p["score"]
        rec[f"proposed_{tag}_runner_up"] = p["runner_up"]
        rec[f"proposed_{tag}_margin"] = p["margin"]
        rec[f"proposed_{tag}_kNN_score"] = p.get("kNN_score", np.nan)
        rec[f"proposed_{tag}_limiting"] = p.get("limiting", "")
        _, p13 = proposed_info(menu, HO13, include_legacy=inc, floor=floor)
        rec[f"proposed_{tag}_HOseed13_winner"] = p13["winner"]
        rec[f"{tag}_n_floor"] = p["n_floor"]
        rec[f"{tag}_structural_winner"] = structural_info(menu, inc)["winner"]
        for hm, col in [("seed13", HO13), ("mean5", HO5)]:
            _, t = two_stage_info(menu, col, include_legacy=inc, floor=floor, cutoff=cutoff)
            rec[f"{tag}_{hm}_winner"] = t["winner"]
            rec[f"{tag}_{hm}_score"] = t["score"]
            rec[f"{tag}_{hm}_runner_up"] = t["runner_up"]
            rec[f"{tag}_{hm}_shortlist"] = t.get("shortlist", "")
            rec[f"{tag}_{hm}_stage2_ran"] = t.get("stage2_ran", False)
            rec[f"{tag}_{hm}_mechanism"] = t["mechanism"]
    return rec


def analysis_scs_weights(ctx, n_draws=N_DRAWS, seed=DRAW_SEED):
    rows = []
    for name, w in weight_settings(n_draws, seed):
        menu, bests, obj, scs = ctx.menu(w, ctx.T0)
        rec = dict(setting=name, **{k: w[k] for k in BASE})
        for r in menu.itertuples():
            rec[f"cfg::{r.family}"] = r.config
            rec[f"obj::{r.family}"] = r.PenalizedObj
        rows.append(setting_record(ctx, menu, rec))
    D = pd.DataFrame(rows)
    J = D[D.setting.str.startswith("joint")]
    S = D[D.setting.str.startswith("single")]
    reach = []
    for c in [c for c in D.columns if c.startswith("cfg::")]:
        for cfg, n in Counter(J[c]).most_common():
            reach.append(dict(family=c.split("::")[1], config=cfg, n_joint_draws=n, n_single=int((S[c] == cfg).sum())))
    return D, pd.DataFrame(reach)


# =========================================================================================
# 5. Guardrails and penalties
# =========================================================================================
def guardrail_settings():
    st = [("baseline", sm.compute_targets(), sm.ADEQUACY_FLOOR, 0.5)]
    for s in (0.0, 0.5, 2.0, 4.0):
        st.append((f"penalty x{s}", sm.compute_targets(pen_scale=s), sm.ADEQUACY_FLOOR, 0.5))
    for t in (0.80, 0.85, 0.90):
        st.append((f"acceptance target {t:.2f}", sm.compute_targets(acc_cap=t), sm.ADEQUACY_FLOOR, 0.5))
    for d in (-0.05, +0.05):
        st.append((f"e/a band shift {d:+.2f}", sm.compute_targets(ea_shift=d), sm.ADEQUACY_FLOOR, 0.5))
    for m in (0.04, 0.08, 0.12):
        st.append((f"band margin {m:.2f}", sm.compute_targets(margin=m), sm.ADEQUACY_FLOOR, 0.5))
    for f in (0.4, 0.5, 0.6):
        st.append((f"adequacy floor {f:.1f}", sm.compute_targets(), f, 0.5))
    for c in (0.4, 0.5, 0.6):
        st.append((f"stage-one cutoff {c:.1f}", sm.compute_targets(), sm.ADEQUACY_FLOOR, c))
    return st


def analysis_guardrails(ctx):
    rows, picks, base_cfg = [], [], None
    for name, T, floor, cut in guardrail_settings():
        menu, bests, obj, scs = ctx.menu(BASE, T)
        cfg = dict(zip(menu.family, menu.config))
        if base_cfg is None:
            base_cfg = dict(cfg)
        for r in menu.itertuples():
            picks.append(dict(setting=name, family=r.family, config=r.config, changed=(r.config != base_cfg[r.family]),
                              PenalizedObj=r.PenalizedObj, SCS=r.SCS, acceptance=r.acceptance,
                              e_over_a=r.e_over_a, E=r.E))
        rec = dict(setting=name, ACC_BAND=str(tuple(round(x, 6) for x in T["ACC_BAND"])),
                   ENAC_BAND=str(tuple(round(x, 6) for x in T["ENAC_BAND"])), PEN=f"{T['PEN_ACC']}/{T['PEN_ENAC']}",
                   floor=floor, cutoff=cut,
                   changed_family_picks=";".join(f"{f}->{c}" for f, c in cfg.items() if c != base_cfg[f]))
        rows.append(setting_record(ctx, menu, rec, floor=floor, cutoff=cut))
    return pd.DataFrame(rows), pd.DataFrame(picks)


# =========================================================================================
# 6. Decision margins of the proposed rule
# =========================================================================================
def _proposed_fast(names, obj, auc, ho, floor=sm.ADEQUACY_FLOOR):
    idx = np.where((obj >= floor) & ~np.isnan(auc) & ~np.isnan(ho))[0]
    if len(idx) == 0:
        return None
    sc = np.minimum(np.minimum(nrm(obj[idx]), nrm(auc[idx])), nrm(ho[idx]))
    return names[idx[np.lexsort((-obj[idx], -sc))[0]]]


def analysis_decision_margins(menu, ho_col=HO5, include_legacy=False):
    m = menu if include_legacy else menu[menu["eligible"]]
    m = m[m[OBJ] >= sm.ADEQUACY_FLOOR].reset_index(drop=True)
    names = list(m.family)
    base = {c: m[c].to_numpy(float) for c in (OBJ, AUC, ho_col)}
    w0 = _proposed_fast(names, base[OBJ], base[AUC], base[ho_col])
    ranked, info = proposed_info(menu, ho_col, include_legacy)
    assert w0 == info["winner"], (w0, info["winner"])
    rows = []
    for k, fam in enumerate(names):
        for col, span in [(AUC, 0.05), (ho_col, 0.05), (OBJ, 0.2)]:
            def win(delta):
                v = {c: a.copy() for c, a in base.items()}
                v[col][k] += delta
                return _proposed_fast(names, v[OBJ], v[AUC], v[ho_col])
            for sign in (+1, -1):
                found, prev = None, 0.0
                for d in np.arange(1e-4, span + 1e-4, 1e-4):
                    if win(sign * d) != w0:
                        lo, hi = prev, d
                        while hi - lo > 1e-7:
                            mid = (lo + hi) / 2
                            if win(sign * mid) != w0:
                                hi = mid
                            else:
                                lo = mid
                        found = (sign * hi, win(sign * hi))
                        break
                    prev = d
                rows.append(dict(ho_column=ho_col, base_winner=w0, family=fam, criterion=col,
                                 direction=("+" if sign > 0 else "-"), value=float(base[col][k]),
                                 flip_delta=(np.nan if found is None else found[0]),
                                 new_winner=(None if found is None else found[1]), searched_span=span))
    return pd.DataFrame(rows)


# =========================================================================================
# Report
# =========================================================================================
def _counts(series):
    s = series.astype(object).where(series.notna(), "none (no family passes the floor)")
    return ", ".join(f"{k} {v}" for k, v in Counter(s).most_common())


def fmt(v, nd=6):
    return sm.fmt(v, nd)


def md_table(df, cols, nd=6):
    return sm.md_table(df, cols, nd)


def build_report(ctx, menu, agg, hov, legdf, D, reach, G, P, margins):
    L = []
    w = L.append
    J = D[D.setting.str.startswith("joint")] if D is not None else None
    S = D[D.setting.str.startswith("single")] if D is not None else None
    _, pA = proposed_info(menu, HO5, False)
    _, pB = proposed_info(menu, HO5, True)
    _, tA = two_stage_info(menu, HO13, False)
    _, tB = two_stage_info(menu, HO13, True)
    w("# Robustness of the round-3 model selection (generated by robustness.py; do not edit by hand)")
    w("")
    w(f"Selection rule under test: `{sm.SELECTION_RULE}` (select_model.py; the selection rule of the article, "
      "Section 2.3.5). Inputs: results/structural_scan_components.csv, results/selection/supervised_grid_per_seed.csv. "
      "All supervised numbers are matched: each family is evaluated at exactly the configuration and percentile "
      "that is its structural best under the setting being tested.")
    w("")
    w("## Headlines")
    w("")
    H = []
    H.append(f"Baseline, proposed rule (menu A = the 11 one-vs-rest families, 9 pass the floor): **{pA['winner']}** "
             f"{pA['score']:.6f}; runner-up {pA['runner_up']} {pA['runner_up_score']:.6f}; margin {pA['margin']:.6f}.")
    rA, _ = two_stage_info(menu, HO13, False)
    kr = rA[rA.family == KNN].iloc[0]
    sh = rA[rA["family"].isin(tA["shortlist"].split(";"))]
    H.append(f"Round-2 two-stage rule on the same matched inputs (held-out seed 13): **{tA['winner']}** "
             f"{tA['score']:.6f} ({tA['mechanism']}); shortlist {tA['shortlist'].replace(';', ', ')}. "
             f"{KNN} scores {kr['score']:.6f} there: re-normalized inside the shortlist its CV AUC "
             f"({kr[AUC]:.6f}) gets s_auc {kr['s_auc']:.6f} (shortlist CV AUC range {sh[AUC].min():.6f} to "
             f"{sh[AUC].max():.6f}).")
    if agg is not None:
        a5 = agg[agg.set == "A_matched_HOmean5"].set_index("rule")
        a13 = agg[agg.set == "A_matched_HOseed13"].set_index("rule")
        H.append("Other aggregations, menu A, held-out 5-seed mean: equal weights " +
                 f"{a5.loc['3C equal weights', 'winner']}; three-criterion rank-sum {a5.loc['3C rank-sum', 'winner']} "
                 f"(sum {a5.loc['3C rank-sum', 'detail']}); stage-one maximin {a5.loc['2C maximin (stage one)', 'winner']}; "
                 f"weighted-sum simplex share won by kNN-21 {a5.loc['3C weighted-sum simplex share', 'score']:.3f} "
                 f"({a5.loc['3C weighted-sum simplex share', 'winner']}); "
                 f"Pareto front {a5.loc['3C Pareto front', 'winner'].replace(';', ', ')}.")
        H.append("Two-criterion rules (structure and CV AUC only), menu A: rank-sum "
                 f"{a13.loc['2C rank-sum', 'winner'].replace('/', ' and ')} tied; epsilon rule at 0.008 AUC "
                 f"{a13.loc['2C eps2 (best obj within 0.008 of best AUC)', 'winner']}; epsilon rule at 0.075 objective "
                 f"{a13.loc['2C eps1 (best AUC within 0.075 of best obj)', 'winner']}; weighted sum: "
                 + "; ".join(f"{r.winner} for {r.detail}" for r in agg[(agg.set == 'A_matched_HOseed13') &
                                                                       (agg.rule == '2C weighted sum')].itertuples()) + ".")
    if hov is not None:
        hp = hov[(hov.menu == "A") & (hov.rule == "proposed")]
        H.append("Held-out variants, proposed rule, menu A: " +
                 "; ".join(f"{r.ho} {r.winner} {r.score:.6f}" for r in hp.itertuples()) + ".")
        ht = hov[(hov.menu == "A") & (hov.rule == "two_stage")]
        H.append("Held-out variants, two-stage rule, menu A: " +
                 "; ".join(f"{r.ho} {r.winner} {r.score:.6f}" for r in ht.itertuples()) + ".")
    lr = menu[menu.family == LEGACY].iloc[0]
    H.append(f"Legacy kNN admitted (evaluation-only construction, `{lr['config']}`, PenalizedObj {lr[OBJ]:.6f}, "
             f"CV AUC {lr[AUC]:.6f}, held-out within-one mean {lr[HO5]:.6f}): proposed rule **{pB['winner']}** "
             f"{pB['score']:.6f} (runner-up {pB['runner_up']} {pB['runner_up_score']:.6f}); two-stage rule "
             f"(seed 13) {tB['winner']} {tB['score']:.6f}. Legacy kNN is not eligible because it cannot run the "
             "production pipeline (one global vote cutoff, no per-level calibrated scores, not exportable).")
    if D is not None:
        H.append(f"SCS weights, {len(J)} joint draws, proposed rule menu A: {_counts(J['proposed_A_winner'])}. "
                 f"Structural-only rule menu A: {_counts(J['A_structural_winner'])}; menu B: "
                 f"{_counts(J['B_structural_winner'])}. Round-2 two-stage rule (matched, seed 13) menu A: "
                 f"{_counts(J['A_seed13_winner'])}.")
        H.append(f"SCS weights, {len(S)} single-weight settings, proposed rule menu A: {_counts(S['proposed_A_winner'])}; "
                 f"structural-only menu A: {_counts(S['A_structural_winner'])}.")
        full = J[J.A_n_floor == J.A_n_floor.max()]
        H.append(f"Of the {len(J)} joint draws, {len(full)} keep all {int(J.A_n_floor.max())} families above the absolute "
                 f"floor; there the proposed rule picks {_counts(full['proposed_A_winner'])}. The remaining draws "
                 "(fewer floor-passing families, because the floor does not rescale with the weights): " +
                 "; ".join(f"{int(k)} families: {_counts(g['proposed_A_winner'])}"
                           for k, g in J[J.A_n_floor < J.A_n_floor.max()].groupby("A_n_floor")) + ".")
        H.append(f"Joint draws with Legacy kNN admitted, proposed rule: {_counts(J['proposed_B_winner'])}.")
    if G is not None:
        gw = G[["setting", "proposed_A_winner"]]
        nk = int((gw.proposed_A_winner == KNN).sum())
        H.append(f"Guardrails and penalties ({len(G)} settings, one change at a time), proposed rule menu A: kNN-21 in "
                 f"{nk} of {len(G)}; other winners: " +
                 (", ".join(f"{r.setting} -> {r.proposed_A_winner}" for r in gw[gw.proposed_A_winner != KNN].itertuples())
                  or "none") + ".")
    if margins is not None:
        mm = margins.dropna(subset=["flip_delta"]).copy()
        mm["abs"] = mm.flip_delta.abs()
        mm = mm.sort_values("abs").head(3)
        H.append("Smallest single changes that would change the proposed rule's winner (menu A, 5-seed mean): " +
                 "; ".join(f"{r.family} {r.criterion} {r.flip_delta:+.6f} -> {r.new_winner}" for r in mm.itertuples()) + ".")
    for i, h in enumerate(H, 1):
        w(f"{i}. {h}")
    w("")
    w("## Baseline menu (structural best configuration of each family, matched supervised numbers)")
    w("")
    w(md_table(menu, ["family", "eligible", "config", "PenalizedObj", "CV_macro_AUC_mean5", "HO_within1_mean5",
                      "HO_within1_seed13", "HO_within1_all_mean5"]))
    w("")
    if agg is not None:
        w("## 1. Aggregation alternatives (robustness_aggregation.csv)")
        w("")
        w("Sets: v2_7fam = round-2 inputs on the 7-family menu of the round-2 version of paper_results.py "
          "(SGDClassifier, LogisticRegression, RidgeClassifier, LinearSVC, KNeighborsClassifier, NearestCentroid, "
          "PassiveAggressive); v2_9fam = round-2 inputs on the 9 floor-passing families of round-2 Table 8a; "
          "A = matched inputs, 9 floor-passing one-vs-rest families; B = A plus "
          "Legacy kNN (evaluation-only supervised numbers). 2C = structure and CV AUC; 3C = plus held-out within-one. "
          "Short names: kNN-21 = KNeighborsClassifier (n=21, cosine, distance-weighted, P50), CalSVC = CalibratedSVC, "
          "LogReg = LogisticRegression, NCentroid = NearestCentroid, PA = PassiveAggressive, a slash marks a tie.")
        w("")
        keep = agg[~agg.rule.str.contains("sweep|Spearman|simplex|weighted sum$|exact switch")]
        piv = keep.drop_duplicates(["set", "rule"]).pivot(index="rule", columns="set", values="winner").fillna("")
        piv = piv.map(lambda s: ";".join("/".join(SHORT.get(y, y) for y in x.split("/")) for x in str(s).split(";"))
                      if s else "")
        w(md_table(piv.reset_index(), ["rule"] + list(piv.columns)))
        w("")
        for st in agg.set.unique():
            sp = agg[(agg.set == st) & (agg.rule == "Spearman(obj, AUC)")].detail.iloc[0]
            ws = agg[(agg.set == st) & (agg.rule == "2C weighted sum")]
            w(f"- {st}: Spearman(PenalizedObj, CV AUC) {sp}; two-criterion weighted sum (structure weight w): " +
              "; ".join(f"{SHORT.get(r.winner, r.winner)} {r.detail}" for r in ws.itertuples()))
        w("")
    if hov is not None:
        w("## 2. Held-out variants (robustness_heldout_variants.csv)")
        w("")
        w(md_table(hov, ["menu", "ho", "rule", "winner", "score", "runner_up", "runner_up_score", "kNN_score"]))
        w("")
    if legdf is not None and len(legdf):
        w("## 3. Legacy kNN, evaluation-only supervised numbers (robustness_legacy_knn_supervised.csv)")
        w("")
        g = legdf.groupby("cfg_id")[["CV_macro_AUC", "CV_macro_F1", "HO_gate", "HO_within1", "HO_within1_all"]].mean()
        w(md_table(g.reset_index(), ["cfg_id", "CV_macro_AUC", "CV_macro_F1", "HO_gate", "HO_within1", "HO_within1_all"]))
        w("")
        w("Means over the five repeats / outer seeds. Construction: votes against the outer-training verbs; vote sums "
          "calibrated per level by an empirical CDF fitted on inner out-of-fold vote sums; hard labels from the vote "
          "rule itself (no per-level thresholds, no soft tier). Production cannot use this model.")
        w("")
    if D is not None:
        w("## 4. SCS weights (robustness_scs_weights.csv, robustness_scs_config_reach.csv)")
        w("")
        w(f"{len(J)} joint draws (numpy default_rng({DRAW_SEED}); each weight times U(0.5, 1.5) in the order "
          "wD, wM, wT, wC, wE) and 20 single-weight settings. Per setting: objective recomputed, each family's best "
          "configuration re-selected, matched supervised numbers pulled, rules re-applied. The adequacy floor 0.5 is "
          "absolute, on a scale that moves with the weights.")
        w("")
        for col, lab in [("proposed_A_winner", "proposed rule, menu A (5-seed mean)"),
                         ("proposed_A_HOseed13_winner", "proposed rule, menu A (held-out seed 13)"),
                         ("proposed_B_winner", "proposed rule, Legacy kNN admitted"),
                         ("A_structural_winner", "structural-only rule, menu A"),
                         ("B_structural_winner", "structural-only rule, with Legacy kNN"),
                         ("A_seed13_winner", "round-2 two-stage rule, menu A, seed 13"),
                         ("A_mean5_winner", "round-2 two-stage rule, menu A, 5-seed mean"),
                         ("B_seed13_winner", "round-2 two-stage rule, with Legacy kNN, seed 13")]:
            w(f"- {lab}: joint {_counts(J[col])}; single {_counts(S[col])}")
        w(f"- floor-passing one-vs-rest families per joint draw: " +
          ", ".join(f"{k}: {v}" for k, v in sorted(Counter(J['A_n_floor']).items())))
        kw = J[J.proposed_A_winner == KNN]
        if len(kw):
            w(f"- proposed rule, menu A, draws won by kNN-21: winning margin median {kw.proposed_A_margin.median():.6f}, "
              f"minimum {kw.proposed_A_margin.min():.6f}")
        w("")
        w(md_table(S, ["setting", "proposed_A_winner", "proposed_A_score", "A_structural_winner", "A_seed13_winner",
                       "proposed_B_winner"]))
        w("")
        w("Configurations reached as a family's best (joint draws / single-weight settings):")
        w("")
        w(md_table(reach, ["family", "config", "n_joint_draws", "n_single"]))
        w("")
    if G is not None:
        w("## 5. Guardrails and penalties (robustness_guardrails.csv, robustness_guardrail_family_picks.csv)")
        w("")
        w(md_table(G, ["setting", "ACC_BAND", "ENAC_BAND", "PEN", "floor", "cutoff", "A_n_floor", "proposed_A_winner",
                       "proposed_A_score", "A_structural_winner", "A_seed13_winner", "A_seed13_score",
                       "proposed_B_winner"]))
        w("")
        w("Changed family picks per setting: " + "; ".join(f"{r.setting}: {r.changed_family_picks or 'none'}"
                                                          for r in G.itertuples()))
        w("")
    if margins is not None:
        w("## 6. Decision margins of the proposed rule (robustness_decision_margins.csv)")
        w("")
        w("Smallest additive change to one family's value (others fixed) that changes the winner; searched within "
          "+-0.05 for CV AUC and held-out within-one, +-0.2 for PenalizedObj; empty = no change within the span.")
        w("")
        mm = margins.copy()
        mm["abs"] = mm.flip_delta.abs()
        w(md_table(mm.sort_values("abs").dropna(subset=["flip_delta"]).head(12),
                   ["family", "criterion", "direction", "value", "flip_delta", "new_winner"]))
        w("")
    txt = "\n".join(L)
    for bad in (chr(0x2013), chr(0x2014)):
        assert bad not in txt
    return txt


# =========================================================================================
# main
# =========================================================================================
def run(only=("all",), scan_path=sm.SCAN_CSV, sel_dir=sm.SEL_DIR, write=True, n_draws=N_DRAWS):
    ctx = Ctx(scan_path, sel_dir)
    menu, _, _, _ = ctx.menu(BASE, ctx.T0)
    every = "all" in only
    agg = analysis_aggregation(ctx, menu) if every or "aggregation" in only else None
    hov = analysis_heldout(menu) if every or "heldout" in only else None
    D, reach = analysis_scs_weights(ctx, n_draws) if every or "weights" in only else (None, None)
    G, P = analysis_guardrails(ctx) if every or "guardrails" in only else (None, None)
    margins = analysis_decision_margins(menu) if every or "margins" in only else None
    legdf = ctx.leg.frame()
    out = dict(menu=menu, aggregation=agg, heldout_variants=hov, legacy_knn_supervised=legdf, scs_weights=D,
               scs_config_reach=reach, guardrails=G, guardrail_family_picks=P, decision_margins=margins)
    if write:
        os.makedirs(sel_dir, exist_ok=True)
        for k, v in out.items():
            if k != "menu" and v is not None and len(v):
                v.to_csv(os.path.join(sel_dir, f"robustness_{k}.csv"), index=False)
        if every:
            txt = build_report(ctx, menu, agg, hov, legdf, D, reach, G, P, margins)
            with open(os.path.join(sel_dir, "robustness_report.md"), "w", encoding="utf-8", newline="\n") as f:
                f.write(txt + "\n")
    return out, ctx


def main():
    ap = argparse.ArgumentParser(description="Robustness of the round-3 model selection")
    ap.add_argument("--scan", default=sm.SCAN_CSV)
    ap.add_argument("--sel-dir", default=sm.SEL_DIR)
    ap.add_argument("--only", nargs="+", default=["all"],
                    choices=["all", "aggregation", "heldout", "weights", "guardrails", "margins"])
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    out, ctx = run(tuple(a.only), a.scan, a.sel_dir, not a.no_write, a.draws)
    m = out["menu"]
    _, p = proposed_info(m, HO5, False)
    print(f"[proposed rule, menu A] {p['winner']} {p['score']!r}; runner-up {p['runner_up']} {p['runner_up_score']!r}")
    if out["scs_weights"] is not None:
        J = out["scs_weights"][out["scs_weights"].setting.str.startswith("joint")]
        print("[SCS joint draws] proposed A:", dict(Counter(J.proposed_A_winner)),
              "| structural A:", dict(Counter(J.A_structural_winner)))
    if out["guardrails"] is not None:
        print("[guardrails] proposed A:", dict(Counter(out["guardrails"].proposed_A_winner)))


if __name__ == "__main__":
    main()
