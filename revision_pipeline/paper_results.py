"""Every number and table the manuscript prints, generated from the committed outputs.

This script runs no model. It reads the files that the reproduction sequence writes (see the
README), checks them against each other, and writes two files:

  results/paper_tables.md     every manuscript table, each with the file(s) it was read from
  results/paper_numbers.json  one flat key per number, with the exact value of the source file

Tables are named by content. The "v2" numbers in the headings refer to the round-2 manuscript and
are given only for orientation; the round-3 numbering follows the restructured manuscript.

Inputs (all committed; nothing else is read):
  revision_pipeline/core_merged.csv                      merge_core.py
  results/core_structure/T1_core_all.csv, T2_core_high.csv, T3_core_low.csv   BT 1 notebook
  results/core_structure/core_gap_means.csv              regen_core_figures.py
  results/model_comparison_table5.csv, model_comparison_pareto.csv, winner_and_constants.json,
  results/Verb List Classified by Model.csv              retrain.py
  results/selection/matched_supervised.csv, baselines.csv, production_per_level.csv,
      mcnemar_vs_production.csv, encoder_ablation.csv     supervised_eval.py
  results/selection/selection_table.csv, v2_default_config_reference.csv   select_model.py
  results/selection/robustness_*.csv                      robustness.py
  results/structural_scan_components.csv                 retrain.py (scan size, keys only)
  results/selection/oof_decisions.npz, grid_meta.json, encoder_ablation_meta.json
                                                          supervised_eval.py (split sizes and
                                                          software versions, keys only)
  results/bank_structure/*.csv                            bank_structure.py
  results/expert/*.csv, expert_summary.json               expert_recompute.py
  results/change_attribution/*.csv                        change_attribution.py (response letter:
                                                          bank change and refit attribution)
  revision_pipeline/cv_results_{mpnet,minilm,droberta}.csv, heldout_results.csv
                                                          cv_metrics.py, methodology_upgrades.py
                                                          (round-2 default configurations, kept for
                                                          traceability)

The checks stop the script with an error when two sources disagree (for example, the selected
family in selection_table.csv against winner_and_constants.json, or the bank counts against
coverage_cardinality.csv), so a stale output cannot reach the tables silently.

Output is deterministic (no timestamps; LF line endings), so a rerun on the same inputs is
byte-identical.

Run from anywhere: python -I -B revision_pipeline/paper_results.py
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LEVELS = ["Kn", "Cm", "Ap", "An", "Sn", "Ev"]
PROD = "KNeighborsClassifier"
OUT_MD = os.path.join(REPO, "results", "paper_tables.md")
OUT_JSON = os.path.join(REPO, "results", "paper_numbers.json")

NUM = {}   # flat key -> exact value
MD = []    # lines of paper_tables.md


# ----------------------------------------------------------------------------- helpers
def path(rel):
    """Repository-relative path written with '/' -> absolute path for this OS."""
    return os.path.join(REPO, *rel.split("/"))


def csv(rel, **kw):
    return pd.read_csv(path(rel), float_precision="round_trip", **kw)


def load_json(rel):
    with open(path(rel), encoding="utf-8") as f:
        return json.load(f)


def clean(v):
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def put(key, value):
    if key in NUM:
        raise SystemExit(f"paper_results.py: duplicate key {key}")
    NUM[key] = clean(value)
    return value


def check(cond, msg):
    if not cond:
        raise SystemExit(f"paper_results.py: CHECK FAILED: {msg}")


def close(a, b, tol, msg):
    check(abs(float(a) - float(b)) <= tol, f"{msg}: {a!r} vs {b!r} (tolerance {tol})")


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def f6(x):
    return f"{x:.6f}"


def fp(p):
    return f"{p:.1e}" if p < 1e-4 else f"{p:.4f}"


def frac(k, n):
    return f"{k}/{n} = {100.0 * k / n:.1f}%"


def w(line=""):
    MD.append(line)


def table(header, rows):
    w("| " + " | ".join(header) + " |")
    w("|" + "---|" * len(header))
    for r in rows:
        w("| " + " | ".join(str(c) for c in r) + " |")
    w()


def heading(title, sources):
    w(f"## {title}")
    w()
    w("Source: " + "; ".join(f"`{s}`" for s in sources) + ".")
    w()


def cfg_text(cfg_id, percentile=None):
    """'KNeighborsClassifier|{"n_neighbors": 21}', 50 -> 'n_neighbors=21, P50'."""
    parts = cfg_id.split("|", 1)
    params = ""
    if len(parts) == 2 and parts[1].startswith("{"):
        params = ", ".join(f"{k}={v}" for k, v in json.loads(parts[1]).items())
    elif len(parts) == 2:
        params = parts[1].replace("|", ", ")
    if percentile is not None and int(percentile) >= 0:
        params = (params + ", " if params else "") + f"P{int(percentile)}"
    return params or "defaults"


def short(fam):
    return {"KNeighborsClassifier": "kNN", "LogisticRegression": "LogisticRegression",
            "RidgeClassifier": "Ridge", "SGDClassifier": "SGD",
            "PassiveAggressive": "PassiveAggressive"}.get(fam, fam)


def slug(text):
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


# ----------------------------------------------------------------------------- inputs
sel = csv("results/selection/selection_table.csv")
ms = csv("results/selection/matched_supervised.csv")
ms_main = ms[~ms["supplementary"].astype(bool)].set_index("family")
base = csv("results/selection/baselines.csv")
wjs = load_json("results/winner_and_constants.json")
BOOT = str(base["bootstrap_method"].iloc[0])
check(base["bootstrap_method"].nunique() == 1, "one bootstrap method in baselines.csv")

sel_by_fam = sel.set_index("family")
winner_row = sel[sel["selected"].astype(bool)]
check(len(winner_row) == 1, "exactly one selected family in selection_table.csv")
winner_row = winner_row.iloc[0]
check(winner_row["family"] == PROD, f"selection_table.csv selects {winner_row['family']}, expected {PROD}")
check(wjs["selection"]["winner_family"] == PROD, "winner_and_constants.json selects the production family")
close(wjs["selection"]["score"], winner_row["score"], 1e-12, "winner score, json vs selection_table")
check(wjs["selection"]["winner_percentile"] == int(winner_row["percentile"]), "winner percentile")
check(wjs["selection"]["winner_cfg_id"] == winner_row["cfg_id"], "winner configuration")
eligible_floor = sel[sel["eligible"].astype(bool) & sel["floor_pass"].astype(bool)]["family"].tolist()
n_floor = len(eligible_floor)

w("# Manuscript tables and numbers (generated; do not edit by hand)")
w()
w("Generated by `revision_pipeline/paper_results.py` from committed outputs only. Every exact value "
  "is in `results/paper_numbers.json`; the tables below round for printing. Families are evaluated "
  "at matched configurations: each family's own structural best configuration and threshold "
  "percentile, for every criterion. The production model is kNN with n_neighbors = 21, cosine "
  "distance, distance weighting, thresholds at P50 (`KNeighborsClassifier` below).")
w()
put("meta.generator", "revision_pipeline/paper_results.py")
put("meta.bootstrap_method", BOOT)
put("meta.production_family", PROD)

# ----------------------------------------------------------------------------- 1. core structure
core = csv("revision_pipeline/core_merged.csv")
n_core_verbs = core["verb"].nunique()
n_core_assign = len(core)
n_high = int((core["freq"] >= 10).sum())
n_low = int((core["freq"] < 10).sum())
check((n_core_verbs, n_core_assign) == (381, 607), f"core size {n_core_verbs}/{n_core_assign}")
put("core.n_verbs", n_core_verbs)
put("core.n_assignments", n_core_assign)
put("core.n_assignments_high_freq_ge10", n_high)
put("core.n_assignments_low_freq_lt10", n_low)

heading("1. Core-list centroid distances (v2 Tables 2 to 4) and gap means (v2 Fig. 2)",
        ["results/core_structure/T1_core_all.csv", "results/core_structure/T2_core_high.csv",
         "results/core_structure/T3_core_low.csv",
         "results/core_structure/core_gap_means.csv", "revision_pipeline/core_merged.csv"])
w(f"Core: {n_core_verbs} verbs, {n_core_assign} verb-level assignments ({n_high} with source "
  f"frequency >= 10, {n_low} below 10). Distance = 1 - cosine similarity of level centroids "
  "(all-mpnet-base-v2).")
w()
gm = csv("results/core_structure/core_gap_means.csv")
for tag, rel, label, gm_row in (("all", "results/core_structure/T1_core_all.csv", "All core", 0),
                                ("high", "results/core_structure/T2_core_high.csv", "High-frequency core", 1),
                                ("low", "results/core_structure/T3_core_low.csv", "Low-frequency core", 2)):
    M = csv(rel, index_col=0)
    check(list(M.index) == LEVELS and list(M.columns) == LEVELS, f"{rel} level order")
    w(f"**{label}** (`{rel.split('/')[-1]}`)")
    w()
    rows = []
    for a in LEVELS:
        cells = []
        for b in LEVELS:
            v = float(M.loc[a, b])
            v = 0.0 if a == b else v
            cells.append(f3(v))
            if a != b:
                put(f"core.matrix.{tag}.{a}_{b}", float(M.loc[a, b]))
        rows.append([a] + cells)
    table(["", *LEVELS], rows)
    # gap means recomputed from the table must equal the figure file (regen_core_figures.py
    # computes from the embeddings; the tables carry float32 values from BT 1)
    for n in range(1, 6):
        d = sum(float(M.loc[LEVELS[i], LEVELS[i + n]]) for i in range(6 - n)) / (6 - n)
        close(d, gm.iloc[gm_row][f"d{n}"], 5e-4, f"core gap mean {tag} d{n}, table vs core_gap_means.csv")
rows = []
for i, tag in enumerate(("all", "high", "low")):
    r = gm.iloc[i]
    rows.append([r["subset"]] + [f4(r[f"d{n}"]) for n in range(1, 6)])
    for n in range(1, 6):
        put(f"core.gap_mean.{tag}.d{n}", r[f"d{n}"])
w("Mean centroid distance d_n by taxonomic gap n (Fig. 2):")
w()
table(["Subset", "d1", "d2", "d3", "d4", "d5"], rows)

# ----------------------------------------------------------------------------- 2. structural comparison
t5 = csv("results/model_comparison_table5.csv")
pareto = csv("results/model_comparison_pareto.csv")
heading("2. Structural comparison of the families (structural part of v2 Table 8a; v1 Table 5)",
        ["results/model_comparison_table5.csv", "results/model_comparison_pareto.csv",
         "results/selection/selection_table.csv"])
w("Each family at its structural best (largest penalized objective, Eq. 19; ties by SCS, then "
  "acceptance), on the 1,650 unlabeled candidates. Floor: PenalizedObj >= 0.5. Legacy kNN is a "
  "structural reference row: it is not eligible for selection (one global vote cutoff, no "
  "per-level calibrated scores, cannot run the production pipeline or be exported).")
w()
rows = []
on_front = set(pareto["Model"])
for _, r in t5.iterrows():
    fam = r["Model"].split(" (")[0]
    eligible = fam != "Legacy kNN"
    key = slug(fam)
    rows.append([r["Model"], f3(r["Eval_SCS"]), f3(r["Eval_Acceptance"]), f3(r["Eval_en_over_ac"]),
                 f3(r["PenalizedObj"]), "yes" if r["PenalizedObj"] >= 0.5 else "no",
                 "yes" if eligible else "no (reference)", "yes" if r["Model"] in on_front else ""])
    put(f"structural.{key}.label", r["Model"])
    put(f"structural.{key}.Eval_SCS", r["Eval_SCS"])
    put(f"structural.{key}.Eval_Acceptance", r["Eval_Acceptance"])
    put(f"structural.{key}.Eval_en_over_ac", r["Eval_en_over_ac"])
    put(f"structural.{key}.PenalizedObj", r["PenalizedObj"])
    if fam in sel_by_fam.index:
        close(r["PenalizedObj"], sel_by_fam.loc[fam, "PenalizedObj"], 1e-9,
              f"PenalizedObj {fam}, table5 vs selection_table")
table(["Family (configuration)", "SCS", "Acceptance", "Entries per accepted verb (pre-collapse)",
       "PenalizedObj", "Floor", "Eligible", "Pareto front (SCS, acceptance)"], rows)
best_eligible = sel[sel["eligible"].astype(bool)].sort_values("PenalizedObj", ascending=False).iloc[0]
put("structural.best_eligible_family", best_eligible["family"])
put("structural.pareto_front", ";".join(pareto["Model"]))
w(f"Highest PenalizedObj among the eligible families: {best_eligible['family']} "
  f"({cfg_text(best_eligible['cfg_id'], best_eligible['percentile'])}, {f3(best_eligible['PenalizedObj'])}). "
  f"The Pareto front of SCS and acceptance is computed over all rows, the Legacy kNN reference row "
  f"included: {', '.join(pareto['Model'])}.")
w()

# ----------------------------------------------------------------------------- 3. CV at matched configs
heading("3. Repeated stratified cross-validation at matched configurations (replaces v2 Table 6)",
        ["results/selection/matched_supervised.csv", "results/selection/baselines.csv"])
w("Repeated stratified 5-fold multi-label cross-validation on the 381-verb core, 5 repeats "
  "(MultilabelStratifiedKFold, seeds 13 to 17); scores calibrated by the out-of-fold empirical CDF "
  "(inner 5-fold KFold, seed 13) and thresholded at each family's own percentile. Mean (SD) over "
  f"the 5 repeats. Interval: {BOOT}.")
w()
rows = []
order = ms_main.sort_values("CV_macro_AUC_mean5", ascending=False).index.tolist()
for fam in order:
    r = ms_main.loc[fam]
    k = slug(fam)
    for col in ("CV_macro_AUC_mean5", "CV_macro_AUC_sd5", "CV_macro_AUC_avg5_ci95_pctboot_lo",
                "CV_macro_AUC_avg5_ci95_pctboot_hi", "CV_macro_F1_mean5", "CV_macro_F1_sd5",
                "CV_macro_F1_avg5_ci95_pctboot_lo", "CV_macro_F1_avg5_ci95_pctboot_hi",
                "CV_micro_F1_mean5", "CV_micro_F1_sd5"):
        put(f"cv.{k}.{col}", r[col])
    put(f"cv.{k}.config", cfg_text(r["cfg_id"], r["percentile"]))
    if fam in sel_by_fam.index:
        close(r["CV_macro_AUC_mean5"], sel_by_fam.loc[fam, "CV_macro_AUC_mean5"], 1e-12,
              f"CV AUC {fam}, matched_supervised vs selection_table")
    name = fam + (" (production)" if fam == PROD else "") + ("" if r["floor_pass"] else " (below floor)")
    rows.append([name, cfg_text(r["cfg_id"], r["percentile"]),
                 f"{f3(r['CV_macro_AUC_mean5'])} ({f3(r['CV_macro_AUC_sd5'])}) "
                 f"[{f3(r['CV_macro_AUC_avg5_ci95_pctboot_lo'])}, {f3(r['CV_macro_AUC_avg5_ci95_pctboot_hi'])}]",
                 f"{f3(r['CV_macro_F1_mean5'])} ({f3(r['CV_macro_F1_sd5'])}) "
                 f"[{f3(r['CV_macro_F1_avg5_ci95_pctboot_lo'])}, {f3(r['CV_macro_F1_avg5_ci95_pctboot_hi'])}]",
                 f"{f3(r['CV_micro_F1_mean5'])} ({f3(r['CV_micro_F1_sd5'])})"])
maj = base[base["baseline"] == "majority-level"].iloc[0]
frm = base[(base["baseline"] == "freq-random") & (base["seed"].astype(str) == "mean5")].iloc[0]
rows.append(["Baseline: majority level (Ap) for every verb", "", "0.500 (by construction)",
             f"{f3(maj['macro_F1'])} [{f3(maj['macro_F1_ci95_pctboot_lo'])}, {f3(maj['macro_F1_ci95_pctboot_hi'])}]",
             f3(maj["micro_F1"])])
rows.append(["Baseline: prevalence-proportional random labels (mean of 5 seeds)", "", "0.500 (by construction)",
             f"{f3(frm['macro_F1'])} [{f3(frm['macro_F1_ci95_pctboot_lo'])}, {f3(frm['macro_F1_ci95_pctboot_hi'])}]",
             f3(frm["micro_F1"])])
for tag, r in (("majority_level", maj), ("freq_random_mean5", frm)):
    for col in ("macro_F1", "micro_F1", "macro_F1_ci95_pctboot_lo", "macro_F1_ci95_pctboot_hi", "macro_AUC",
                "HO_gate", "HO_prim", "HO_within1", "HO_within1_all", "HO_mdist",
                "HO_within1_ci95_pctboot_lo", "HO_within1_ci95_pctboot_hi"):
        put(f"baseline.{tag}.{col}", r[col])
table(["Family", "Configuration", "Macro AUC mean (SD) [95% CI]", "Macro F1 mean (SD) [95% CI]",
       "Micro F1 mean (SD)"], rows)
supp = ms[ms["supplementary"].astype(bool)]
w(f"Not shown: {len(supp)} supplementary rows (" + ", ".join(supp["cfg_id"]) + ") that use the "
  "24-component PCA of round-2 cv_metrics.py; they exist only to reproduce the round-2 CV table. "
  f"Rank of the production model by CV macro AUC: {order.index(PROD) + 1} of {len(order)} "
  f"({sum(1 for f in order[:order.index(PROD)] if f in eligible_floor) + 1} of the {n_floor} "
  "floor-passing families).")
w()
put("cv.production_rank_auc_all", order.index(PROD) + 1)
put("cv.production_rank_auc_floor", sum(1 for f in order[:order.index(PROD)] if f in eligible_floor) + 1)

# ----------------------------------------------------------------------------- 4. per-level F1
ppl = csv("results/selection/production_per_level.csv")
heading("4. Production model per level, cross-validation (per-level F1)",
        ["results/selection/production_per_level.csv"])
w(f"kNN (n = 21), P50; mean (SD) over the 5 repeats; F1 interval: {BOOT}.")
w()
rows = []
for _, r in ppl.iterrows():
    lv = r["level"]
    rows.append([lv, int(r["n_positive"]), f"{f3(r['CV_F1_mean5'])} ({f3(r['CV_F1_sd5'])})",
                 f"[{f3(r['CV_F1_avg5_ci95_pctboot_lo'])}, {f3(r['CV_F1_avg5_ci95_pctboot_hi'])}]",
                 f3(r["CV_P_mean5"]), f3(r["CV_R_mean5"]), f3(r["CV_AUC_mean5"])])
    for col in ("n_positive", "CV_F1_mean5", "CV_F1_sd5", "CV_F1_avg5_ci95_pctboot_lo",
                "CV_F1_avg5_ci95_pctboot_hi", "CV_P_mean5", "CV_R_mean5", "CV_AUC_mean5"):
        put(f"per_level.{lv}.{col}", r[col])
table(["Level", "Core verbs at level", "F1 mean (SD)", "F1 95% CI", "Precision", "Recall", "AUC"], rows)
close(ppl["CV_F1_mean5"].mean(), ms_main.loc[PROD, "CV_macro_F1_mean5"], 1e-12,
      "mean of per-level F1 equals macro F1 of the production model")

# ----------------------------------------------------------------------------- 5. encoder ablation
ea = csv("results/selection/encoder_ablation.csv")
eaw = ea[ea["encoding"].str.startswith("per word")]
ENC = [("mpnet", "all-mpnet-base-v2 (training encoder)"), ("minilm", "all-MiniLM-L6-v2"),
       ("droberta", "all-distilroberta-v1")]
heading("5. Encoder ablation at matched configurations (replaces v2 Table 7)",
        ["results/selection/encoder_ablation.csv", "revision_pipeline/cv_results_mpnet.csv",
         "revision_pipeline/cv_results_minilm.csv", "revision_pipeline/cv_results_droberta.csv"])
w("CV macro AUC (mean of 5 repeats) with each family at its MPNet structural best configuration "
  "and percentile; the outer partitions, calibration and thresholds are those of Section 3. Rank "
  "in brackets, among the families shown, within each encoder.")
w()
piv = eaw.pivot_table(index="family", columns="encoder_tag", values="CV_macro_AUC_mean5", aggfunc="first")
ranks = {t: piv[t].rank(ascending=False, method="min").astype(int) for t, _ in ENC}
rows = []
for fam in piv.sort_values("mpnet", ascending=False).index:
    rows.append([fam + (" (production)" if fam == PROD else "")]
                + [f"{f3(piv.loc[fam, t])} ({ranks[t][fam]})" for t, _ in ENC])
    for t, _ in ENC:
        put(f"encoder.{t}.{slug(fam)}.CV_macro_AUC_mean5", piv.loc[fam, t])
        put(f"encoder.{t}.{slug(fam)}.rank", int(ranks[t][fam]))
    close(piv.loc[fam, "mpnet"], ms_main.loc[fam, "CV_macro_AUC_mean5"], 1e-12,
          f"encoder ablation MPNet row {fam} equals the main grid")
table(["Family", *[lab for _, lab in ENC]], rows)
for t, lab in ENC:
    kr = eaw[(eaw["encoder_tag"] == t) & (eaw["family"] == PROD)].iloc[0]
    for col in ("CV_macro_AUC_mean5", "CV_macro_F1_mean5", "HO_within1_mean5", "HO_within1_all_mean5"):
        put(f"encoder.{t}.production.{col}", kr[col])
    ob = ea[(ea["encoder_tag"] == t) & (ea["family"] == PROD) & ea["encoding"].str.startswith("one batch")]
    check(len(ob) == 1, f"one-batch check row for {t}")
    close(ob.iloc[0]["CV_macro_AUC_mean5"], kr["CV_macro_AUC_mean5"], 1e-6,
          f"per-word vs one-batch encoding, production model, {t}")
    check(round(kr["CV_macro_AUC_mean5"], 3) == round(float(kr["v2_table7_CV_macro_AUC"]), 3),
          f"production AUC {t} rounds to the v2 Table 7 value")
rows = []
for t, lab in ENC:
    kr = eaw[(eaw["encoder_tag"] == t) & (eaw["family"] == PROD)].iloc[0]
    rows.append([lab, f3(kr["CV_macro_AUC_mean5"]), f3(kr["CV_macro_F1_mean5"]), f3(kr["HO_within1_mean5"]),
                 f"{ranks[t][PROD]} of {len(piv)}"])
w("Production model across encoders:")
w()
table(["Encoder", "CV macro AUC", "CV macro F1", "Held-out within-one (accepted verbs)", "AUC rank"], rows)
# round-2 default-configuration files: does MPNet give the highest AUC for every configuration?
cvr = {t: csv(f"revision_pipeline/cv_results_{t}.csv").groupby("model")["macro_AUC"].mean() for t, _ in ENC}
models = [m for m in cvr["mpnet"].index if m not in ("majority-level", "freq-random")]
n_mp_best = sum(1 for m in models if cvr["mpnet"][m] >= max(cvr["minilm"][m], cvr["droberta"][m]))
put("encoder.round2_defaults.n_configs", len(models))
put("encoder.round2_defaults.n_mpnet_highest_auc", n_mp_best)
w(f"Round-2 default configurations at P50 (traceability files `revision_pipeline/cv_results_*.csv`): "
  f"MPNet gives the highest macro AUC of the three encoders for {n_mp_best} of {len(models)} configurations.")
w()

# ----------------------------------------------------------------------------- 6. selection
heading("6. Model selection (replaces v2 Tables 8a and 8b)",
        ["results/selection/selection_table.csv", "results/winner_and_constants.json",
         "results/selection/baselines.csv"])
w("Rule (`select_model.rule_balanced_maximin`): among the eligible families that pass the adequacy "
  "floor (PenalizedObj >= 0.5), each criterion is min-max normalized across those families; the score "
  "is the smallest of the three normalized values; the highest score wins (ties by PenalizedObj). "
  "Criteria, all at the family's structural best configuration and percentile: PenalizedObj (Eq. 19), "
  "CV macro AUC (mean of 5 repeats), held-out within-one-level accuracy over accepted verbs (full "
  "production pipeline, mean of outer seeds 13 to 17).")
w()
rows = []
for _, r in sel.sort_values(["rank", "PenalizedObj"], ascending=[True, False], na_position="last").iterrows():
    fam = r["family"]
    k = slug(fam)
    elig = bool(r["eligible"])
    for col in ("PenalizedObj", "SCS", "acceptance", "CV_macro_AUC_mean5", "HO_within1_mean5",
                "HO_within1_all_mean5", "HO_within1_seed13", "n_obj", "n_auc", "n_ho", "score", "rank"):
        put(f"selection.{k}.{col}", r[col])
    put(f"selection.{k}.config", cfg_text(r["cfg_id"], r["percentile"]))
    put(f"selection.{k}.eligible", elig)
    put(f"selection.{k}.floor_pass", bool(r["floor_pass"]))
    put(f"selection.{k}.limiting_criterion", r["limiting_criterion"])
    put(f"selection.{k}.v2_two_stage_score_HOseed13", r["v2_two_stage_score_HOseed13"])
    if not elig:
        name = f"{fam} (reference, not eligible)"
    else:
        name = fam + (" (selected)" if bool(r["selected"]) else "")
    na = "n/a"
    rows.append([name, cfg_text(r["cfg_id"], r["percentile"]), f3(r["PenalizedObj"]),
                 f3(r["CV_macro_AUC_mean5"]) if elig else na, f3(r["HO_within1_mean5"]) if elig else na,
                 f3(r["n_obj"]) if r["floor_pass"] and elig else "", f3(r["n_auc"]) if r["floor_pass"] and elig else "",
                 f3(r["n_ho"]) if r["floor_pass"] and elig else "",
                 f3(r["score"]) if r["floor_pass"] and elig else ("below floor" if elig else na),
                 str(r["limiting_criterion"]) if isinstance(r["limiting_criterion"], str) else ""])
table(["Family", "Configuration", "PenalizedObj", "CV macro AUC", "Held-out within-one",
       "n_obj", "n_auc", "n_ho", "Score (min)", "Limiting"], rows)
sj = wjs["selection"]
put("selection.rule", sj["rule"])
put("selection.adequacy_floor", sj["adequacy_floor"])
put("selection.winner", sj["winner_family"])
put("selection.winner_score", sj["score"])
put("selection.runner_up", sj["runner_up"])
put("selection.runner_up_score", sj["runner_up_score"])
put("selection.margin", sj["margin"])
put("selection.n_floor_passing_eligible", n_floor)
ts = sj["v2_two_stage_same_inputs"]
put("selection.two_stage_v2.winner", ts["winner"])
put("selection.two_stage_v2.score", ts["score"])
close(ts["score"], sel_by_fam.loc[ts["winner"], "v2_two_stage_score_HOseed13"], 1e-9,
      "two-stage score, json vs selection_table")
w(f"Selected: **{sj['winner_family']}** ({cfg_text(sj['winner_cfg_id'], sj['winner_percentile'])}), "
  f"score {f3(sj['score'])}; runner-up {sj['runner_up']} {f3(sj['runner_up_score'])}; margin "
  f"{f3(sj['margin'])}. The production model's score is limited by its normalized PenalizedObj "
  f"(n_obj = {f3(sel_by_fam.loc[PROD, 'n_obj'])}).")
w()
w(f"Trivial reference for the held-out criterion: predicting the majority level (Ap) for every verb "
  f"gives within-one {f3(maj['HO_within1'])} [{f3(maj['HO_within1_ci95_pctboot_lo'])}, "
  f"{f3(maj['HO_within1_ci95_pctboot_hi'])}] (all verbs accepted); prevalence-proportional random "
  f"levels give {f3(frm['HO_within1'])}. The production model reaches "
  f"{f3(sel_by_fam.loc[PROD, 'HO_within1_mean5'])} over the verbs it accepts and "
  f"{f3(sel_by_fam.loc[PROD, 'HO_within1_all_mean5'])} over all held-out verbs (a rejected verb "
  "counts as a miss).")
w()
rsw0 = csv("results/selection/robustness_scs_weights.csv")
b_row = rsw0[rsw0["setting"] == "baseline"].iloc[0]
check(b_row["A_seed13_winner"] == ts["winner"], "two-stage winner, robustness baseline vs json")
shortlist = str(b_row["A_seed13_shortlist"]).split(";")
check(PROD in shortlist, "production model in the two-stage shortlist")
sl_auc = sel_by_fam.loc[shortlist, "CV_macro_AUC_mean5"].sort_values()
check(sl_auc.index[0] == PROD, "production model has the lowest CV AUC in the shortlist")
gap_auc = float(sl_auc.iloc[1] - sl_auc.iloc[0])
put("selection.two_stage_v2.shortlist", ";".join(shortlist))
put("selection.two_stage_v2.production_auc_gap_to_next", gap_auc)
put("selection.two_stage_v2.production_score", sel_by_fam.loc[PROD, "v2_two_stage_score_HOseed13"])
w(f"Disclosure: the round-2 two-stage form of the rule (stage-one maximin of PenalizedObj and CV AUC, "
  f"shortlist, then maximin with held-out seed 13 re-normalized inside the shortlist), applied to the "
  f"same matched inputs, selects **{ts['winner']}** with {f3(ts['score'])}. Shortlist: "
  f"{', '.join(shortlist)}. The production model scores "
  f"{f3(sel_by_fam.loc[PROD, 'v2_two_stage_score_HOseed13'])} there because, re-normalized inside the "
  f"shortlist, its CV AUC ({f4(sl_auc.iloc[0])}) is the lowest, {f4(gap_auc)} below the next "
  f"({sl_auc.index[1]}, {f4(sl_auc.iloc[1])}). Section 9 reports how this depends on the inputs.")
w()

# ----------------------------------------------------------------------------- 7. held-out pipeline
heading("7. Held-out evaluation of the full production pipeline at matched configurations (replaces v2 Table 11)",
        ["results/selection/matched_supervised.csv", "results/selection/selection_table.csv",
         "results/selection/baselines.csv"])
w("Outer 5-fold split of the core (MultilabelStratifiedKFold, seeds 13 to 17): calibration and "
  "thresholds fitted on the outer-training verbs only, then the production pipeline (per-level "
  "thresholds, top-2 margin collapse with delta 0.060, primary = highest calibrated score among "
  "accepted levels) applied to the held-out verbs. Mean over the 5 outer seeds. Gate recall = share "
  "of held-out verbs accepted; primary in labels and within-one are over accepted verbs; "
  "within-one (all) counts rejected verbs as misses. Interval for within-one (accepted, pooled): "
  f"{BOOT}.")
w()
rows = []
for fam in ms_main.sort_values("HO_within1_mean5", ascending=False).index:
    r = ms_main.loc[fam]
    k = slug(fam)
    for col in ("HO_gate_mean5", "HO_prim_mean5", "HO_within1_mean5", "HO_within1_all_mean5", "HO_mdist_mean5",
                "HO_f1_mean5", "HO_auc_mean5", "HO_within1_pooled_avg5", "HO_within1_pooled_avg5_ci95_pctboot_lo",
                "HO_within1_pooled_avg5_ci95_pctboot_hi"):
        put(f"heldout.{k}.{col}", r[col])
    if fam in sel_by_fam.index:
        close(r["HO_within1_mean5"], sel_by_fam.loc[fam, "HO_within1_mean5"], 1e-12, f"HO within-one {fam}")
    rows.append([fam + (" (production)" if fam == PROD else "") + ("" if r["floor_pass"] else " (below floor)"),
                 f3(r["HO_gate_mean5"]), f3(r["HO_prim_mean5"]),
                 f"{f3(r['HO_within1_mean5'])} [{f3(r['HO_within1_pooled_avg5_ci95_pctboot_lo'])}, "
                 f"{f3(r['HO_within1_pooled_avg5_ci95_pctboot_hi'])}]",
                 f3(r["HO_within1_all_mean5"]), f2(r["HO_mdist_mean5"]), f3(r["HO_f1_mean5"]), f3(r["HO_auc_mean5"])])
for lab, r in (("Baseline: majority level (Ap)", maj), ("Baseline: prevalence-proportional random level", frm)):
    rows.append([lab, f3(r["HO_gate"]), f3(r["HO_prim"]),
                 f"{f3(r['HO_within1'])} [{f3(r['HO_within1_ci95_pctboot_lo'])}, {f3(r['HO_within1_ci95_pctboot_hi'])}]",
                 f3(r["HO_within1_all"]), f2(r["HO_mdist"]), "", "0.500"])
table(["Family", "Gate recall", "Primary in labels", "Within-one (accepted) [95% CI]", "Within-one (all)",
       "Mean ordinal distance", "Macro F1", "Macro AUC"], rows)
seeds = [13, 14, 15, 16, 17]
kseed = [float(sel_by_fam.loc[PROD, f"HO_within1_seed{s}"]) for s in seeds]
for s, v in zip(seeds, kseed):
    put(f"heldout.production.HO_within1_seed{s}", v)
w("Production model, held-out within-one by outer seed: "
  + ", ".join(f"seed {s} {f3(v)}" for s, v in zip(seeds, kseed))
  + f". Seed 13 (the only seed used in round 2) is its most favourable repeat.")
w()
check(max(kseed) == kseed[0], "seed 13 is the production model's best held-out repeat (text above)")

# ----------------------------------------------------------------------------- 8. paired comparisons
mc = csv("results/selection/mcnemar_vs_production.csv")
heading("8. Paired comparisons with the production model",
        ["results/selection/matched_supervised.csv", "results/selection/mcnemar_vs_production.csv"])
w("Differences are other family minus kNN-21 P50. Bootstrap: " + BOOT + ". Exact McNemar tests on the "
  "held-out decision 'primary accepted and within one level' (all held-out verbs); one test per outer "
  "seed, not corrected for multiplicity. The pooled count over the 5 seeds reuses the same verbs and "
  "is descriptive only.")
w()
rows = []
fams_cmp = [f for f in sel.sort_values("rank")["family"] if f in eligible_floor and f != PROD]
for fam in fams_cmp:
    r = ms_main.loc[fam]
    k = slug(fam)
    sub = mc[mc["family"] == fam].set_index("repeat")
    pvals = [float(sub.loc[f"seed{s}", "heldoutW1all_p_exact_mcnemar"]) for s in seeds]
    pooled = sub.loc["pooled5_NOT_INDEPENDENT_descriptive"]
    s13 = sub.loc["seed13"]
    for col in ("dCV_macro_AUC_vs_kNN21P50_avg5_ci95_pctboot_lo", "dCV_macro_AUC_vs_kNN21P50_avg5_ci95_pctboot_hi",
                "dHO_within1_pooled_vs_kNN21P50_avg5_ci95_pctboot_lo",
                "dHO_within1_pooled_vs_kNN21P50_avg5_ci95_pctboot_hi"):
        put(f"paired.{k}.{col}", r[col])
    for s, p in zip(seeds, pvals):
        put(f"paired.{k}.mcnemar_heldoutW1all_p_seed{s}", p)
    put(f"paired.{k}.mcnemar_heldoutW1all_seed13_ref_only", s13["heldoutW1all_ref_only_correct"])
    put(f"paired.{k}.mcnemar_heldoutW1all_seed13_cmp_only", s13["heldoutW1all_cmp_only_correct"])
    put(f"paired.{k}.mcnemar_heldoutW1all_pooled_p_descriptive", pooled["heldoutW1all_p_exact_mcnemar"])
    rows.append([fam,
                 f"[{f3(r['dCV_macro_AUC_vs_kNN21P50_avg5_ci95_pctboot_lo'])}, {f3(r['dCV_macro_AUC_vs_kNN21P50_avg5_ci95_pctboot_hi'])}]",
                 f"[{f3(r['dHO_within1_pooled_vs_kNN21P50_avg5_ci95_pctboot_lo'])}, {f3(r['dHO_within1_pooled_vs_kNN21P50_avg5_ci95_pctboot_hi'])}]",
                 f"{int(s13['heldoutW1all_ref_only_correct'])}/{int(s13['heldoutW1all_cmp_only_correct'])}, p = {fp(pvals[0])}",
                 f"{fp(min(pvals[1:]))} to {fp(max(pvals[1:]))}",
                 fp(float(pooled["heldoutW1all_p_exact_mcnemar"]))])
table(["Family", "Diff. CV macro AUC, 95% CI", "Diff. held-out within-one, 95% CI",
       "Seed 13: kNN only / other only, exact p", "Seeds 14 to 17, p range", "Pooled p (descriptive)"], rows)
n_ci_zero = sum(1 for fam in fams_cmp
                if ms_main.loc[fam, "dCV_macro_AUC_vs_kNN21P50_avg5_ci95_pctboot_lo"] <= 0
                <= ms_main.loc[fam, "dCV_macro_AUC_vs_kNN21P50_avg5_ci95_pctboot_hi"]
                and ms_main.loc[fam, "dHO_within1_pooled_vs_kNN21P50_avg5_ci95_pctboot_lo"] <= 0
                <= ms_main.loc[fam, "dHO_within1_pooled_vs_kNN21P50_avg5_ci95_pctboot_hi"])
put("paired.n_families_both_intervals_include_zero", n_ci_zero)
w(f"Families whose CV AUC and held-out within-one intervals both include zero: {n_ci_zero} of "
  f"{len(fams_cmp)}. A single-seed p below 0.05 is not evidence that kNN-21 is better: it is "
  "uncorrected, seed 13 is kNN-21's most favourable repeat, and the other seeds do not repeat it.")
w()

# ----------------------------------------------------------------------------- 9. robustness
heading("9. Robustness of the selection",
        ["results/selection/" + f for f in ("robustness_scs_weights.csv", "robustness_heldout_variants.csv",
         "robustness_guardrails.csv", "robustness_decision_margins.csv", "robustness_aggregation.csv",
         "robustness_legacy_knn_supervised.csv", "selection_table.csv")])
rsw = csv("results/selection/robustness_scs_weights.csv")
rhv = csv("results/selection/robustness_heldout_variants.csv")
rgd = csv("results/selection/robustness_guardrails.csv")
rdm = csv("results/selection/robustness_decision_margins.csv")
rag = csv("results/selection/robustness_aggregation.csv")
rlk = csv("results/selection/robustness_legacy_knn_supervised.csv")

b0 = rhv[(rhv["menu"] == "A") & (rhv["ho"] == "mean5") & (rhv["rule"] == "proposed")].iloc[0]
check(b0["winner"] == PROD, "robustness baseline winner")
close(b0["score"], sj["score"], 1e-9, "robustness baseline score vs selection")
joint = rsw[rsw["setting"].str.startswith("joint")]
single = rsw[rsw["setting"].str.startswith("single")]
check(len(joint) == 500 and len(single) == 20, "500 joint and 20 single-weight settings")


def counts(series):
    vc = series.fillna("none (no family passes the floor)").value_counts()
    return "; ".join(f"{k} {v}" for k, v in sorted(vc.items(), key=lambda kv: (-kv[1], kv[0])))


def put_counts(prefix, series):
    vc = series.fillna("none").value_counts()
    for k, v in vc.items():
        put(f"{prefix}.{slug(k)}", int(v))


put_counts("robustness.scs_joint500.proposed", joint["proposed_A_winner"])
put_counts("robustness.scs_joint500.structural_only", joint["A_structural_winner"])
put_counts("robustness.scs_joint500.two_stage_v2_seed13", joint["A_seed13_winner"])
put_counts("robustness.scs_joint500.proposed_legacy_admitted", joint["proposed_B_winner"])
put_counts("robustness.scs_single20.proposed", single["proposed_A_winner"])
put_counts("robustness.scs_single20.structural_only", single["A_structural_winner"])
all9 = joint[joint["A_n_floor"] == n_floor]
put_counts("robustness.scs_joint500_all_floor_pass.proposed", all9["proposed_A_winner"])
put("robustness.scs_joint500_all_floor_pass.n", len(all9))
kw = joint[joint["proposed_A_winner"] == PROD]
closest = kw.loc[kw["proposed_A_margin"].idxmin()]
put("robustness.scs_joint500.closest_kNN_margin", closest["proposed_A_margin"])
put("robustness.scs_joint500.closest_kNN_setting", closest["setting"])

ho_lab = {"seed13": "seed 13", "mean5": "mean of seeds 13 to 17", "seed14": "seed 14", "seed15": "seed 15",
          "seed16": "seed 16", "seed17": "seed 17", "all_verbs_mean5": "all verbs, mean", "all_verbs_seed13":
          "all verbs, seed 13"}
hvA = rhv[rhv["menu"] == "A"]
prop_wins = hvA[hvA["rule"] == "proposed"]
two = hvA[hvA["rule"] == "two_stage"]
for _, r in hvA.iterrows():
    put(f"robustness.heldout_variant.{r['ho']}.{r['rule']}.winner", r["winner"])
    put(f"robustness.heldout_variant.{r['ho']}.{r['rule']}.score", r["score"])
g_win = rgd["proposed_A_winner"]
put("robustness.guardrails.n_settings", len(rgd))
put("robustness.guardrails.n_production_wins", int((g_win == PROD).sum()))
for _, r in rgd[g_win != PROD].iterrows():
    put(f"robustness.guardrails.{slug(r['setting'])}.winner", r["proposed_A_winner"])
    put(f"robustness.guardrails.{slug(r['setting'])}.score", r["proposed_A_score"])
dm_all = rdm[rdm["ho_column"] == "HO_within1_mean5"]
dm = dm_all[dm_all["flip_delta"].notna()]
CRIT = [("CV_macro_AUC_mean5", "CV macro AUC"), ("HO_within1_mean5", "held-out within-one"),
        ("PenalizedObj", "PenalizedObj")]
margin_text = []
for crit, lab in CRIT:
    own = dm[(dm["family"] == PROD) & (dm["criterion"] == crit)]
    oth = dm[(dm["family"] != PROD) & (dm["criterion"] == crit)]
    span = float(dm_all[dm_all["criterion"] == crit]["searched_span"].max())
    part = f"{lab}: "
    if len(own):
        r = own.loc[own["flip_delta"].abs().idxmin()]
        put(f"robustness.decision_margin.{crit}.production.flip_delta", r["flip_delta"])
        put(f"robustness.decision_margin.{crit}.production.new_winner", r["new_winner"])
        part += f"kNN-21 {r['flip_delta']:+.6f} -> {r['new_winner']}"
    else:
        part += f"no change of kNN-21 within +/-{span:g}"
    if len(oth):
        r = oth.loc[oth["flip_delta"].abs().idxmin()]
        put(f"robustness.decision_margin.{crit}.competitor.family", r["family"])
        put(f"robustness.decision_margin.{crit}.competitor.flip_delta", r["flip_delta"])
        put(f"robustness.decision_margin.{crit}.competitor.new_winner", r["new_winner"])
        part += f"; nearest competitor change {r['family']} {r['flip_delta']:+.6f} -> {r['new_winner']}"
    else:
        part += f"; no competitor change within +/-{span:g}"
    margin_text.append(part)
lk = rlk[rlk["cfg_id"] == sel_by_fam.loc["Legacy kNN", "cfg_id"]]
check(len(lk) == 5, "Legacy kNN evaluation-only rows, 5 seeds")
put("robustness.legacy_admitted.CV_macro_AUC_mean5", lk["CV_macro_AUC"].mean())
put("robustness.legacy_admitted.HO_within1_mean5", lk["HO_within1"].mean())
put("robustness.legacy_admitted.HO_within1_all_mean5", lk["HO_within1_all"].mean())
hvB = rhv[(rhv["menu"] == "B") & (rhv["ho"] == "mean5")].set_index("rule")
put("robustness.legacy_admitted.proposed.winner", hvB.loc["proposed", "winner"])
put("robustness.legacy_admitted.proposed.score", hvB.loc["proposed", "score"])
put("robustness.legacy_admitted.proposed.kNN_score", hvB.loc["proposed", "kNN_score"])
put("robustness.legacy_admitted.two_stage.winner", hvB.loc["two_stage", "winner"])
put("robustness.legacy_admitted.two_stage.score", hvB.loc["two_stage", "score"])
agA = rag[rag["set"] == "A_matched_HOmean5"].set_index("rule")
sp = agA.loc["Spearman(obj, AUC)"]
put("robustness.spearman_obj_auc.rho", sp["score"])
put("robustness.spearman_obj_auc.detail", sp["detail"])
three = [("3C equal weights", "equal-weight sum of the three normalized criteria"),
         ("3C rank-sum", "sum of the three ranks"),
         ("3C one-stage maximin", "minimum of the three normalized criteria (the rule in use)"),
         ("3C weighted-sum simplex share", "share of weight vectors on a 0.01 simplex grid won"),
         ("3C Pareto front", "Pareto front of the three criteria"),
         ("2C rank-sum", "structure and CV AUC only: rank sum"),
         ("2C maximin (stage one)", "structure and CV AUC only: maximin"),
         ("2C weighted sum w=0.5", "structure and CV AUC only: equal-weight sum"),
         ("2C eps1 (best AUC within 0.075 of best obj)", "best CV AUC within 0.075 of the best PenalizedObj"),
         ("2C eps2 (best obj within 0.008 of best AUC)", "best PenalizedObj within 0.008 of the best CV AUC"),
         ("published two-stage", "round-2 two-stage rule with the 5-seed held-out mean")]
for rule, _ in three:
    put(f"robustness.aggregation.{slug(rule)}.winner", agA.loc[rule, "winner"])

w(f"Menu: the {len(sel[sel['eligible'].astype(bool)])} eligible one-vs-rest families ({n_floor} pass the "
  "floor), each at its structural best under the setting tested, with matched supervised numbers. "
  "Held-out criterion: mean of seeds 13 to 17 unless stated.")
w()
rows = [
    ["Rule in use, baseline", f"{b0['winner']} {f3(b0['score'])}; runner-up {b0['runner_up']}, margin {f3(b0['margin'])}"],
    ["Rule in use, 8 held-out variants (each seed, mean, all-verbs)",
     f"{PROD} in {int((prop_wins['winner'] == PROD).sum())} of {len(prop_wins)}"],
    ["Round-2 two-stage form, 8 held-out variants",
     "; ".join(f"{ho_lab[r['ho']]}: {r['winner']} {f3(r['score'])}" for _, r in two.iterrows())],
    ["SCS weights, 500 joint draws (each weight x U(0.5, 1.5)), rule in use", counts(joint["proposed_A_winner"])],
    [f"  of which all {n_floor} families pass the floor ({len(all9)} draws)", counts(all9["proposed_A_winner"])],
    ["  closest win of the production model", f"margin {f6(closest['proposed_A_margin'])} ({closest['setting']})"],
    ["SCS weights, 500 joint draws, structure-only rule", counts(joint["A_structural_winner"])],
    ["SCS weights, 500 joint draws, round-2 two-stage form (seed 13)", counts(joint["A_seed13_winner"])],
    ["SCS weights, 20 single-weight changes (x0.5, x0.75, x1.25, x1.5), rule in use", counts(single["proposed_A_winner"])],
    [f"Guardrails and penalties, {len(rgd)} settings (one change at a time), rule in use",
     f"{PROD} in {int((g_win == PROD).sum())} of {len(rgd)}; "
     + "; ".join(f"{r['setting']}: {r['proposed_A_winner']} {f3(r['proposed_A_score'])}"
                 for _, r in rgd[g_win != PROD].iterrows())],
    ["Smallest single change of one criterion value that changes the winner (others fixed)",
     ". ".join(margin_text)],
    ["Legacy kNN admitted (evaluation-only construction), rule in use",
     f"{hvB.loc['proposed', 'winner']} {f3(hvB.loc['proposed', 'score'])} (kNN-21 {f3(hvB.loc['proposed', 'kNN_score'])}); "
     f"500 joint draws: {counts(joint['proposed_B_winner'])}"],
    ["Legacy kNN admitted, round-2 two-stage form", f"{hvB.loc['two_stage', 'winner']} {f3(hvB.loc['two_stage', 'score'])}"],
    ["Spearman correlation of PenalizedObj and CV AUC across the floor-passing families", sp["detail"]],
]
for rule, lab in three:
    rows.append([f"Other aggregation: {lab}", f"{agA.loc[rule, 'winner']}"
                 + (f" ({agA.loc[rule, 'detail']})" if isinstance(agA.loc[rule, "detail"], str)
                    and rule in ("3C weighted-sum simplex share",) else "")])
table(["Analysis", "Winner(s)"], rows)
w("Legacy kNN, when admitted, wins the rule in use; it is excluded on eligibility grounds (one global "
  f"vote cutoff, no per-level calibrated scores, not exportable by retrain.py), not on performance. Its "
  f"evaluation-only numbers: CV macro AUC {f3(lk['CV_macro_AUC'].mean())}, held-out within-one "
  f"{f3(lk['HO_within1'].mean())} (accepted) and {f3(lk['HO_within1_all'].mean())} (all verbs). "
  "The adequacy floor is absolute while the SCS scale moves with the weights, so in some joint draws "
  "fewer families pass it.")
w()

# ----------------------------------------------------------------------------- 10. extension outcomes
bank = csv("results/Verb List Classified by Model.csv")
cc = csv("results/bank_structure/coverage_cardinality.csv").set_index("bank")
ps = wjs["production_summary"]
heading("10. Extension outcomes of the delivered bank (v2 Table 9, both density ratios)",
        ["results/bank_structure/coverage_cardinality.csv", "results/Verb List Classified by Model.csv",
         "results/winner_and_constants.json"])
acc_mask = bank["Accepted"].astype(str).str.lower() == "true"
n_in, n_acc = len(bank), int(acc_mask.sum())
entries = int(bank["AcceptedCount"].sum())
v2 = cc.loc["v2 delivered bank"]
check((int(v2["inputs"]), int(v2["accepted"]), int(v2["entries"])) == (n_in, n_acc, entries),
      "coverage_cardinality.csv v2 row vs bank recount")
check((ps["accepted_words"], ps["rejected_words"]) == (n_acc, n_in - n_acc), "production_summary vs bank")
for lv in LEVELS:
    check(int(v2[f"entries_{lv}"]) == int(bank[f"accept_{lv}"].sum()) == ps["per_domain_totals"][lv],
          f"entries at {lv}: csv, bank, json")
    check(int(v2[f"primary_{lv}"]) == int((bank.loc[acc_mask, "PrimaryDomain"] == lv).sum()),
          f"primary-domain count {lv}")
for c in range(7):
    check(int(v2[f"verbs_with_{c}_levels"]) == int((bank["AcceptedCount"] == c).sum()) == ps["hist_accepted_count"][str(c)],
          f"cardinality {c}")
v1_key = [i for i in cc.index if i.startswith("v1 bank")]
cols = [("v2", v2)] + ([("v1", cc.loc[v1_key[0]])] if v1_key else [])
rows_def = [("Input verbs", "inputs", "int"), ("Accepted verbs", "accepted", "int"),
            ("Unassigned (rejected) verbs", "unassigned", "int"), ("Acceptance rate", "acceptance", "ratio"),
            ("Verb-level entries", "entries", "int"),
            ("Entries per input verb (v1 'entry rate')", "entries_per_input", "ratio"),
            ("Entries per accepted verb (v2 'levels per accepted verb')", "entries_per_accepted", "ratio")]
rows_def += [(f"Entries at {lv}", f"entries_{lv}", "int") for lv in LEVELS]
rows_def += [(f"Verbs with {c} level{'s' if c != 1 else ''}", f"verbs_with_{c}_levels", "int") for c in range(1, 6)]
rows_def += [(f"Primary domain {lv}", f"primary_{lv}", "int") for lv in LEVELS]
exact = {"acceptance": ("accepted", "inputs"), "entries_per_input": ("entries", "inputs"),
         "entries_per_accepted": ("entries", "accepted")}
rows = []
for lab, col, kind in rows_def:
    cells = []
    for tag, S in cols:
        if kind == "ratio":
            num, den = exact[col]
            v = int(S[num]) / int(S[den])
            close(v, S[col], 1e-6, f"{col} {tag}")
            cells.append(f"{v:.3f} ({int(S[num]):,}/{int(S[den]):,})")
        else:
            v = int(S[col])
            cells.append(f"{v:,}")
        put(f"bank.{tag}.{col}", v)
    rows.append([lab] + cells)
hdr = ["Quantity", "Delivered bank (round 2 and 3)"] + (["Original submission (v1, commit 2a87a33)"] if v1_key else [])
table(hdr, rows)
if not v1_key:
    w("The v1 column is missing: coverage_cardinality.csv was regenerated without git history "
      "(bank_structure.py reads the v1 bank with git show 2a87a33).")
    w()
put("bank.v2.multi_label_verbs", ps["multi_label"])
put("bank.v2.single_label_verbs", ps["single_label"])
put("bank.v2.suggested_only_verbs", ps["suggested_only"])
w(f"Of the {n_acc:,} accepted verbs, {ps['single_label']:,} carry one level and {ps['multi_label']} more than one; "
  f"{ps['suggested_only']} rejected verbs carry a soft (suggested) level at 0.85 x the threshold.")
w()

# ----------------------------------------------------------------------------- 11. gap means core vs extended
gmm = csv("results/bank_structure/gap_means_mpnet.csv")
heading("11. Mean centroid distance by taxonomic gap, core and extended bank (restores v1 Table 8)",
        ["results/bank_structure/gap_means_mpnet.csv", "results/core_structure/core_gap_means.csv"])
w("all-mpnet-base-v2; Spearman rho of d_n against n = 1..5 with the exact one-tailed p (share of the "
  "120 orderings with rho at least the observed one).")
w()
rows = []
for i, r in gmm.iterrows():
    tag = slug(r["labeling"].split("(")[0])
    rows.append([r["labeling"], int(r["entries"])] + [f4(r[f"d{n}"]) for n in range(1, 6)]
                + [f2(r["spearman_rho"]), f4(r["p_exact_one_tailed"])])
    for col in ("entries", "d1", "d2", "d3", "d4", "d5", "spearman_rho", "p_exact_one_tailed"):
        put(f"gap.{tag}.{col}", r[col])
for i, lab in ((2, "All core"), (3, "High-frequency core"), (4, "Low-frequency core")):
    for n in range(1, 6):
        close(gmm.iloc[i][f"d{n}"], gm.iloc[i - 2][f"d{n}"], 1e-6, f"{lab} d{n}: bank_structure vs regen_core_figures")
table(["Labeling", "Entries", "d1", "d2", "d3", "d4", "d5", "rho", "Exact p"], rows)

# ----------------------------------------------------------------------------- 12. encoder trend
et = csv("results/bank_structure/encoder_trend.csv")
heading("12. Gap trend in three encoders (v2 Table 10)", ["results/bank_structure/encoder_trend.csv"])
rows = []
for _, r in et.iterrows():
    tag = slug(r["encoder"].split(" (")[0])
    rows.append([r["encoder"], f"{f2(r['delivered_rho'])} (p = {f4(r['delivered_p_exact'])})",
                 f"{f2(r['pre_collapse_rho'])} (p = {f4(r['pre_collapse_p_exact'])})",
                 f"{f2(r['core_rho'])} (p = {f4(r['core_p_exact'])})"])
    for col in ("delivered_rho", "delivered_p_exact", "pre_collapse_rho", "pre_collapse_p_exact", "core_rho",
                "core_p_exact", "delivered_gap_means", "core_gap_means"):
        put(f"trend.{tag}.{col}", r[col])
mp = et[et["encoder"].str.startswith("all-mpnet")].iloc[0]
for n, v in zip(range(1, 6), str(mp["delivered_gap_means"]).split()):
    close(float(v), gmm.iloc[0][f"d{n}"], 6e-5, f"delivered gap mean d{n}, encoder_trend vs gap_means_mpnet")
table(["Encoder", "Delivered bank (post-collapse)", "Pre-collapse labeling", "Core (381 verbs)"], rows)
w("Exact one-tailed p from the 120 orderings of five values; with five gap values the trend is descriptive.")
w()

# ----------------------------------------------------------------------------- 13. expert agreement
es = load_json("results/expert/expert_summary.json")
lfl = csv("results/expert/like_for_like.csv")
bp = csv("results/expert/baselines_and_permutation.csv")
mo = csv("results/expert/mcnemar_old_vs_prod.csv")
fk = csv("results/expert/fleiss_kappa.csv", keep_default_na=False)
rs = csv("results/expert/rater_subsampling.csv")
sw = csv("results/expert/threshold_sweep.csv")
pv = csv("results/expert/per_verb.csv")
heading("13. Expert agreement, production model (Section 3.3.3)",
        ["results/expert/expert_summary.json", "results/expert/baselines_and_permutation.csv"])
p26 = es["production"]["C26"]
hi_k, hi_n = es["production"]["high"]
lo_k, lo_n = es["production"]["low"]
gate = es["production"]["gate"]
spm = es["production"]["spearman"]
prodC26 = bp[(bp["model"] == "PROD") & (bp["set"] == "C26")].set_index("metric")
for k_, v_ in p26.items():
    put(f"expert.prod.C26.{k_}", v_)
put("expert.prod.high_consensus.hits", hi_k)
put("expert.prod.high_consensus.n", hi_n)
put("expert.prod.low_consensus.hits", lo_k)
put("expert.prod.low_consensus.n", lo_n)
put("expert.prod.fisher_band_p", es["production"]["fisher_band"])
put("expert.prod.p_endorsed_uniform_exact", es["production"]["p_endorsed_uniform_exact"])
put("expert.prod.p_high_consensus_exact", es["production"]["p_high_exact"])
put("expert.prod.spearman_mean_rho", spm["mean_rho"])
put("expert.prod.spearman_p_mc", spm["p_mc"])
put("expert.prod.spearman_mc_draws", spm["B"])
for k_, v_ in gate.items():
    put(f"expert.prod.gate_D29.{k_}", v_)
close(prodC26.loc["endorsed", "p_uniform_exact"], es["production"]["p_endorsed_uniform_exact"], 5e-5,
      "endorsed exact p, csv vs json")
n26 = p26["N"]
rows = [["Strict top-1 (primary = expert plurality level)", frac(p26["strict"], n26), ""],
        ["Endorsed top-1 (primary endorsed by >= 2 of 8)", frac(p26["endorsed"], n26),
         f"uniform chance {prodC26.loc['endorsed', 'uniform_expected'] / n26:.3f}, exact p = "
         f"{fp(es['production']['p_endorsed_uniform_exact'])}; label permutation exact p = "
         f"{fp(prodC26.loc['endorsed', 'p_perm_exact'])}"],
        ["Hard-label Jaccard / precision / recall", f"{f3(p26['J'])} / {f3(p26['P'])} / {f3(p26['R'])}", ""],
        ["High-consensus verbs (>= 4 of 8 on one level), endorsed", frac(hi_k, hi_n),
         f"exact p = {fp(es['production']['p_high_exact'])}"],
        ["Low-consensus verbs, endorsed", frac(lo_k, lo_n), ""],
        ["High vs low consensus", "", f"Fisher exact p = {fp(es['production']['fisher_band'])}"],
        ["Score-vote Spearman (mean per verb)", f"rho = {f3(spm['mean_rho'])}",
         f"permutation p = {fp(spm['p_mc'])} ({spm['B']:,} Monte Carlo draws)"],
        ["Accept/reject gate on 29 pool verbs (None plurality = should reject)",
         f"{gate['correct_accept']} / {int(gate['correct_reject'])} / {int(gate['false_accept'])} / "
         f"{gate['false_reject']} (correct accept / correct reject / false accept / false reject)",
         f"accuracy {f3(gate['accuracy'])}, Fisher exact p = {fp(gate['fisher_p'])}"]]
table(["Statistic (26 accepted rated verbs unless stated)", "Value", "Chance / test"], rows)

heading("14. Like-for-like comparison with the original-submission model",
        ["results/expert/like_for_like.csv", "results/expert/mcnemar_old_vs_prod.csv"])
w("Both models scored on the same normalized verb forms with the same definitions. D30 = all 30 rated "
  "verbs; D29 = the 29 in the 1,650-verb pool; C26 = the 26 the production model accepts. On D30 and "
  "D29 a rejected or missing verb is a miss.")
w()
rows = []
for _, r in lfl.iterrows():
    tag = "old" if r["model"].startswith("OLD (archived") else ("prod" if r["model"].startswith("PROD") else "v1_stored")
    st = slug(r["set"].split(" ")[0]) if tag != "v1_stored" else "D30_as_reported"
    for col in ("N", "accepted", "strict", "endorsed", "J", "P", "R", "P_accepted_only", "coverage"):
        put(f"expert.like_for_like.{tag}.{st}.{col}", r[col])
    rows.append([r["model"], r["set"], f"{int(r['accepted'])}/{int(r['N'])}", frac(int(r["strict"]), int(r["N"])),
                 frac(int(r["endorsed"]), int(r["N"])), f"{f3(r['J'])} / {f3(r['P'])} / {f3(r['R'])}",
                 f"{int(r['coverage'])}/{int(r['N'])}"])
close(lfl[(lfl["model"].str.startswith("PROD")) & (lfl["set"] == "C26")].iloc[0]["J"], p26["J"], 5e-5,
      "C26 Jaccard, like_for_like vs summary")
table(["Model", "Set", "Accepted", "Strict top-1", "Endorsed top-1", "J / P / R", "Coverage"], rows)
rows = []
for _, r in mo.iterrows():
    rows.append([r["set"], r["metric"], int(r["OLD_hits"]), int(r["PROD_hits"]),
                 f"{int(r['OLD_only'])}/{int(r['PROD_only'])}", fp(r["p_exact"])])
    put(f"expert.mcnemar_old_vs_prod.{r['set']}.{r['metric']}.p_exact", r["p_exact"])
w("Paired exact McNemar test, original-submission model (OLD) vs production model (PROD):")
w()
table(["Set", "Metric", "OLD hits", "PROD hits", "OLD only / PROD only", "Exact p"], rows)

heading("15. Expert agreement against baselines and a label-permutation test",
        ["results/expert/baselines_and_permutation.csv"])
w("Guessers answer every verb. Uniform: one level drawn uniformly from six. Core frequency: one level "
  "drawn with the 381-verb core shares of verb-level assignments. Uniform and core-frequency p-values are "
  "exact Poisson-binomial tails P(X >= hits). Always-Ap: constant Application, compared by an exact paired "
  "McNemar test (model only / Ap only). Label permutation: the model's own primaries shuffled across the "
  "verbs it accepts in the set (level mix kept); exact p from the full permutation distribution "
  "(`p_perm_mc` in the CSV is a Monte Carlo cross-check).")
w()
rows = []
for _, r in bp.iterrows():
    key = f"expert.baseline.{r['model'].lower()}.{r['set']}.{r['metric']}"
    for col in ("hits", "N", "uniform_expected", "p_uniform_exact", "corefreq_expected", "p_corefreq_exact",
                "always_Ap_hits", "p_mcnemar_vs_Ap", "perm_null_mean", "p_perm_exact", "p_perm_mc"):
        put(f"{key}.{col}", r[col])
    rows.append([r["model"], r["set"], r["metric"], f"{int(r['hits'])}/{int(r['N'])}",
                 f"{f2(r['uniform_expected'])} (p = {fp(r['p_uniform_exact'])})",
                 f"{f2(r['corefreq_expected'])} (p = {fp(r['p_corefreq_exact'])})",
                 f"{int(r['always_Ap_hits'])}; {int(r['model_only_vs_Ap'])}/{int(r['Ap_only'])}, p = {fp(r['p_mcnemar_vs_Ap'])}",
                 f"{f2(r['perm_null_mean'])} (p = {fp(r['p_perm_exact'])})"])
table(["Model", "Set", "Metric", "Hits", "Uniform expected (p)", "Core-frequency expected (p)",
       "Always Ap hits; model only / Ap only, p", "Permutation null mean (exact p)"], rows)

heading("16. Rater reliability", ["results/expert/fleiss_kappa.csv", "results/expert/rater_subsampling.csv",
                                  "results/expert/expert_summary.json"])
rows = []
for _, r in fk.iterrows():
    rows.append([r["category"], f3(r["prevalence"]), f3(r["observed_agreement"]), f3(r["fleiss_kappa"])])
    put(f"expert.fleiss_kappa.{r['category']}", r["fleiss_kappa"])
table(["Category", "Prevalence", "Observed agreement", "Fleiss kappa"], rows)
put("expert.fleiss_kappa_range_min", es["kappa_range"][0])
put("expert.fleiss_kappa_range_max", es["kappa_range"][1])
w(f"Fleiss kappa per category (8 raters, 30 verbs) ranges from {f3(es['kappa_range'][0])} to "
  f"{f3(es['kappa_range'][1])}.")
w()
rows = []
for _, r in rs.iterrows():
    rows.append([int(r["panel_size"]), f4(r["mean_kappa"]), f"[{f4(r['pct_2_5'])}, {f4(r['pct_97_5'])}]"])
    put(f"expert.rater_subsampling.{int(r['panel_size'])}.mean_kappa", r["mean_kappa"])
w("Mean Fleiss kappa (all categories) over rater subsets of each size, with the 2.5 to 97.5 percentile range:")
w()
table(["Raters", "Mean kappa", "Range"], rows)

heading("17. Threshold sweep of the production model on the 29 rated pool verbs",
        ["results/expert/threshold_sweep.csv"])
rows = []
for _, r in sw.iterrows():
    tag = "after" if r["counting"].startswith("after") else "before"
    for col in ("pool_acceptance", "levels_per_accepted_verb", "rated_accepted_of_29", "endorsed_top1_of_29",
                "strict_top1_of_29", "precision_accepted_only", "recall_of_29", "jaccard_of_29"):
        put(f"expert.threshold_sweep.x{r['scale']:.2f}.{tag}.{col}", r[col])
    rows.append([f2(r["scale"]), r["counting"], f3(r["pool_acceptance"]), f3(r["levels_per_accepted_verb"]),
                 int(r["rated_accepted_of_29"]), int(r["endorsed_top1_of_29"]), int(r["strict_top1_of_29"]),
                 f3(r["precision_accepted_only"]), f3(r["recall_of_29"])])
table(["Threshold scale", "Counting", "Pool acceptance", "Levels per accepted verb", "Rated accepted (of 29)",
       "Endorsed top-1", "Strict top-1", "Precision (accepted)", "Recall"], rows)

heading("18. Votes and model outputs per rated verb (Appendix B, Table B1)", ["results/expert/per_verb.csv"])
rows = []
for _, r in pv.iterrows():
    rows.append([r["verb"], r["rated_form"], r["votes"], int(r["none_votes"]), r["E25"],
                 r["PROD_status"], r["PROD_primary"] if isinstance(r["PROD_primary"], str) else "",
                 r["PROD_hard"] if isinstance(r["PROD_hard"], str) else "",
                 r["OLD_primary"] if isinstance(r["OLD_primary"], str) else ""])
put("expert.n_rated_verbs", len(pv))
table(["Verb", "Rated form", "Votes Kn/Cm/Ap/An/Sn/Ev", "None", "Endorsed levels (>= 2 of 8)",
       "Production status", "Production primary", "Production hard levels", "Original-submission primary"], rows)

# ----------------------------------------------------------------------------- 19. round-2 reference
v2r = csv("results/selection/v2_default_config_reference.csv")
ho2 = csv("revision_pipeline/heldout_results.csv")
heading("19. Round-2 reference values (traceability; not for the round-3 tables)",
        ["results/selection/v2_default_config_reference.csv", "revision_pipeline/heldout_results.csv"])
w("Round 2 evaluated CV and held-out at default configurations and P50 while the structural objective "
  "used each family's own best configuration. These rows reproduce that construction.")
w()
rows = []
for variant, g in v2r.groupby("variant", sort=False):
    r = g.iloc[0]
    rows.append([variant, r["winner"], f"{r['winner_score']:.4f}", str(r["stage2_ran"]), r["mechanism"]])
    put(f"round2.selection.{slug(variant)}.winner", r["winner"])
    put(f"round2.selection.{slug(variant)}.score", r["winner_score"])
table(["Variant", "Winner", "Score", "Stage two ran", "Mechanism"], rows)
rows = []
for _, r in ho2.iterrows():
    rows.append([r["model"], f3(r["gate_recall"]), f3(r["primary_in_labels"]), f3(r["within1"]),
                 f2(r["mean_ordinal_dist"]), f3(r["macro_F1"]), f3(r["macro_AUC"])])
    put(f"round2.heldout_seed13.{slug(r['model'])}.within1", r["within1"])
w("Round-2 held-out table (default configurations, seed 13 only; `methodology_upgrades.py`):")
w()
table(["Model", "Gate recall", "Primary in labels", "Within-one", "Mean ordinal distance", "Macro F1",
       "Macro AUC"], rows)

# ----------------------------------------------------------------------------- 20. change from the original submission
CA = "results/change_attribution"
bcs = csv(f"{CA}/bank_change_summary.csv").set_index("rule")
bcd = csv(f"{CA}/bank_change_distance.csv")
bpv = csv(f"{CA}/bank_change_per_verb.csv", keep_default_na=False)
cmod = csv(f"{CA}/refit_models.csv").set_index("model")
cra = csv(f"{CA}/refit_agreement.csv")
crm = csv(f"{CA}/refit_mcnemar_vs_production.csv")
c22 = csv(f"{CA}/core_family_2x2.csv")
heading("20. Change from the original submission (response letter)",
        [f"{CA}/bank_change_summary.csv", f"{CA}/bank_change_distance.csv", f"{CA}/refit_models.csv",
         f"{CA}/refit_agreement.csv", f"{CA}/refit_mcnemar_vs_production.csv", f"{CA}/core_family_2x2.csv"])
RULES = [("lemma_form", "Lemma-form rule (reported)"), ("first_form", "First-form rule (alternative)")]
lf = bcs.loc["lemma_form"]
check((int(lf["pool"]), int(lf["delivered_accepted"])) == (n_in, n_acc), "change: pool and delivered acceptance vs the bank")
if v1_key:
    v1row = cc.loc[v1_key[0]]
    check((int(lf["v1_raw_inputs"]), int(lf["v1_raw_accepted"])) == (int(v1row["inputs"]), int(v1row["accepted"])),
          "change: v1 bank counts vs coverage_cardinality.csv")
both_pv = bpv[bpv["v1_accepted"].astype(str).eq("True") & bpv["delivered_accepted"].astype(str).eq("True")]
check(len(both_pv) == int(lf["accepted_both"]), "change: accepted by both, per-verb recount")
check(int((both_pv["v1_primary"] == both_pv["delivered_primary"]).sum()) == int(lf["same_primary"]),
      "change: same primary level, per-verb recount")
for rid, _ in RULES:
    d_ = bcd[bcd["rule"] == rid].set_index("distance")
    check(int(d_["verbs"].sum()) == int(bcs.loc[rid, "accepted_both"]) and int(d_.loc[0, "verbs"]) == int(bcs.loc[rid, "same_primary"]),
          f"change: distance distribution ({rid})")
    for col in bcs.columns:
        put(f"change.bank.{rid}.{col}", bcs.loc[rid, col])
    for dist, r in d_.iterrows():
        put(f"change.bank.{rid}.distance_{int(dist)}.verbs", r["verbs"])
        put(f"change.bank.{rid}.distance_{int(dist)}.share", r["share"])
w("Numbers that only the response letter quotes, generated by `revision_pipeline/change_attribution.py` "
  "(definitions, mapping rules and checks in its docstring; full report in `results/change_attribution/change_attribution.md`).")
w()
w("Bank change: the original-submission labels carried over to the 1,650 normalized candidates through "
  "`revision_pipeline/normalization_audit.csv` and compared with the delivered bank. Lemma-form rule: a lemma takes the "
  "v1 row of the raw form equal to it (else the first kept or lemmatized raw form); first-form rule: the first surviving "
  "raw form in the v1 row order.")
w()
rows = []
for lab, col, den in [("v1 bank on its own raw candidates: accepted", "v1_raw_accepted", "v1_raw_inputs"),
                      ("v1 labels on the pool: accepted", "v1_accepted_on_pool", "pool"),
                      ("Delivered bank: accepted", "delivered_accepted", "pool"),
                      ("Same accept or reject decision", "same_decision", "pool"),
                      ("Accepted by both", "accepted_both", "pool"),
                      ("Same primary level (of accepted by both)", "same_primary", "accepted_both"),
                      ("Primary levels within one level (of accepted by both)", "within_one_level", "accepted_both"),
                      ("Identical hard level set (of accepted by both)", "same_level_set", "accepted_both")]:
    rows.append([lab] + [f"{int(bcs.loc[rid, col]):,}/{int(bcs.loc[rid, den]):,} = "
                         f"{f3(int(bcs.loc[rid, col]) / int(bcs.loc[rid, den]))}" for rid, _ in RULES])
rows.append(["Mean distance between the primary levels (positions in Kn to Ev)"]
            + [f3(bcs.loc[rid, "mean_primary_distance"]) for rid, _ in RULES])
table(["Quantity"] + [lab for _, lab in RULES], rows)
rows = []
for dist in range(6):
    rows.append([dist] + [f"{int(r['verbs'])} ({100.0 * r['share']:.1f}%)" for rid, _ in RULES
                          for _, r in bcd[(bcd["rule"] == rid) & (bcd["distance"] == dist)].iterrows()])
w("Distance between the two primary levels, verbs accepted by both banks:")
w()
table(["Distance"] + [lab for _, lab in RULES], rows)

MOD_ORDER = cmod.index.tolist()
check("kNN21_381_P50" in MOD_ORDER, "change: refit of the production model present")
prc = cra[(cra["model"] == "kNN21_381_P50") & (cra["set"] == "C26")].iloc[0]
for k_ in ("N", "accepted", "strict", "endorsed", "coverage"):
    check(int(prc[k_]) == int(p26[k_]), f"change: production refit {k_} on C26 vs expert_summary.json")
for k_ in ("J", "P", "R"):
    close(prc[k_], p26[k_], 5e-6, f"change: production refit {k_} on C26 vs expert_summary.json")
for st_ in ("D30", "D29"):
    lp = lfl[lfl["model"].str.startswith("PROD") & (lfl["set"] == st_)].iloc[0]
    rp = cra[(cra["model"] == "kNN21_381_P50") & (cra["set"] == st_)].iloc[0]
    check((int(rp["strict"]), int(rp["endorsed"])) == (int(lp["strict"]), int(lp["endorsed"])),
          f"change: production refit on {st_} vs like_for_like.csv")
w("Expert agreement of refits (production procedure of `retrain.py`; scikit-learn 1.9.0) of each model family on each "
  "version of the training core: the original 358-verb core (`revision_pipeline/core_notebook_original.csv`) and the "
  "corrected 381-verb core. kNN21_381_P50 is the refit of the production model and reproduces the delivered bank. "
  "D30 = all 30 rated verbs, D29 = the 29 in the candidate pool, OWN = the verbs of D29 the model accepts, C26 = the 26 "
  "the production model accepts; a rejected or missing verb is a miss.")
w()
rows = []
for m_ in MOD_ORDER:
    put(f"change.refit.{m_}.percentile", cmod.loc[m_, "percentile"])
    put(f"change.refit.{m_}.pool_acceptance", cmod.loc[m_, "pool_acceptance"])
    g_ = cra[cra["model"] == m_].set_index("set")
    for st_, r in g_.iterrows():
        for col in ("N", "accepted", "strict", "endorsed", "J", "P", "R", "P_accepted_only", "coverage"):
            put(f"change.refit.{m_}.{st_}.{col}", r[col])
    rows.append([m_, cmod.loc[m_, "family"], cmod.loc[m_, "core"].split(" ")[-2].lstrip("("), f"P{int(cmod.loc[m_, 'percentile'])}",
                 f"{int(g_.loc['D29', 'accepted'])}/29"]
                + [frac(int(g_.loc[s, "strict"]), int(g_.loc[s, "N"])) for s in ("D30", "D29", "OWN", "C26")]
                + [frac(int(g_.loc[s, "endorsed"]), int(g_.loc[s, "N"])) for s in ("D30", "D29", "OWN", "C26")])
table(["Model", "Family", "Core verbs", "Percentile", "Accepted of D29", "Strict D30", "Strict D29", "Strict OWN",
       "Strict C26", "Endorsed D30", "Endorsed D29", "Endorsed OWN", "Endorsed C26"], rows)
rows = []
for m_ in MOD_ORDER:
    g_ = cra[cra["model"] == m_].set_index("set")
    rows.append([m_] + [f"{f3(g_.loc[s, 'J'])} / {f3(g_.loc[s, 'P'])} / {f3(g_.loc[s, 'R'])}"
                        for s in ("D30", "D29", "OWN", "C26")])
w("Hard-label Jaccard / precision / recall against the endorsed levels:")
w()
table(["Model", "D30", "D29", "OWN", "C26"], rows)
rows = []
for _, r in crm.iterrows():
    key = f"change.refit.{r['model']}.mcnemar_vs_production.{r['set']}.{r['metric']}"
    for col in ("model_hits", "production_hits", "model_only", "production_only", "p_exact"):
        put(f"{key}.{col}", r[col])
    rows.append([r["model"], r["set"], r["metric"], int(r["model_hits"]), int(r["production_hits"]),
                 f"{int(r['model_only'])}/{int(r['production_only'])}", fp(r["p_exact"])])
w("Paired exact McNemar test of each refit against the production model:")
w()
table(["Model", "Set", "Metric", "Model hits", "Production hits", "Model only / production only", "Exact p"], rows)
rows = []
for _, r in c22.iterrows():
    key = f"change.core_family.{r['grid']}.{r['set']}.{r['metric']}"
    for col in ("SGD_358", "SGD_381", "kNN_358", "kNN_381", "family_effect", "core_effect", "interaction"):
        put(f"{key}.{col}", r[col])
    rows.append([r["description"], r["set"], r["metric"], int(r["SGD_358"]), int(r["SGD_381"]), int(r["kNN_358"]),
                 int(r["kNN_381"]), f"{r['family_effect']:+.1f}", f"{r['core_effect']:+.1f}", f"{int(r['interaction']):+d}"])
w("Training core x model family (hits; Shapley main effects and interaction):")
w()
table(["Comparison", "Set", "Metric", "SGD, 358 core", "SGD, 381 core", "kNN, 358 core", "kNN, 381 core",
       "Family effect", "Core effect", "Interaction"], rows)

# ----------------------------------------------------------------------------- cross-check keys (no table)
# Keys only (paper_numbers.json); paper_tables.md is unchanged by this block. These are values the
# manuscript prints in prose (split sizes, scan sizes, robustness counts, software versions) so that
# every printed number can be checked against one key.

# Guardrail settings that change something the rule in use depends on (acceptance band, band for
# levels per accepted verb, penalty strength or adequacy floor). Excluded: the baseline row, rows that
# repeat baseline values, and stage-one cutoffs (used only by the round-2 two-stage form).
g_base = rgd[rgd["setting"] == "baseline"].iloc[0]
RULE_COLS = ["ACC_BAND", "ENAC_BAND", "PEN", "floor"]
g_rule = rgd[rgd[RULE_COLS].astype(str).apply(tuple, axis=1) != tuple(str(g_base[c]) for c in RULE_COLS)]
g_cut_only = rgd[(rgd[RULE_COLS].astype(str).apply(tuple, axis=1) == tuple(str(g_base[c]) for c in RULE_COLS))
                 & (rgd["cutoff"].astype(str) != str(g_base["cutoff"]))]
put("robustness.guardrails.n_rule_changes", len(g_rule))
put("robustness.guardrails.n_rule_changes_production_wins", int((g_rule["proposed_A_winner"] == PROD).sum()))
put("robustness.guardrails.n_stage_one_cutoff_only", len(g_cut_only))
put("robustness.guardrails.n_repeat_baseline", len(rgd) - 1 - len(g_rule) - len(g_cut_only))

# Data splits. Outer partitions from the stored fold ids (identical for every configuration); inner
# cross-fitting and the calibration folds use KFold(5): the first n % 5 folds hold n // 5 + 1 verbs.
oof = np.load(path("results/selection/oof_decisions.npz"))
fid = oof["fold_id"]
check(all((fid[i] == fid[0]).all() for i in range(fid.shape[0])), "outer folds identical across configurations")
n_c = fid.shape[2]
check(n_c == n_core_verbs, "fold ids cover the core")
outer_test = [int((fid[0, s] == k).sum()) for s in range(fid.shape[1]) for k in sorted(set(fid[0, s].tolist()))]
outer_train = [n_c - t for t in outer_test]


def kfold_sizes(n, k=5):
    return [n // k + (1 if i < n % k else 0) for i in range(k)]


inner_scored = [t for n in outer_train for t in kfold_sizes(n)]
inner_fit = [n - t for n in outer_train for t in kfold_sizes(n)]
cal_scored = kfold_sizes(n_c)
cal_fit = [n_c - t for t in cal_scored]
for tag, vals in (("outer_test", outer_test), ("outer_train", outer_train), ("inner_scored", inner_scored),
                  ("inner_fit", inner_fit), ("calibration_scored", cal_scored), ("calibration_fit", cal_fit)):
    put(f"splits.{tag}.min", min(vals))
    put(f"splits.{tag}.max", max(vals))
    put(f"splits.{tag}.min_share_pct", 100.0 * min(vals) / n_c)
    put(f"splits.{tag}.max_share_pct", 100.0 * max(vals) / n_c)
put("splits.n_outer_test_parts", len(outer_test))

# Structural scan size.
scan = csv("results/structural_scan_components.csv")
legacy_rows = scan[scan["Model"].str.startswith("Legacy kNN")]
ovr_rows = scan[~scan["Model"].str.startswith("Legacy kNN")]
put("scan.ovr.n_families", ovr_rows["Model"].nunique())
put("scan.ovr.n_settings", len(ovr_rows[["Model", "Params"]].drop_duplicates()))
put("scan.ovr.n_percentiles", ovr_rows["Mode"].nunique())
put("scan.ovr.n_configurations", len(ovr_rows))
put("scan.legacy.n_vote_modes", legacy_rows["Model"].nunique())
put("scan.legacy.n_settings_per_mode", int(legacy_rows.groupby("Model").size().iloc[0]))
put("scan.legacy.n_rows", len(legacy_rows))
check(len(ovr_rows) == put("scan.ovr.settings_x_percentiles", NUM["scan.ovr.n_settings"] * NUM["scan.ovr.n_percentiles"]),
      "scan rows = settings x percentiles")

# Macro AUC gap between CalibratedSVC and the production model (the two-stage form turns on it).
put("selection.two_stage_v2.production_auc_gap_to_calibratedsvc",
    sel_by_fam.loc["CalibratedSVC", "CV_macro_AUC_mean5"] - sel_by_fam.loc[PROD, "CV_macro_AUC_mean5"])

# Expert agreement with hard labels plus soft suggestions.
for k_, v_ in es["production"]["C26_hard_soft"].items():
    put(f"expert.prod.C26_hard_soft.{k_}", v_)

# Software versions recorded by the evaluation runs.
gmeta = load_json("results/selection/grid_meta.json")
emeta = load_json("results/selection/encoder_ablation_meta.json")
put("meta.versions.python", str(gmeta["versions"]["python"]).split(" ")[0])
for k_ in ("numpy", "pandas", "sklearn"):
    put(f"meta.versions.{k_}", gmeta["versions"][k_])
for k_, v_ in emeta["encoders"]["versions"].items():
    put(f"meta.versions.{k_}", v_)

# ----------------------------------------------------------------------------- write
text = "\n".join(MD).rstrip("\n") + "\n"
bad = [ch for ch in (chr(0x2013), chr(0x2014)) if ch in text]
check(not bad, "no en or em dashes in the generated tables")
with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as f:
    json.dump(NUM, f, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False)
    f.write("\n")
print(f"paper_results.py: {len(NUM)} numbers, {sum(1 for l in MD if l.startswith('## '))} sections")
print("wrote", os.path.relpath(OUT_MD, REPO).replace(os.sep, "/"), "and",
      os.path.relpath(OUT_JSON, REPO).replace(os.sep, "/"))
