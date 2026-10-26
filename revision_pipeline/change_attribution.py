"""Change from the original submission: the verb bank and the expert agreement (response letter).

The response letter (round 3) compares the delivered bank and the production model with those of the
original submission (v1). This script writes the numbers it quotes that no other script produces.

A. Bank change on the common candidate pool
   The original-submission bank (legacy/original_submission_bank_v1.csv, a byte copy of
   results/Verb List Classified by Model.csv at commit 2a87a33; 1,824 raw candidates) is carried over
   to the 1,650 normalized candidates through revision_pipeline/normalization_audit.csv and compared
   verb by verb with the delivered bank (results/Verb List Classified by Model.csv).
   A normalized lemma can stand for several raw candidates (for example 'hold', 'held', 'holding').
   Mapping rule used for the reported numbers ("lemma form"): the lemma takes the v1 row of the raw
   candidate whose surface form equals the lemma, which is the input string the current model
   scores; when no raw form equals the lemma, the first raw form in the v1 row order that
   normalization kept or lemmatized (not a merged duplicate); otherwise the first raw form.
   Alternative ("first form"), reported for comparison: the first surviving raw form in the v1 row
   order. The two rules differ only for lemmas that merged several raw forms.
   Reported per rule: acceptance of the v1 labels on the pool, agreement of the accept or reject
   decision, verbs accepted by both banks, same primary level, the distance between the two primary
   levels (positions in the order Kn, Cm, Ap, An, Sn, Ev), the share within one level, the
   primary-level transitions (rejection as a seventh category) and identical hard level sets.

B. Expert-agreement attribution (refits)
   Each model is refitted with the production procedure of retrain.py, imported (OneVsRest fit with
   the scikit-learn fallback to the unweighted fit, out-of-fold empirical-CDF calibration with
   KFold(5, shuffle, seed 13), row-weighted percentile thresholds, top-2 margin collapse with
   delta 0.060, primary = highest calibrated score among the hard labels):
     SGD1e5_358_P60  SGDClassifier in the archived original configuration (hinge, alpha 1e-5, P60;
                     legacy/original-submission-model/.../metadata.json) on the original 358-verb
                     core (core_notebook_original.csv), refitted with the pinned scikit-learn
     SGD1e4_358_P50  SGDClassifier hinge, alpha 1e-4, P50, original core
     kNN21_358_P50   the production configuration (production_model/metadata.json) on the original core
     SGD1e5_381_P55  the original SGD configuration on the corrected 381-verb core (core_merged.csv)
                     at P55, its structural best percentile there
     SGD1e5_381_P60  the same at P60
     kNN21_381_P50   the production configuration on the corrected core: a refit of the production
                     model, which must reproduce the delivered bank (checked)
   Every model is scored on the rated verbs with the definitions of expert_recompute.py:
     Emax(v) = levels with the top vote count (ties allowed, count > 0, None excluded)
     E25(v)  = levels endorsed by at least 2 of 8 experts
     strict top-1 = primary in Emax; endorsed top-1 = primary in E25
     hard Jaccard / precision / recall (J / P / R) against E25, averaged over verbs
   Denominators: D30 (all 30 rated verbs, normalized forms; 'copy' is a core verb and not a
   candidate, so it is a miss for every model), D29 (the 29 in the candidate pool), OWN (the verbs of
   D29 the model accepts) and C26 (the 26 the production model accepts). On D30, D29 and C26 a
   rejected or missing verb is a miss (hit 0, J = P = R = 0). Paired exact McNemar tests compare
   each refit with the production model, whose outputs are read from the delivered bank; on D30 the
   test equals the one on D29 because 'copy' is a miss for both. The core x model-family 2 x 2 gives
   Shapley main effects and the interaction, in hits.

Checks (the script stops when one fails): the v1 bank is internally consistent and every raw
candidate has one audit row; every pool lemma maps to a v1 row; the model configurations are in
retrain.MODEL_GRID; the production refit reproduces production_model/thresholds.json and every hard
pattern, primary level and soft pattern of the delivered bank; the penalized objective of the
corrected-core refits equals results/model_comparison_table5.csv.

Outputs (results/change_attribution/): change_attribution.md (generated report), bank_change_summary.csv,
bank_change_distance.csv, bank_change_transitions.csv, bank_change_per_verb.csv, refit_models.csv,
refit_agreement.csv, refit_mcnemar_vs_production.csv, refit_per_verb.csv, core_family_2x2.csv.
Read-only on every input; every vector comes from all_embeddings.npz (no encoder, no network).
Output is deterministic (no timestamps, LF line endings), so a rerun is byte-identical.
Run from anywhere: python -I -B revision_pipeline/change_attribution.py
"""
import json
import math
import os
import re
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import retrain as rt  # noqa: E402

LEVELS = ["Kn", "Cm", "Ap", "An", "Sn", "Ev"]
V1_BANK = os.path.join(REPO, "legacy", "original_submission_bank_v1.csv")
BANK = os.path.join(REPO, "results", "Verb List Classified by Model.csv")
NORM_AUDIT = os.path.join(HERE, "normalization_audit.csv")
CANDS = os.path.join(HERE, "candidates_final.txt")
CORE381 = os.path.join(HERE, "core_merged.csv")
CORE358 = os.path.join(HERE, "core_notebook_original.csv")
XLSX = os.path.join(REPO, "legacy", "Final File(with model & Expert Classification).xlsx")
V1_META = os.path.join(REPO, "legacy", "original-submission-model", "artifacts",
                       "SGDClassifier_P60_2025-10-12_01-47-29", "metadata.json")
PROD_DIR = os.path.join(REPO, "production_model")
TABLE5 = os.path.join(REPO, "results", "model_comparison_table5.csv")
LFL = os.path.join(REPO, "results", "expert", "like_for_like.csv")
OUT = os.path.join(REPO, "results", "change_attribution")
SURVIVING = ("kept", "lemmatized", "kept+spelling", "merged-duplicate")
REPORT = []


def w(s=""):
    REPORT.append(s)


def check(cond, msg):
    if not cond:
        raise SystemExit(f"change_attribution.py: CHECK FAILED: {msg}")


def pct(k, n):
    return f"{k}/{n} = {100.0 * k / n:.1f}%" if n else "n/a"


def f4(x):
    return f"{x:.4f}"


def table(header, rows):
    w("| " + " | ".join(header) + " |")
    w("|" + "---|" * len(header))
    for r in rows:
        w("| " + " | ".join(str(c) for c in r) + " |")
    w()


def to_csv(df, name):
    df.to_csv(os.path.join(OUT, name), index=False, float_format="%.6f", lineterminator="\n")


def rel(p):
    return os.path.relpath(p, REPO).replace(os.sep, "/")


def truthy(x):
    return str(x).strip().lower() in ("true", "1", "1.0")


def hard_levels(row):
    return [l for l in LEVELS if int(row[f"accept_{l}"]) == 1]


# ============================================================================ A. bank change
v1 = pd.read_csv(V1_BANK, keep_default_na=False, dtype={"Verb": str})
v2 = pd.read_csv(BANK, keep_default_na=False, dtype={"Verb": str})
v1["k"] = v1["Verb"].str.strip().str.lower()
v2["k"] = v2["Verb"].str.strip().str.lower()
check(len(v1) == 1824 and v1["k"].is_unique, "v1 bank: 1,824 unique raw candidates")
POOL, _ = rt.load_candidates(CANDS)
check(len(POOL) == 1650 and len(set(POOL)) == 1650, "candidate pool: 1,650 unique lemmas")
check(len(v2) == 1650 and set(v2["k"]) == set(POOL), "delivered bank covers the candidate pool")
for tag, b in (("v1", v1), ("delivered", v2)):
    for _, r in b.iterrows():
        H = hard_levels(r)
        check(truthy(r["Accepted"]) == bool(H), f"{tag} bank: Accepted agrees with the hard labels ({r['k']})")
        check((r["PrimaryDomain"] in H) if H else (r["PrimaryDomain"] == ""),
              f"{tag} bank: primary level is one of the hard labels ({r['k']})")
V1 = {r["k"]: (hard_levels(r), r["PrimaryDomain"]) for _, r in v1.iterrows()}
V2 = {r["k"]: (hard_levels(r), r["PrimaryDomain"]) for _, r in v2.iterrows()}
V1_POS = {k: i for i, k in enumerate(v1["k"])}

aud = pd.read_csv(NORM_AUDIT, keep_default_na=False, dtype=str)
aud["original"] = aud["original"].str.strip().str.lower()
aud["normalized_lemma"] = aud["normalized_lemma"].str.strip().str.lower()
check(len(aud) == 1824 and aud["original"].is_unique and set(aud["original"]) == set(V1),
      "normalization audit: one row per v1 raw candidate")
FORMS = defaultdict(list)                 # lemma -> surviving raw rows, in v1 row order
for _, a in aud.iterrows():
    if a["action"] in SURVIVING:
        FORMS[a["normalized_lemma"]].append((V1_POS[a["original"]], a["original"], a["action"]))
for lem in FORMS:
    FORMS[lem].sort()
check(set(POOL) <= set(FORMS), "every pool lemma has a surviving raw form")
# lemmas that survive normalization but left the pool in merge_core.py because they joined the corrected core
CORE_JOINED = sorted(set(FORMS) - set(POOL))
_core381_verbs = set(pd.read_csv(os.path.join(HERE, "core_merged.csv"))["verb"].astype(str).str.strip().str.lower())
check(all(lem in _core381_verbs for lem in CORE_JOINED), "lemmas outside the pool are verbs of the corrected core")


def pick_lemma_form(lem):
    rows = FORMS[lem]
    for _, orig, act in rows:
        if orig == lem:
            return orig, "raw form equals the lemma"
    for _, orig, act in rows:
        if act != "merged-duplicate":
            return orig, "first kept or lemmatized raw form"
    return rows[0][1], "first raw form"


def pick_first_form(lem):
    return FORMS[lem][0][1], "first raw form"


RULES = [("lemma_form", "lemma form (reported)", pick_lemma_form),
         ("first_form", "first form (alternative)", pick_first_form)]


def lvl_dist(a, b):
    return abs(LEVELS.index(a) - LEVELS.index(b))


v1_raw_acc = sum(1 for k in V1 if V1[k][0])
v2_acc = sum(1 for k in POOL if V2[k][0])
BANK_SUMMARY, BANK_DIST, MAPPED = [], [], {}
for rid, rlab, pick in RULES:
    mp = {lem: pick(lem) for lem in POOL}
    MAPPED[rid] = mp
    acc1 = {lem: bool(V1[mp[lem][0]][0]) for lem in POOL}
    acc2 = {lem: bool(V2[lem][0]) for lem in POOL}
    both = [lem for lem in POOL if acc1[lem] and acc2[lem]]
    dist = [lvl_dist(V1[mp[lem][0]][1], V2[lem][1]) for lem in both]
    jac = [len(set(V1[mp[lem][0]][0]) & set(V2[lem][0])) / len(set(V1[mp[lem][0]][0]) | set(V2[lem][0]))
           for lem in both]
    branch = pd.Series([mp[lem][1] for lem in POOL]).value_counts()
    rec = dict(rule=rid, pool=len(POOL),
               lemmas_with_several_raw_forms=sum(1 for lem in POOL if len(FORMS[lem]) > 1),
               mapped_raw_form_equals_lemma=int(branch.get("raw form equals the lemma", 0)),
               mapped_first_kept_or_lemmatized=int(branch.get("first kept or lemmatized raw form", 0)),
               mapped_first_raw_form=int(branch.get("first raw form", 0)),
               v1_raw_inputs=len(V1), v1_raw_accepted=v1_raw_acc, v1_raw_acceptance=v1_raw_acc / len(V1),
               v1_accepted_on_pool=sum(acc1.values()), v1_acceptance_on_pool=sum(acc1.values()) / len(POOL),
               delivered_accepted=v2_acc, delivered_acceptance=v2_acc / len(POOL),
               same_decision=sum(acc1[l] == acc2[l] for l in POOL),
               same_decision_share=sum(acc1[l] == acc2[l] for l in POOL) / len(POOL),
               accepted_both=len(both),
               accepted_v1_only=sum(acc1[l] and not acc2[l] for l in POOL),
               accepted_delivered_only=sum(acc2[l] and not acc1[l] for l in POOL),
               rejected_both=sum(not acc1[l] and not acc2[l] for l in POOL),
               same_primary=dist.count(0), same_primary_share=dist.count(0) / len(both),
               within_one_level=sum(d <= 1 for d in dist), within_one_level_share=sum(d <= 1 for d in dist) / len(both),
               mean_primary_distance=float(np.mean(dist)),
               same_level_set=sum(set(V1[mp[l][0]][0]) == set(V2[l][0]) for l in both),
               mean_level_set_jaccard=float(np.mean(jac)))
    check(rec["accepted_both"] + rec["accepted_v1_only"] + rec["accepted_delivered_only"] + rec["rejected_both"] == 1650,
          f"{rid}: decision cells add up to the pool")
    BANK_SUMMARY.append(rec)
    for d in range(6):
        BANK_DIST.append(dict(rule=rid, distance=d, verbs=dist.count(d), share=dist.count(d) / len(both)))
SUMM = {r["rule"]: r for r in BANK_SUMMARY}
L_, F_ = SUMM["lemma_form"], SUMM["first_form"]

# transitions (lemma-form rule), all 1,650 verbs: v1 primary (or none) x delivered primary (or none)
CATS = LEVELS + ["none"]
T = np.zeros((7, 7), dtype=int)
for lem in POOL:
    p1 = V1[MAPPED["lemma_form"][lem][0]][1] or "none"
    p2 = V2[lem][1] or "none"
    T[CATS.index(p1), CATS.index(p2)] += 1
check(T.sum() == 1650 and int(np.trace(T[:6, :6])) == L_["same_primary"], "transition table")
trans = pd.DataFrame(T, columns=[f"delivered_{c}" for c in CATS])
trans.insert(0, "v1_primary", CATS)
trans["total"] = T.sum(1)

per_verb_bank = []
for lem in POOL:
    fm, br = MAPPED["lemma_form"][lem]
    ff, _ = MAPPED["first_form"][lem]
    H1, P1 = V1[fm]
    H2, P2 = V2[lem]
    Hf, Pf = V1[ff]
    per_verb_bank.append(dict(
        lemma=lem, raw_forms=" ".join(o for _, o, _ in FORMS[lem]), v1_form=fm, v1_form_rule=br,
        v1_accepted=bool(H1), v1_primary=P1, v1_levels=" ".join(H1),
        delivered_accepted=bool(H2), delivered_primary=P2, delivered_levels=" ".join(H2),
        primary_distance=lvl_dist(P1, P2) if (H1 and H2) else "",
        v1_first_form=ff, v1_first_form_accepted=bool(Hf), v1_first_form_primary=Pf))

# ============================================================================ B. refits
with open(V1_META, encoding="utf-8") as fh:
    v1_meta = json.load(fh)
with open(os.path.join(PROD_DIR, "metadata.json"), encoding="utf-8") as fh:
    prod_meta = json.load(fh)
with open(os.path.join(PROD_DIR, "thresholds.json"), encoding="utf-8") as fh:
    prod_thr = np.asarray(json.load(fh)["thresholds_tuning"], float)


def name_and_percentile(meta):
    m = re.fullmatch(r"(\w+) \(P(\d+)\)", meta["winner_name"])
    check(m is not None, f"winner_name {meta['winner_name']!r}")
    return m.group(1), int(m.group(2))


V1_FAM, V1_P = name_and_percentile(v1_meta)
PROD_FAM, PROD_P = name_and_percentile(prod_meta)
check((V1_FAM, V1_P, PROD_FAM, PROD_P) == ("SGDClassifier", 60, "KNeighborsClassifier", 50),
      "archived original model is SGDClassifier P60 and the production model kNN P50")
V1_PARAMS = dict(v1_meta["winner_params"])
PROD_PARAMS = dict(prod_meta["winner_params"])
SGD1E4_PARAMS = dict(V1_PARAMS, alpha=1e-4)
GRID = {label: (base, grid) for label, base, grid in rt.MODEL_GRID}
for fam, params in ((V1_FAM, V1_PARAMS), (V1_FAM, SGD1E4_PARAMS), (PROD_FAM, PROD_PARAMS)):
    check(params in GRID[fam][1], f"{fam} {params} is a configuration of retrain.MODEL_GRID")
check(V1_PARAMS["alpha"] == 1e-5 and V1_PARAMS["loss"] == "hinge" and PROD_PARAMS["n_neighbors"] == 21,
      "configurations: SGD hinge alpha 1e-5 and kNN n_neighbors 21")
check(rt.TOP2_MARGIN_DEFAULT == 0.060 and rt.SUGGEST_ALPHA == 0.85 and rt.USE_OOF and rt.USE_ROW_WEIGHTS
      and rt.N_SPLITS_OOF == 5 and rt.RNG_SEED == 13, "production constants of retrain.py")
CORES = {"358": CORE358, "381": CORE381}

# embeddings: every word must be in all_embeddings.npz; the encoder is never loaded
def _no_encoder(*args, **kwargs):
    raise SystemExit("change_attribution.py: CHECK FAILED: a word is missing from all_embeddings.npz")


rt._get_st = _no_encoder
rt._load_npz_cache()
rt.EMBED_CACHE_DIR = os.path.join(REPO, "results", "_embed_cache")   # required by embed_batch; never read or written


def setup(core_csv):
    """Data part of retrain.run_pipeline: core labels, frequencies, vectors, row weights, targets."""
    core_words, Y, freq = rt.load_core_from_csv(core_csv)
    check(all(wd in rt._NPZ_CACHE for wd in core_words + POOL), f"{rel(core_csv)}: every vector is stored")
    rt.core_words, rt.Y_core, rt.core_freq = core_words, Y, freq
    rt.X_core = rt.embed_batch(core_words)
    rt.eval_verbs, rt.X_eval = list(POOL), rt.embed_batch(POOL)
    rt.ROW_W = rt._build_core_row_weights() if rt.USE_ROW_WEIGHTS else np.ones(rt.X_core.shape[0])
    T_ = rt.core_targets_from_labels(Y)
    lo, hi = T_["ACC_BAND"]
    lo = max(rt.ACC_BAND_CLAMP[0], min(lo, rt.ACC_BAND_CLAMP[1]))
    hi = max(lo + 1e-3, min(rt.ACC_BAND_CLAMP[1], hi))
    T_["ACC_BAND"] = (lo, hi)
    rt.TARGETS = T_
    return dict(n_core=len(core_words), n_assignments=int(Y.sum()))


def fit(fam, params):
    rt.ROW_WEIGHT_FIT.update(weighted_fit_ok=0, weighted_fit_fallback_unweighted=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S_core_raw, S_eval_raw, clf, used_oof = rt._get_scores(fam, GRID[fam][0], dict(params))
    check(used_oof, f"{fam}: out-of-fold calibration was used")
    cdfs = rt._fit_cdf_per_class(S_core_raw)
    return dict(S_core01=rt._apply_cdf_per_class(S_core_raw, cdfs), S_eval01=rt._apply_cdf_per_class(S_eval_raw, cdfs),
                row_weight_fit=dict(rt.ROW_WEIGHT_FIT))


def thresholds(f, p):
    w_ = rt.ROW_W if rt.USE_ROW_WEIGHTS else np.ones_like(rt.ROW_W)
    return rt.thresholds_from_percentile_weighted(f["S_core01"], rt.Y_core, p, w_)


def postprocess(S01, thr):
    Yb = (S01 >= thr.reshape(1, -1)).astype(int)
    Yb = rt._apply_top2_margin(S01, Yb, delta=rt.TOP2_MARGIN_DEFAULT)
    prim = rt._choose_primary(S01, Yb)
    Ys = (S01 >= (thr * float(rt.SUGGEST_ALPHA)).reshape(1, -1)).astype(int)
    return Yb, prim, Ys


FITS = [("SGD1e5", V1_FAM, V1_PARAMS), ("SGD1e4", V1_FAM, SGD1E4_PARAMS), ("kNN21", PROD_FAM, PROD_PARAMS)]
FIT_PLAN = [("358", "SGD1e5"), ("358", "SGD1e4"), ("358", "kNN21"), ("381", "SGD1e5"), ("381", "kNN21")]
MODELS = [  # id, fit, core, percentile, description
    ("SGD1e5_358_P60", "SGD1e5", "358", 60, "SGDClassifier, original configuration (hinge, alpha 1e-5, P60), original core; refit with current software"),
    ("SGD1e4_358_P50", "SGD1e4", "358", 50, "SGDClassifier hinge, alpha 1e-4, P50, original core"),
    ("kNN21_358_P50", "kNN21", "358", 50, "kNN-21 (production configuration, P50), original core"),
    ("SGD1e5_381_P55", "SGD1e5", "381", 55, "SGDClassifier, original configuration at P55 (its structural best on the corrected core), corrected core"),
    ("SGD1e5_381_P60", "SGD1e5", "381", 60, "SGDClassifier, original configuration (P60), corrected core"),
    ("kNN21_381_P50", "kNN21", "381", 50, "kNN-21 (production configuration, P50), corrected core: refit of the production model"),
]
PCTS_NEEDED = defaultdict(set)
for _, fk, core, p, _ in MODELS:
    PCTS_NEEDED[(core, fk)].add(p)
PCTS_NEEDED[("358", "SGD1e5")].add(55)        # 2 x 2 at P55

table5 = pd.read_csv(TABLE5, float_precision="round_trip")
RES = {}           # (core, fit, p) -> dict(thr, S_pool01, pool_acceptance)
FIT_INFO = {}
STRUCT = {}
for core, fk in FIT_PLAN:
    info = setup(CORES[core])
    fam, params = next((fam, params) for k, fam, params in FITS if k == fk)
    f = fit(fam, params)
    FIT_INFO[(core, fk)] = dict(info, row_weight_fit=f["row_weight_fit"])
    for p in sorted(PCTS_NEEDED[(core, fk)]):
        thr = thresholds(f, p)
        Yb, prim, Ys = postprocess(f["S_eval01"], thr)
        RES[(core, fk, p)] = dict(thr=thr, S01=f["S_eval01"], Yb=Yb, prim=prim, Ys=Ys,
                                  pool_acceptance=float((Yb.sum(1) > 0).mean()))
    if core == "381":
        # structural objective of the corrected-core refits, as tune_ovr_model computes it
        for p in sorted(PCTS_NEEDED[(core, fk)]):
            thr = RES[(core, fk, p)]["thr"]
            _, _, _, out_e, _, scs_e, _, parts_e = rt.evaluate_with_thresholds(
                f["S_core01"], f["S_eval01"], thr, rt.core_words, rt.eval_verbs)
            obj = rt._objective(scs_e - rt.SCS_EMPTY_LEVEL_PEN * parts_e["empty_levels"], out_e, rt.TARGETS)
            STRUCT[(fk, p)] = dict(PenalizedObj=float(obj), Eval_SCS=float(scs_e),
                                   acceptance=float(out_e["acc"]["acceptance"]), empty_levels=int(parts_e["empty_levels"]))
for (fk, p), label in ((("SGD1e5", 55), f"{V1_FAM} (P55)"), (("kNN21", 50), f"{PROD_FAM} (P50)")):
    row = table5[table5["Model"] == label]
    check(len(row) == 1, f"{label} in model_comparison_table5.csv")
    STRUCT[(fk, p)]["table5_PenalizedObj"] = float(row["PenalizedObj"].iloc[0])
    check(abs(STRUCT[(fk, p)]["PenalizedObj"] - float(row["PenalizedObj"].iloc[0])) <= 1e-9,
          f"{label}: penalized objective of the refit vs model_comparison_table5.csv")

# production refit vs the delivered bank and the production bundle
pr = RES[("381", "kNN21", 50)]
b2 = v2.set_index("k").loc[POOL]
cal_bank = b2[[f"cal_score_{l}" for l in LEVELS]].to_numpy(float)
pat = ["".join(str(int(x)) for x in r) for r in pr["Yb"]]
pat_bank = [str(x).split(".")[0].zfill(6) for x in b2["AcceptedPattern"]]
spat = ["".join(str(int(x)) for x in r) for r in pr["Ys"]]
spat_bank = [str(x).split(".")[0].zfill(6) for x in b2["SoftPattern"]]
PROD_CHECK = dict(
    max_abs_threshold_diff_vs_bundle=float(np.abs(pr["thr"] - prod_thr).max()),
    max_abs_calibrated_score_diff_vs_bank=float(np.abs(pr["S01"] - cal_bank).max()),
    same_hard_pattern=sum(a == c for a, c in zip(pat, pat_bank)),
    same_primary=sum(a == c for a, c in zip(pr["prim"], b2["PrimaryDomain"])),
    same_soft_pattern=sum(a == c for a, c in zip(spat, spat_bank)))
check(PROD_CHECK["max_abs_threshold_diff_vs_bundle"] <= 1e-12, "production refit: thresholds vs production_model/thresholds.json")
check(PROD_CHECK["max_abs_calibrated_score_diff_vs_bank"] <= 1e-6, "production refit: calibrated scores vs the bank (6 decimals)")
check(PROD_CHECK["same_hard_pattern"] == PROD_CHECK["same_primary"] == PROD_CHECK["same_soft_pattern"] == 1650,
      "production refit reproduces every hard pattern, primary level and soft pattern of the bank")

# ---------------------------------------------------------------- votes and scoring
votes = pd.read_excel(XLSX)
votes["clean"] = votes["Verb_clean"].astype(str).str.strip().str.lower()
check(len(votes) == 30 and votes["clean"].is_unique, "30 rated verbs")
VERBS30 = votes["clean"].tolist()
CNT = {r["clean"]: np.array([int(r[f"cnt_{l}"]) for l in LEVELS]) for _, r in votes.iterrows()}
NONE = {r["clean"]: int(r["cnt_None"]) for _, r in votes.iterrows()}


def emax(v):
    c = CNT[v]
    return {l for l, x in zip(LEVELS, c) if x == c.max() and x > 0}


def e25(v):
    return {l for l, x in zip(LEVELS, CNT[v]) if x >= 2}


POOL_IX = {wd: i for i, wd in enumerate(POOL)}
D30 = VERBS30
D29 = [v for v in D30 if v in POOL_IX]
check(len(D29) == 29 and [v for v in D30 if v not in POOL_IX] == ["copy"], "D29 = D30 minus 'copy'")


def bank_out(v):
    if v not in POOL_IX:
        return dict(status="missing", hard=set(), primary="")
    H, P = V2[v]
    return dict(status="accepted" if H else "rejected", hard=set(H), primary=P)


def refit_out(core, fk, p):
    r = RES[(core, fk, p)]
    out = {}
    for v in D30:
        if v not in POOL_IX:
            out[v] = dict(status="missing", hard=set(), primary="")
            continue
        i = POOL_IX[v]
        H = {LEVELS[j] for j in range(6) if r["Yb"][i, j]}
        out[v] = dict(status="accepted" if H else "rejected", hard=H, primary=r["prim"][i])
    return out


PROD = {v: bank_out(v) for v in D30}
C26 = [v for v in D29 if PROD[v]["status"] == "accepted"]
check(len(C26) == 26, "C26: the production model accepts 26 rated pool verbs")
OUTS = {mid: refit_out(core, fk, p) for mid, fk, core, p, _ in MODELS}
check(all(OUTS["kNN21_381_P50"][v] == PROD[v] for v in D30), "production refit equals the bank on the rated verbs")


def jpr(A, E):
    i = len(A & E)
    return (i / len(A | E) if A | E else 0.0, i / len(A) if A else 0.0, i / len(E) if E else 0.0)


def per_verb(o, v):
    r = o[v]
    if r["status"] != "accepted":
        return dict(strict=0, endorsed=0, J=0.0, P=0.0, R=0.0, cov=0, acc=0)
    J, P, R = jpr(r["hard"], e25(v))
    return dict(strict=int(r["primary"] in emax(v)), endorsed=int(r["primary"] in e25(v)), J=J, P=P, R=R,
                cov=int(bool(r["hard"] & e25(v))), acc=1)


def summary(o, S):
    pv = [per_verb(o, v) for v in S]
    acc = [x for x in pv if x["acc"]]
    return dict(N=len(S), accepted=sum(x["acc"] for x in pv), strict=sum(x["strict"] for x in pv),
                endorsed=sum(x["endorsed"] for x in pv), J=float(np.mean([x["J"] for x in pv])),
                P=float(np.mean([x["P"] for x in pv])), R=float(np.mean([x["R"] for x in pv])),
                P_accepted_only=float(np.mean([x["P"] for x in acc])) if acc else float("nan"),
                coverage=sum(x["cov"] for x in pv))


def mcnemar_exact(a, b):
    """Exact two-sided McNemar (binomial on discordant pairs). Returns (a only, b only, p)."""
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    t = a_only + b_only
    p = 1.0 if t == 0 else min(1.0, 2 * sum(math.comb(t, k) for k in range(min(a_only, b_only) + 1)) / 2 ** t)
    return a_only, b_only, p


def own(o):
    return [v for v in D29 if o[v]["status"] == "accepted"]


AGREE, MCN, MODEL_ROWS = [], [], []
for mid, fk, core, p, desc in MODELS:
    o = OUTS[mid]
    r = RES[(core, fk, p)]
    fam, params = next((fam, params) for k, fam, params in FITS if k == fk)
    rec = dict(model=mid, family=fam, params=json.dumps(params, sort_keys=True), core=f"{rel(CORES[core])} ({core} verbs)",
               percentile=p, description=desc, pool_acceptance=r["pool_acceptance"],
               fits_unweighted_fallback=FIT_INFO[(core, fk)]["row_weight_fit"]["weighted_fit_fallback_unweighted"],
               fits_weighted=FIT_INFO[(core, fk)]["row_weight_fit"]["weighted_fit_ok"])
    rec.update({f"thr_{l}": float(r["thr"][j]) for j, l in enumerate(LEVELS)})
    if core == "381":
        st = STRUCT[(fk, p)]
        rec.update(PenalizedObj=st["PenalizedObj"], Eval_SCS=st["Eval_SCS"], empty_levels=st["empty_levels"])
    MODEL_ROWS.append(rec)
    for sname, S in (("D30", D30), ("D29", D29), ("OWN", own(o)), ("C26", C26)):
        AGREE.append(dict(model=mid, set=sname, **summary(o, S)))
    if mid == "kNN21_381_P50":
        continue
    for sname, S in (("D29", D29), ("C26", C26)):
        for met in ("strict", "endorsed"):
            a = [per_verb(o, v)[met] for v in S]
            b = [per_verb(PROD, v)[met] for v in S]
            ao, bo, pv = mcnemar_exact(a, b)
            MCN.append(dict(model=mid, set=sname, metric=met, model_hits=sum(a), production_hits=sum(b),
                            model_only=ao, production_only=bo, p_exact=pv))
for sname, S in (("D30", D30), ("D29", D29), ("OWN", own(PROD)), ("C26", C26)):
    s_ref = summary(PROD, S)
    s_fit = next(x for x in AGREE if x["model"] == "kNN21_381_P50" and x["set"] == sname)
    check(all(s_fit[k] == s_ref[k] for k in s_ref if not (isinstance(s_ref[k], float) and math.isnan(s_ref[k]))),
          f"production refit summary equals the bank summary on {sname}")
agree = pd.DataFrame(AGREE)
mcn = pd.DataFrame(MCN)

# core x family 2 x 2
GRIDS = [("original_configuration", "SGDClassifier original configuration (alpha 1e-5, P60) vs kNN-21 (P50)", 60),
         ("sgd_structural_best", "SGDClassifier alpha 1e-5 at P55 (its structural best on the corrected core) vs kNN-21 (P50)", 55)]
TWO = []
for gid, glab, psgd in GRIDS:
    cells = {"SGD_358": refit_out("358", "SGD1e5", psgd), "SGD_381": refit_out("381", "SGD1e5", psgd),
             "kNN_358": refit_out("358", "kNN21", 50), "kNN_381": refit_out("381", "kNN21", 50)}
    for sname, S in (("D29", D29), ("C26", C26)):
        for met in ("strict", "endorsed"):
            a, b, c, d = (summary(cells[k], S)[met] for k in ("SGD_358", "SGD_381", "kNN_358", "kNN_381"))
            TWO.append(dict(grid=gid, description=glab, set=sname, metric=met, SGD_358=a, SGD_381=b, kNN_358=c, kNN_381=d,
                            family_effect=0.5 * ((c - a) + (d - b)), core_effect=0.5 * ((b - a) + (d - c)),
                            interaction=d - c - b + a, family_effect_at_358=c - a, family_effect_at_381=d - b,
                            core_effect_within_SGD=b - a, core_effect_within_kNN=d - c))
two = pd.DataFrame(TWO)

per_verb_rows = []
for v in D30:
    row = dict(verb=v, votes="/".join(str(x) for x in CNT[v]), none_votes=NONE[v],
               Emax=",".join(l for l in LEVELS if l in emax(v)), E25=",".join(l for l in LEVELS if l in e25(v)),
               in_C26=v in C26)
    for mid, *_ in MODELS:
        o = OUTS[mid][v]
        row[f"{mid}_primary"] = o["primary"] if o["status"] == "accepted" else o["status"]
        row[f"{mid}_hard"] = ",".join(l for l in LEVELS if l in o["hard"])
    per_verb_rows.append(row)

# ============================================================================ write
os.makedirs(OUT, exist_ok=True)
to_csv(pd.DataFrame(BANK_SUMMARY), "bank_change_summary.csv")
to_csv(pd.DataFrame(BANK_DIST), "bank_change_distance.csv")
to_csv(trans, "bank_change_transitions.csv")
to_csv(pd.DataFrame(per_verb_bank), "bank_change_per_verb.csv")
models_df = pd.DataFrame(MODEL_ROWS)
models_df["empty_levels"] = models_df["empty_levels"].astype("Int64")     # integer; empty for the original core
to_csv(models_df, "refit_models.csv")
to_csv(agree, "refit_agreement.csv")
to_csv(mcn, "refit_mcnemar_vs_production.csv")
to_csv(pd.DataFrame(per_verb_rows), "refit_per_verb.csv")
to_csv(two, "core_family_2x2.csv")

w("# Change from the original submission: verb bank and expert agreement")
w()
w("Generated by `revision_pipeline/change_attribution.py`. Do not edit by hand; rerun the script. "
  "Definitions, mapping rules and checks are in the script docstring. These are the numbers the response "
  "letter quotes that no other script produces; the like-for-like comparison of the archived original model "
  "with the production model is in `results/expert/` (Section 14 of `results/paper_tables.md`).")
w()
w("## 0. Reproduction checks")
w()
w(f"- Refit of the production configuration on the corrected core: thresholds equal `production_model/thresholds.json` "
  f"(max abs difference {PROD_CHECK['max_abs_threshold_diff_vs_bundle']:.1e}); calibrated scores equal the bank to its "
  f"6 decimals (max abs difference {PROD_CHECK['max_abs_calibrated_score_diff_vs_bank']:.2e}); same hard pattern "
  f"{PROD_CHECK['same_hard_pattern']}/1650, same primary level {PROD_CHECK['same_primary']}/1650, same soft pattern "
  f"{PROD_CHECK['same_soft_pattern']}/1650.")
for (fk, p), label in ((("SGD1e5", 55), f"{V1_FAM} alpha 1e-5 P55"), (("kNN21", 50), "kNN-21 P50")):
    st = STRUCT[(fk, p)]
    w(f"- {label} on the corrected core: penalized objective {st['PenalizedObj']:.6f} "
      f"(`results/model_comparison_table5.csv`: {st['table5_PenalizedObj']:.6f}), pool acceptance {st['acceptance']:.6f}.")
fb = sorted({(v_["row_weight_fit"]["weighted_fit_ok"], v_["row_weight_fit"]["weighted_fit_fallback_unweighted"])
             for v_ in FIT_INFO.values()})
w(f"- Row weights: in every fit the weighted one-vs-rest fit was rejected by scikit-learn and the model was fit "
  f"unweighted, as in the production run (weighted, unweighted fits per model: {', '.join(f'{a}, {b}' for a, b in fb)}).")
for core in ("358", "381"):
    inf = FIT_INFO[(core, "kNN21")]
    w(f"- Core `{rel(CORES[core])}`: {inf['n_core']} verbs, {inf['n_assignments']} verb-level assignments.")
w(f"- v1 bank `{rel(V1_BANK)}`: {len(V1):,} raw candidates, internally consistent; `{rel(NORM_AUDIT)}` has one row "
  f"per raw candidate and every one of the {len(POOL):,} pool lemmas keeps at least one raw form. "
  f"{len(CORE_JOINED)} further lemmas survive normalization but are verbs of the corrected core and not candidates "
  f"({', '.join(CORE_JOINED)}).")
w()

w("## 1. Bank change on the common candidate pool")
w()
w(f"The v1 labels are carried over to the {len(POOL):,} normalized candidates. {L_['lemmas_with_several_raw_forms']} "
  f"lemmas keep more than one raw form; under the lemma-form rule {L_['mapped_raw_form_equals_lemma']:,} lemmas take "
  f"the raw form equal to the lemma, {L_['mapped_first_kept_or_lemmatized']} the first kept or lemmatized form and "
  f"{L_['mapped_first_raw_form']} the first raw form. The two rules give different v1 rows for "
  f"{sum(MAPPED['lemma_form'][l][0] != MAPPED['first_form'][l][0] for l in POOL)} lemmas.")
w()
rows = []
for lab, key, kind in [("v1 bank on its own raw candidates: accepted", "v1_raw_accepted", "count_raw"),
                       ("v1 labels on the pool: accepted", "v1_accepted_on_pool", "count_pool"),
                       ("Delivered bank: accepted", "delivered_accepted", "count_pool"),
                       ("Same accept or reject decision", "same_decision", "count_pool"),
                       ("Accepted by both", "accepted_both", "count_pool"),
                       ("Accepted by v1 only", "accepted_v1_only", "count_pool"),
                       ("Accepted by the delivered bank only", "accepted_delivered_only", "count_pool"),
                       ("Rejected by both", "rejected_both", "count_pool"),
                       ("Same primary level (of accepted by both)", "same_primary", "count_both"),
                       ("Primary levels within one level (of accepted by both)", "within_one_level", "count_both"),
                       ("Mean distance between the primary levels", "mean_primary_distance", "float"),
                       ("Identical hard level set (of accepted by both)", "same_level_set", "count_both"),
                       ("Mean Jaccard index of the hard level sets (accepted by both)", "mean_level_set_jaccard", "float")]:
    cells = []
    for r in (L_, F_):
        if kind == "float":
            cells.append(f"{r[key]:.3f}")
        else:
            n = {"count_raw": r["v1_raw_inputs"], "count_pool": r["pool"], "count_both": r["accepted_both"]}[kind]
            cells.append(f"{r[key]:,}/{n:,} = {r[key] / n:.3f}")
    rows.append([lab] + cells)
table(["Quantity", "Lemma-form rule (reported)", "First-form rule (alternative)"], rows)
w("Distance between the two primary levels for the verbs accepted by both banks (positions in Kn, Cm, Ap, An, Sn, Ev):")
w()
rows = []
for d in range(6):
    a = next(x for x in BANK_DIST if x["rule"] == "lemma_form" and x["distance"] == d)
    b = next(x for x in BANK_DIST if x["rule"] == "first_form" and x["distance"] == d)
    rows.append([d, f"{a['verbs']} ({100 * a['share']:.1f}%)", f"{b['verbs']} ({100 * b['share']:.1f}%)"])
table(["Distance", "Lemma-form rule", "First-form rule"], rows)
w("Primary level in the v1 labels (rows) and in the delivered bank (columns), all 1,650 pool verbs, lemma-form rule "
  "(none = rejected):")
w()
table(["v1 \\ delivered"] + CATS + ["total"],
      [[CATS[i]] + [int(x) for x in T[i]] + [int(T[i].sum())] for i in range(7)]
      + [["total"] + [int(x) for x in T.sum(0)] + [int(T.sum())]])

w("## 2. Expert agreement of the refits on fixed sets of verbs")
w()
w("Each model is fitted with the production procedure on the core shown and scored on the rated verbs; a rejected or "
  "missing verb is a miss. The production model's outputs are read from the delivered bank; kNN21_381_P50 is its refit "
  "and reproduces it exactly. The refits use current software (scikit-learn 1.9.0); the archived original model itself "
  "is scored in `results/expert/like_for_like.csv`.")
w()
w("Models:")
w()
for mid, fk, core, p, desc in MODELS:
    w(f"- {mid}: {desc}.")
w()


def cellv(mid, sname, key):
    return next(x for x in AGREE if x["model"] == mid and x["set"] == sname)[key]


rows = []
for mid, *_ in MODELS:
    n_own = cellv(mid, "OWN", "N")
    rows.append([mid, f"{cellv(mid, 'D29', 'accepted')}/29"]
                + [pct(cellv(mid, s, "strict"), cellv(mid, s, "N")) for s in ("D30", "D29", "OWN", "C26")]
                + [pct(cellv(mid, s, "endorsed"), cellv(mid, s, "N")) for s in ("D30", "D29", "OWN", "C26")])
table(["Model", "Accepted of D29", "Strict D30", "Strict D29", "Strict OWN", "Strict C26",
       "Endorsed D30", "Endorsed D29", "Endorsed OWN", "Endorsed C26"], rows)
w("Hard-label Jaccard / precision / recall against the endorsed levels (misses = 0), precision over the accepted verbs "
  "only, and cover (at least one endorsed level among the hard labels):")
w()
rows = []
for mid, *_ in MODELS:
    rows.append([mid] + [f"{f4(cellv(mid, s, 'J'))} / {f4(cellv(mid, s, 'P'))} / {f4(cellv(mid, s, 'R'))}"
                         for s in ("D30", "D29", "OWN", "C26")]
                + [f4(cellv(mid, "D29", "P_accepted_only")), f"{cellv(mid, 'D29', 'coverage')}/29",
                   f"{cellv(mid, 'C26', 'coverage')}/26"])
table(["Model", "J / P / R D30", "J / P / R D29", "J / P / R OWN", "J / P / R C26", "P accepted only (D29)",
       "Cover D29", "Cover C26"], rows)
w("Paired exact McNemar test of each refit against the production model (hits; model only / production only):")
w()
rows = [[r["model"], r["set"], r["metric"], r["model_hits"], r["production_hits"],
         f"{r['model_only']}/{r['production_only']}", f4(r["p_exact"])] for r in MCN]
table(["Model", "Set", "Metric", "Model hits", "Production hits", "Model only / production only", "Exact p"], rows)
w("Primary level per rated verb (S = strict hit, E = endorsed but not strict, x = miss; rejected or missing as stated):")
w()
rows = []
for v in sorted(D30):
    cells = []
    for mid, *_ in MODELS:
        o = OUTS[mid][v]
        if o["status"] != "accepted":
            cells.append(o["status"])
        else:
            mark = "S" if o["primary"] in emax(v) else ("E" if o["primary"] in e25(v) else "x")
            cells.append(f"{o['primary']} ({mark})")
    rows.append([v, "/".join(str(x) for x in CNT[v]) + f"; {NONE[v]}", ",".join(l for l in LEVELS if l in emax(v)),
                 ",".join(l for l in LEVELS if l in e25(v))] + cells)
table(["Verb", "Votes Kn/Cm/Ap/An/Sn/Ev; None", "Emax", "E25"] + [m for m, *_ in MODELS], rows)

w("## 3. Training core x model family")
w()
w("Hits of each family fitted on each version of the training core. Shapley main effects average the change over the "
  "two levels of the other factor; the interaction is the difference of differences.")
w()
rows = [[r["description"], r["set"], r["metric"], r["SGD_358"], r["SGD_381"], r["kNN_358"], r["kNN_381"],
         f"{r['family_effect']:+.1f}", f"{r['core_effect']:+.1f}", f"{r['interaction']:+d}",
         f"{r['family_effect_at_358']:+d} / {r['family_effect_at_381']:+d}",
         f"{r['core_effect_within_SGD']:+d} / {r['core_effect_within_kNN']:+d}"] for r in TWO]
table(["Comparison", "Set", "Metric", "SGD, original core", "SGD, corrected core", "kNN, original core",
       "kNN, corrected core", "Family effect", "Core effect", "Interaction", "Family effect at original / corrected core",
       "Core effect within SGD / kNN"], rows)
lfl = pd.read_csv(LFL)
old26 = lfl[lfl["model"].str.startswith("OLD (archived") & (lfl["set"] == "C26")].iloc[0]
g1 = {(r["set"], r["metric"]): r for r in TWO if r["grid"] == "original_configuration"}
w("Strict / endorsed top-1 hits on the 26 verbs (C26), original configurations:")
w()
table(["Training core", "SGDClassifier, original configuration (hinge, alpha 1e-5, P60)", "kNN-21 (P50)"],
      [["Original (358 verbs)", f"{g1[('C26', 'strict')]['SGD_358']} / {g1[('C26', 'endorsed')]['SGD_358']}",
        f"{g1[('C26', 'strict')]['kNN_358']} / {g1[('C26', 'endorsed')]['kNN_358']}"],
       ["Corrected (381 verbs)", f"{g1[('C26', 'strict')]['SGD_381']} / {g1[('C26', 'endorsed')]['SGD_381']}",
        f"{g1[('C26', 'strict')]['kNN_381']} / {g1[('C26', 'endorsed')]['kNN_381']} (production model)"]])
w(f"The SGDClassifier refit on the original core uses current software; the archived original model scores "
  f"{int(old26['strict'])} / {int(old26['endorsed'])} on the same 26 verbs (`results/expert/like_for_like.csv`).")
w()

text = "\n".join(REPORT).rstrip("\n") + "\n"
check(not any(ch in text for ch in (chr(0x2013), chr(0x2014))), "no en or em dashes in the report")
with open(os.path.join(OUT, "change_attribution.md"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write(text)
print(text)
print("written to", rel(OUT))
