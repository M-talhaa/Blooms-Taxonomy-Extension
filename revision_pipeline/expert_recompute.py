"""Expert-agreement analysis of the production model (Section 3.4 of the paper).

Eight experts rated 30 verbs sampled from the original-submission (v1) bank. The vote matrix is
stored only in legacy/Final File(with model & Expert Classification).xlsx (per-verb counts
cnt_Kn ... cnt_Ev and cnt_None, each 0 to 8), keyed by the rated form (column Verb_clean). The
workbook also holds the original-submission model's outputs for the surface forms in its Verb
column ('copying' and 'resting' where the rated forms are 'copy' and 'rest').

Two models are scored:
  PROD  the production model (kNN, n = 21, cosine, distance-weighted, P50). Its outputs are read
        from the delivered bank results/Verb List Classified by Model.csv; the script also checks
        that the bundle in production_model/ reproduces the bank rows of the rated verbs.
  OLD   the archived original-submission model (SGDClassifier, P60), loaded from
        legacy/original-submission-model/ and run on the same normalized verb forms as PROD,
        with the production post-processing (calibrated scores, per-level thresholds, top-2
        margin collapse with delta 0.060, primary = highest calibrated score among hard labels).

Definitions (as in the paper):
  Emax(v) = levels with the top vote count (ties allowed, count > 0, None excluded)
  E25(v)  = levels endorsed by at least 2 of 8 experts
  strict top-1 = primary in Emax; endorsed top-1 = primary in E25
  hard Jaccard / precision / recall against E25, averaged over verbs
  gate reference: a verb should be rejected when None is the plurality response
Denominators:
  D30 all 30 rated verbs (normalized forms; 'copy' is a core training verb and not a candidate,
      so it is a miss for every model)
  D29 the 29 rated verbs in the 1,650-verb candidate pool
  C26 the 26 of those accepted by PROD (the set the paper's level-assignment metrics use)
On D30 and D29 a rejected or missing verb counts as a miss (hit 0, J = P = R = 0).

Outputs (results/expert/): expert_report.md and CSV tables. Read-only on every input.
Run from anywhere: python -I -B revision_pipeline/expert_recompute.py
Optional: set HF_HUB_OFFLINE=1; the locally cached all-mpnet-base-v2 is used only for the
two raw surface forms 'copying' and 'resting' (a reproduction check of the archived model);
the check is skipped if the encoder is not available.
"""
import json
import math
import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LEVELS = ["Kn", "Cm", "Ap", "An", "Sn", "Ev"]
N_RATERS = 8
TOP2_DELTA = 0.060          # production top-2 margin collapse (retrain.TOP2_MARGIN_DEFAULT)
SOFT_ALPHA = 0.85           # production soft tier (retrain.SUGGEST_ALPHA)
SEED = 13
N_PERM_MC = 100_000

XLSX = os.path.join(REPO, "legacy", "Final File(with model & Expert Classification).xlsx")
BANK = os.path.join(REPO, "results", "Verb List Classified by Model.csv")
EMB = os.path.join(HERE, "all_embeddings.npz")
CANDS = os.path.join(HERE, "candidates_final.txt")
CORE381 = os.path.join(HERE, "core_merged.csv")
CORE358 = os.path.join(HERE, "core_notebook_original.csv")
NORM_AUDIT = os.path.join(HERE, "normalization_audit.csv")
OLD_DIR = os.path.join(REPO, "legacy", "original-submission-model", "artifacts",
                       "SGDClassifier_P60_2025-10-12_01-47-29")
PROD_DIR = os.path.join(REPO, "production_model")
OUT = os.path.join(REPO, "results", "expert")
os.makedirs(OUT, exist_ok=True)

REPORT = []


def w(s=""):
    REPORT.append(s)


def pct(k, n):
    return f"{k}/{n} = {100.0 * k / n:.1f}%" if n else "n/a"


def f4(x):
    return f"{x:.4f}"


# ---------------------------------------------------------------- votes
votes = pd.read_excel(XLSX)
votes["clean"] = votes["Verb_clean"].astype(str).str.strip().str.lower()
votes["raw"] = votes["Verb"].astype(str).str.strip().str.lower()
assert len(votes) == 30 and votes["clean"].is_unique
VERBS30 = votes["clean"].tolist()
CNT = {r["clean"]: np.array([int(r[f"cnt_{l}"]) for l in LEVELS]) for _, r in votes.iterrows()}
NONE = {r["clean"]: int(r["cnt_None"]) for _, r in votes.iterrows()}
RAW = dict(zip(votes["clean"], votes["raw"]))


def emax(v):
    c = CNT[v]
    return {l for l, x in zip(LEVELS, c) if x == c.max() and x > 0}


def e25(v):
    return {l for l, x in zip(LEVELS, CNT[v]) if x >= 2}


def none_plurality(v):
    return NONE[v] > CNT[v].max()


POOL = [l.strip().lower() for l in open(CANDS, encoding="utf-8") if l.strip()]
assert len(POOL) == 1650
POOLSET = set(POOL)
D30 = VERBS30
D29 = [v for v in D30 if v in POOLSET]
assert len(D29) == 29, len(D29)

# ---------------------------------------------------------------- post-processing (production)


def apply_cdfs(S, cdfs):
    S = np.nan_to_num(np.asarray(S, float), nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros_like(S)
    for j, (xs, cdf) in enumerate(cdfs):
        out[:, j] = np.interp(S[:, j], xs, cdf, left=cdf[0], right=cdf[-1])
    return np.clip(out, 0.0, 1.0)


def top2_collapse(S01, Yb, delta=TOP2_DELTA):
    """retrain._apply_top2_margin: with >= 2 labels, keep only the top one if it leads by >= delta."""
    Yb = Yb.copy()
    for i in range(S01.shape[0]):
        idx = np.where(Yb[i] == 1)[0]
        if idx.size >= 2:
            s = S01[i, idx]
            o = np.argsort(s)[::-1]
            if S01[i, idx[o[0]]] - S01[i, idx[o[1]]] >= delta:
                Yb[i, :] = 0
                Yb[i, idx[o[0]]] = 1
    return Yb


def primary(S01, Yb):
    """retrain._choose_primary: argmax of the calibrated score among hard labels."""
    return ["" if Yb[i].sum() == 0 else LEVELS[int(np.argmax(S01[i] * Yb[i]))] for i in range(len(Yb))]


def postprocess(S01, thr, scale=1.0, collapse=True):
    Yb = (S01 >= thr.reshape(1, -1) * scale).astype(int)
    if collapse:
        Yb = top2_collapse(S01, Yb)
    return Yb, primary(S01, Yb)


def load_bundle(d):
    import joblib
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # the archived bundle was pickled with sklearn 1.2.2
        m = joblib.load(os.path.join(d, "model.pkl"))
    cdfs = [(np.asarray(x["xs"], float), np.asarray(x["cdf"], float))
            for x in json.load(open(os.path.join(d, "calibration_cdfs.json"), encoding="utf-8"))]
    thr = np.asarray(json.load(open(os.path.join(d, "thresholds.json"), encoding="utf-8"))["thresholds_tuning"], float)
    return m, cdfs, thr


def raw_scores(m, X):
    if hasattr(m, "decision_function"):
        return np.asarray(m.decision_function(X), float)
    P = m.predict_proba(X)
    if isinstance(P, list):
        return np.column_stack([p[:, 1] for p in P])
    return np.asarray(P, float)


_z = np.load(EMB)
VEC = {str(a): b for a, b in zip(_z["words"], _z["vectors"])}

# ---------------------------------------------------------------- PROD outputs from the bank
bank = pd.read_csv(BANK)
bank["verb"] = bank["Verb"].astype(str).str.strip().str.lower()
assert len(bank) == 1650 and set(bank["verb"]) == POOLSET
bank = bank.set_index("verb")
PROD_THR = bank[[f"thr_{l}" for l in LEVELS]].iloc[0].to_numpy(float)


def bank_out(v):
    if v not in bank.index:
        return dict(status="missing", hard=set(), primary="", soft=set(), S01=None)
    r = bank.loc[v]
    hard = {l for l in LEVELS if int(r[f"accept_{l}"]) == 1}
    soft_pat = str(r["SoftPattern"]).split(".")[0].zfill(6)
    soft = {l for l, c in zip(LEVELS, soft_pat) if c == "1"}
    prim = r["PrimaryDomain"] if isinstance(r["PrimaryDomain"], str) else ""
    return dict(status="accepted" if hard else "rejected", hard=hard, primary=prim, soft=soft,
                S01=r[[f"cal_score_{l}" for l in LEVELS]].to_numpy(float))


PROD = {v: bank_out(v) for v in D30}
C26 = [v for v in D29 if PROD[v]["status"] == "accepted"]
assert len(C26) == 26, len(C26)
SETS = {"D30": D30, "D29": D29, "C26": C26}

# check: the production bundle reproduces the bank rows of the rated verbs
pm, pcdf, pthr = load_bundle(PROD_DIR)
Xp = np.stack([VEC[v] for v in D29])
Sp = apply_cdfs(raw_scores(pm, Xp), pcdf)
Ybp, primp = postprocess(Sp, pthr)
prod_check = dict(
    max_abs_cal_diff=float(max(np.abs(Sp[i] - PROD[v]["S01"]).max() for i, v in enumerate(D29))),
    same_hard=sum({LEVELS[j] for j in range(6) if Ybp[i, j]} == PROD[v]["hard"] for i, v in enumerate(D29)),
    same_primary=sum(primp[i] == PROD[v]["primary"] for i, v in enumerate(D29)),
    thr_max_abs_diff=float(np.abs(np.round(pthr, 6) - PROD_THR).max()))

# ---------------------------------------------------------------- OLD (archived v1 model)
om, ocdf, othr = load_bundle(OLD_DIR)


def run_old(words, vecs):
    S = apply_cdfs(raw_scores(om, np.stack(vecs)), ocdf)
    Yb, prim = postprocess(S, othr)
    Ys = (S >= othr.reshape(1, -1) * SOFT_ALPHA).astype(int)
    return {wd: dict(hard={LEVELS[j] for j in range(6) if Yb[i, j]}, primary=prim[i],
                     soft={LEVELS[j] for j in range(6) if Ys[i, j]}, S01=S[i]) for i, wd in enumerate(words)}


_old29 = run_old(D29, [VEC[v] for v in D29])
OLD = {}
for v in D30:
    if v in _old29:
        o = _old29[v]
        OLD[v] = dict(status="accepted" if o["hard"] else "rejected", **o)
    else:
        OLD[v] = dict(status="missing", hard=set(), primary="", soft=set(), S01=None)

# v1 outputs as stored in the workbook (model run on the raw forms the v1 paper paired with the votes)
XL = {}
for _, r in votes.iterrows():
    pat = str(r["AcceptedPattern"]).split(".")[0].zfill(6)
    spat = str(r["SoftPattern"]).split(".")[0].zfill(6)
    hard = {l for l, c in zip(LEVELS, pat) if c == "1"}
    XL[r["clean"]] = dict(status="accepted" if hard else "rejected", hard=hard,
                          primary=r["PrimaryDomain"] if isinstance(r["PrimaryDomain"], str) else "",
                          soft={l for l, c in zip(LEVELS, spat) if c == "1"},
                          S01=np.array([float(r[f"cal_score_{l}"]) for l in LEVELS]))

# reproduction check of the archived model against the workbook (identical surface forms)
same_form = [v for v in D30 if RAW[v] == v]
_oldsf = run_old(same_form, [VEC[v] for v in same_form])
old_check = dict(n_same_form=len(same_form),
                 max_abs_cal_diff=float(max(np.abs(_oldsf[v]["S01"] - XL[v]["S01"]).max() for v in same_form)),
                 same_hard=sum(_oldsf[v]["hard"] == XL[v]["hard"] for v in same_form),
                 same_primary=sum(_oldsf[v]["primary"] == XL[v]["primary"] for v in same_form))
raw_forms = [RAW[v] for v in D30 if RAW[v] != v]
raw_check = None
try:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    probe = enc.encode(["rest", "deviate"], normalize_embeddings=True, convert_to_numpy=True)
    enc_ok = float(max(np.abs(probe[0] - VEC["rest"]).max(), np.abs(probe[1] - VEC["deviate"]).max()))
    R = enc.encode(raw_forms, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    _oldraw = run_old(raw_forms, list(R))
    inv = {RAW[v]: v for v in D30}
    raw_check = dict(encoder_vs_stored_max_abs=enc_ok, forms=raw_forms,
                     max_abs_cal_diff=float(max(np.abs(_oldraw[f]["S01"] - XL[inv[f]]["S01"]).max() for f in raw_forms)),
                     same_hard=sum(_oldraw[f]["hard"] == XL[inv[f]]["hard"] for f in raw_forms),
                     same_primary=sum(_oldraw[f]["primary"] == XL[inv[f]]["primary"] for f in raw_forms))
except Exception as exc:  # encoder not cached: the check is optional
    raw_check = dict(skipped=f"{type(exc).__name__}: {exc}"[:200])

# ---------------------------------------------------------------- metrics


def jpr(A, E):
    i = len(A & E)
    return (i / len(A | E) if A | E else 0.0, i / len(A) if A else 0.0, i / len(E) if E else 0.0)


def per_verb(o, v, soft=False):
    r = o[v]
    if r["status"] != "accepted":
        return dict(strict=0, endorsed=0, J=0.0, P=0.0, R=0.0, cov=0, acc=0)
    H = r["hard"] | r["soft"] if soft else r["hard"]
    J, P, Rr = jpr(H, e25(v))
    return dict(strict=int(r["primary"] in emax(v)), endorsed=int(r["primary"] in e25(v)),
                J=J, P=P, R=Rr, cov=int(bool(H & e25(v))), acc=1)


def summary(o, S, soft=False):
    pv = [per_verb(o, v, soft) for v in S]
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


def poisson_binomial_sf(probs, k):
    """Exact P(X >= k) for a sum of independent Bernoulli(p_i)."""
    d = np.zeros(len(probs) + 1)
    d[0] = 1.0
    for p in probs:
        d[1:] = d[1:] * (1 - p) + d[:-1] * p
        d[0] *= 1 - p
    return float(d[k:].sum())


def exact_perm_dist(H, labels):
    """Exact null distribution of sum_i H[i, label_i] when the multiset of labels is assigned to the
    rows uniformly at random (all permutations equally likely). Dynamic programming over rows with
    the remaining label counts as state. Returns P(hits = h), h = 0..n."""
    n = len(labels)
    cnt = np.bincount(labels, minlength=6)
    shape = tuple(int(c) + 1 for c in cnt) + (n + 1,)
    A = np.zeros(shape)
    A[tuple(int(c) for c in cnt) + (0,)] = 1.0
    for i in range(n):
        B = np.zeros(shape)
        for l in range(6):
            if cnt[l] == 0:
                continue
            src = [slice(None)] * 7
            dst = [slice(None)] * 7
            src[l], dst[l] = slice(1, None), slice(0, -1)
            if H[i, l]:
                src[6], dst[6] = slice(0, -1), slice(1, None)
            B[tuple(dst)] += A[tuple(src)]
        A = B
    d = A[(0,) * 6]
    return d / d.sum()


def label_permutation(o, S):
    """Shuffle the model's primaries across the verbs it accepts within S, keeping its own level
    distribution; rejected or missing verbs stay misses. Exact p (full permutation distribution)
    plus a Monte Carlo cross-check."""
    A = [v for v in S if o[v]["status"] == "accepted"]
    lab = np.array([LEVELS.index(o[v]["primary"]) for v in A])
    res = {}
    for name, setfun in (("endorsed", e25), ("strict", emax)):
        H = np.array([[l in setfun(v) for l in LEVELS] for v in A])
        obs = int(H[np.arange(len(A)), lab].sum())
        dist = exact_perm_dist(H, lab)
        rng = np.random.default_rng(SEED)
        perm = np.argsort(rng.random((N_PERM_MC, len(A))), axis=1)
        mc = H[np.arange(len(A))[None, :], lab[perm]].sum(1)
        res[name] = dict(observed=obs, null_mean=float((np.arange(len(A) + 1) * dist).sum()),
                         p_exact=float(dist[obs:].sum()), p_mc=float((mc >= obs).mean()))
    res["n_accepted"] = len(A)
    res["level_mix"] = {l: int((lab == i).sum()) for i, l in enumerate(LEVELS)}
    return res


core381 = pd.read_csv(CORE381)
FR381 = core381["level"].value_counts(normalize=True).reindex(LEVELS).to_numpy()

# ---------------------------------------------------------------- 1. production metrics (paper Section 3.4)
pv26 = summary(PROD, C26)
pv26s = summary(PROD, C26, soft=True)
cov_h = [per_verb(PROD, v)["cov"] for v in C26]
cov_s = [per_verb(PROD, v, True)["cov"] for v in C26]
mc_hs = mcnemar_exact(cov_h, cov_s)
pmax = {v: CNT[v].max() / N_RATERS for v in C26}
HI = [v for v in C26 if pmax[v] >= 0.5]
LO = [v for v in C26 if pmax[v] < 0.5]
hi_hits = sum(PROD[v]["primary"] in e25(v) for v in HI)
lo_hits = sum(PROD[v]["primary"] in e25(v) for v in LO)
fisher_band = stats.fisher_exact([[hi_hits, len(HI) - hi_hits], [lo_hits, len(LO) - lo_hits]]).pvalue
ch_hi = [len(e25(v)) / 6 for v in HI]
p_hi_exact = poisson_binomial_sf(ch_hi, hi_hits)
ch26 = [len(e25(v)) / 6 for v in C26]
p26_exact = poisson_binomial_sf(ch26, pv26["endorsed"])


def gate(o):
    ca = cr = fa = fr = 0
    for v in D29:
        a = o[v]["status"] == "accepted"
        nd = none_plurality(v)
        ca += a and not nd
        fa += a and nd
        cr += (not a) and nd
        fr += (not a) and not nd
    p = stats.fisher_exact([[ca, fa], [fr, cr]]).pvalue
    return dict(correct_accept=ca, correct_reject=cr, false_accept=fa, false_reject=fr,
                accuracy=(ca + cr) / len(D29), fisher_p=float(p))


g_prod, g_old = gate(PROD), gate(OLD)

# ---------------------------------------------------------------- 2. score-vote Spearman


def rank_z(M):
    R = np.apply_along_axis(stats.rankdata, 1, np.asarray(M, float))
    R = R - R.mean(1, keepdims=True)
    nrm = np.linalg.norm(R, axis=1, keepdims=True)
    return np.divide(R, nrm, out=np.zeros_like(R), where=nrm > 0), nrm.ravel() > 0


def score_vote(o, S, B=N_PERM_MC):
    Smat = np.stack([o[v]["S01"] for v in S])
    Vmat = np.stack([CNT[v] for v in S]).astype(float)
    zs, oks = rank_z(Smat)
    zv, okv = rank_z(Vmat)
    keep = okv
    obs_each = (zs * zv).sum(1)
    chk = [stats.spearmanr(Smat[i], Vmat[i]).statistic for i in range(len(S)) if keep[i] and oks[i]]
    obs = float(obs_each[keep & oks].mean())
    assert abs(obs - float(np.mean(chk))) < 1e-12
    rng = np.random.default_rng(SEED)
    perm = np.argsort(rng.random((B, len(S))), axis=1)
    null = ((zs[perm] * zv[None, :, :]).sum(2)[:, keep]).mean(1)
    k = int((null >= obs - 1e-12).sum())
    return dict(n=len(S), n_used=int(keep.sum()), mean_rho=obs, p_mc=(k + 1) / (B + 1), B=B)


sv_prod26 = score_vote(PROD, C26)
sv_old26 = score_vote(OLD, C26)
sv_xl30 = score_vote(XL, D30)

# ---------------------------------------------------------------- 3. like-for-like OLD vs PROD
LFL = []
for sname, S in SETS.items():
    for mname, o in (("OLD (archived v1 model, normalized forms)", OLD), ("PROD (kNN-21, P50)", PROD)):
        s = summary(o, S)
        LFL.append(dict(model=mname, set=sname, N=s["N"], accepted=s["accepted"], strict=s["strict"],
                        endorsed=s["endorsed"], J=s["J"], P=s["P"], R=s["R"],
                        P_accepted_only=s["P_accepted_only"], coverage=s["coverage"]))
for S, sname in ((D30, "D30 v1 as reported (raw forms copying, resting)"),):
    s = summary(XL, S)
    ss = summary(XL, S, soft=True)
    LFL.append(dict(model="OLD as stored in the workbook (v1 paper)", set=sname, N=s["N"], accepted=s["accepted"],
                    strict=s["strict"], endorsed=s["endorsed"], J=s["J"], P=s["P"], R=s["R"],
                    P_accepted_only=s["P_accepted_only"], coverage=s["coverage"]))
lfl = pd.DataFrame(LFL)
lfl.to_csv(os.path.join(OUT, "like_for_like.csv"), index=False, float_format="%.4f")
xl30s = summary(XL, D30, soft=True)

MCN = []
for sname, S in SETS.items():
    for met in ("strict", "endorsed"):
        a = [per_verb(OLD, v)[met] for v in S]
        b = [per_verb(PROD, v)[met] for v in S]
        ao, bo, p = mcnemar_exact(a, b)
        MCN.append(dict(set=sname, metric=met, OLD_hits=sum(a), PROD_hits=sum(b), OLD_only=ao, PROD_only=bo, p_exact=p))
mcn = pd.DataFrame(MCN)
mcn.to_csv(os.path.join(OUT, "mcnemar_old_vs_prod.csv"), index=False, float_format="%.4f")

# ---------------------------------------------------------------- 4. baselines and permutation tests
BASE = []
for sname, S in SETS.items():
    pe_u = [len(e25(v)) / 6 for v in S]
    ps_u = [len(emax(v)) / 6 for v in S]
    pe_f = [sum(FR381[j] for j, l in enumerate(LEVELS) if l in e25(v)) for v in S]
    ps_f = [sum(FR381[j] for j, l in enumerate(LEVELS) if l in emax(v)) for v in S]
    ap_e = [int("Ap" in e25(v)) for v in S]
    ap_s = [int("Ap" in emax(v)) for v in S]
    for mname, o in (("OLD", OLD), ("PROD", PROD)):
        pv = [per_verb(o, v) for v in S]
        he = [x["endorsed"] for x in pv]
        hs = [x["strict"] for x in pv]
        pt = label_permutation(o, S)
        for met, h, pu, pf, ap in (("endorsed", he, pe_u, pe_f, ap_e), ("strict", hs, ps_u, ps_f, ap_s)):
            mo, ao, pmc = mcnemar_exact(h, ap)
            BASE.append(dict(model=mname, set=sname, metric=met, N=len(S), hits=sum(h),
                             uniform_expected=sum(pu), p_uniform_exact=poisson_binomial_sf(pu, sum(h)),
                             corefreq_expected=sum(pf), p_corefreq_exact=poisson_binomial_sf(pf, sum(h)),
                             always_Ap_hits=sum(ap), model_only_vs_Ap=mo, Ap_only=ao, p_mcnemar_vs_Ap=pmc,
                             perm_n_accepted=pt["n_accepted"], perm_null_mean=pt[met]["null_mean"],
                             p_perm_exact=pt[met]["p_exact"], p_perm_mc=pt[met]["p_mc"]))
base = pd.DataFrame(BASE)
base.to_csv(os.path.join(OUT, "baselines_and_permutation.csv"), index=False, float_format="%.4f")

# ---------------------------------------------------------------- 5. Fleiss kappa and rater subsampling
CATS = LEVELS + ["None"]
K7 = np.array([[CNT[v][j] for j in range(6)] + [NONE[v]] for v in D30])


def fleiss_kappa_binary(k, n):
    """Fleiss kappa for one binary category from per-item positive-vote counts k out of n raters."""
    k = np.asarray(k, float)
    p_obs = ((k * (k - 1) + (n - k) * (n - k - 1)) / (n * (n - 1))).mean()
    prev = k.sum() / (k.size * n)
    p_exp = prev ** 2 + (1 - prev) ** 2
    kap = np.nan if p_exp == 1.0 else (p_obs - p_exp) / (1 - p_exp)
    return kap, p_obs, prev


KAP = []
for j, c in enumerate(CATS):
    kap, po, prev = fleiss_kappa_binary(K7[:, j], N_RATERS)
    KAP.append(dict(category=c, prevalence=prev, observed_agreement=po, fleiss_kappa=kap))
kap = pd.DataFrame(KAP)
kap.to_csv(os.path.join(OUT, "fleiss_kappa.csv"), index=False, float_format="%.4f")
lev_k = kap[kap["category"].isin(LEVELS)]["fleiss_kappa"]

K6 = K7[:, :6]
rng = np.random.default_rng(SEED)          # same generator and order as the panel-size curve in BT 5, Section 5
CURVE = []
for j in range(2, N_RATERS + 1):
    vals = np.empty(3000)
    for i in range(3000):
        sub = rng.hypergeometric(K6, N_RATERS - K6, j)
        vals[i] = np.nanmean([fleiss_kappa_binary(sub[:, l], j)[0] for l in range(6)])
    CURVE.append(dict(panel_size=j, mean_kappa=vals.mean(), pct_2_5=np.percentile(vals, 2.5),
                      pct_97_5=np.percentile(vals, 97.5)))
curve = pd.DataFrame(CURVE)
curve.to_csv(os.path.join(OUT, "rater_subsampling.csv"), index=False, float_format="%.4f")

# ---------------------------------------------------------------- 6. threshold sweep, after and before the top-2 collapse
Sall = bank[[f"cal_score_{l}" for l in LEVELS]].to_numpy(float)
Sr = np.stack([PROD[v]["S01"] for v in D29])
E29 = np.array([[l in e25(v) for l in LEVELS] for v in D29])
M29 = np.array([[l in emax(v) for l in LEVELS] for v in D29])
SW = []
for scale in (1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70):
    for collapse in (True, False):
        Yall, _ = postprocess(Sall, PROD_THR, scale, collapse)
        acc_all = Yall.sum(1) > 0
        Y, prim = postprocess(Sr, PROD_THR, scale, collapse)
        accd = Y.sum(1) > 0
        inter = (Y & E29).sum(1)
        nY = Y.sum(1)
        P_acc = (inter[accd] / nY[accd]).mean()
        R29 = (inter / E29.sum(1)).mean()
        J29 = (inter / np.maximum((Y | E29).sum(1), 1)).mean()
        endo = sum(1 for i, p in enumerate(prim) if p and E29[i, LEVELS.index(p)])
        stri = sum(1 for i, p in enumerate(prim) if p and M29[i, LEVELS.index(p)])
        SW.append(dict(scale=scale, counting="after top-2 collapse" if collapse else "before collapse",
                       pool_acceptance=acc_all.mean(), levels_per_accepted_verb=Yall[acc_all].sum() / acc_all.sum(),
                       rated_accepted_of_29=int(accd.sum()), endorsed_top1_of_29=endo, strict_top1_of_29=stri,
                       precision_accepted_only=P_acc, precision_misses_0=(np.where(accd, inter / np.maximum(nY, 1), 0)).mean(),
                       recall_of_29=R29, jaccard_of_29=J29,
                       precision_v2_style=P_acc, recall_v2_style=(inter[accd] / E29.sum(1)[accd]).mean()))
sweep = pd.DataFrame(SW)
sweep.to_csv(os.path.join(OUT, "threshold_sweep.csv"), index=False, float_format="%.4f")

# ---------------------------------------------------------------- per-verb table
norm_audit = pd.read_csv(NORM_AUDIT)
core358 = set(pd.read_csv(CORE358)["verb"].str.lower())
core381 = set(core381["verb"].str.lower())
PV = []
for v in D30:
    def fmt(o):
        r = o[v]
        if r["status"] != "accepted":
            return r["status"], "", ""
        return "accepted", r["primary"], " ".join(l for l in LEVELS if l in r["hard"])
    ps, pp, ph = fmt(PROD)
    os_, op, oh = fmt(OLD)
    xs, xp, xh = fmt(XL)
    PV.append(dict(verb=v, rated_form=RAW[v], votes="/".join(str(x) for x in CNT[v]), none_votes=NONE[v],
                   Emax=" ".join(l for l in LEVELS if l in emax(v)), E25=" ".join(l for l in LEVELS if l in e25(v)),
                   none_plurality=none_plurality(v), in_pool=v in POOLSET, in_C26=v in C26,
                   PROD_status=ps, PROD_primary=pp, PROD_hard=ph,
                   PROD_strict=per_verb(PROD, v)["strict"], PROD_endorsed=per_verb(PROD, v)["endorsed"],
                   OLD_status=os_, OLD_primary=op, OLD_hard=oh,
                   OLD_strict=per_verb(OLD, v)["strict"], OLD_endorsed=per_verb(OLD, v)["endorsed"],
                   v1_stored_primary=xp, v1_stored_hard=xh))
pd.DataFrame(PV).to_csv(os.path.join(OUT, "per_verb.csv"), index=False)

# ---------------------------------------------------------------- report
w("# Expert agreement: production model, like-for-like comparison and reliability")
w()
w("Generated by `revision_pipeline/expert_recompute.py`. Do not edit by hand; rerun the script.")
w()
w("Inputs: votes `legacy/Final File(with model & Expert Classification).xlsx` (8 raters x 30 verbs); "
  "production outputs `results/Verb List Classified by Model.csv`; archived original-submission (v1) model "
  "`legacy/original-submission-model/`; candidate pool `revision_pipeline/candidates_final.txt`. "
  "Definitions and denominators are in the script docstring. Exact tests unless marked Monte Carlo "
  f"(MC, numpy default_rng({SEED}), {N_PERM_MC:,} draws, p = (k + 1)/(B + 1)).")
w()
w("## 0. Reproduction checks")
w()
w(f"- Production bundle `production_model/` on the 29 rated pool verbs vs the bank: max abs calibrated-score "
  f"difference {prod_check['max_abs_cal_diff']:.2e}, same hard set {prod_check['same_hard']}/29, same primary "
  f"{prod_check['same_primary']}/29, max abs threshold difference {prod_check['thr_max_abs_diff']:.1e}.")
w(f"- Archived original-submission model on the {old_check['n_same_form']} rated verbs whose rated form equals the "
  f"stored form vs the workbook: max abs calibrated-score difference {old_check['max_abs_cal_diff']:.2e}, same hard "
  f"set {old_check['same_hard']}/{old_check['n_same_form']}, same primary {old_check['same_primary']}/{old_check['n_same_form']}.")
if raw_check and "skipped" not in raw_check:
    w(f"- Raw forms {raw_check['forms']} encoded with the cached all-mpnet-base-v2 (encoder vs stored vectors: max abs "
      f"{raw_check['encoder_vs_stored_max_abs']:.1e}): max abs calibrated-score difference {raw_check['max_abs_cal_diff']:.2e}, "
      f"same hard set {raw_check['same_hard']}/2, same primary {raw_check['same_primary']}/2.")
else:
    w(f"- Raw-form check skipped ({raw_check.get('skipped')}).")
w(f"- Sets: D30 = 30 rated verbs; D29 = in the candidate pool (D30 minus 'copy'); C26 = accepted by the production model. "
  f"D29 not in C26: {[v for v in D29 if v not in C26]}.")
w()
w("## 1. Production model, Section 3.4 figures (C26 unless stated)")
w()
w("| Statistic | Value |")
w("|---|---|")
w(f"| Strict top-1 | {pct(pv26['strict'], 26)} |")
w(f"| Endorsed top-1 | {pct(pv26['endorsed'], 26)}; uniform chance mean {np.mean(ch26):.1%}, exact Poisson-binomial p = {f4(p26_exact)} |")
w(f"| Hard Jaccard / precision / recall | {pv26['J']:.2f} / {pv26['P']:.2f} / {pv26['R']:.2f} ({pv26['J']:.4f} / {pv26['P']:.4f} / {pv26['R']:.4f}) |")
w(f"| Hard + soft Jaccard / precision / recall | {pv26s['J']:.2f} / {pv26s['P']:.2f} / {pv26s['R']:.2f} |")
w(f"| Coverage (at least one endorsed level): hard, hard + soft | {sum(cov_h)}/26, {sum(cov_s)}/26 (McNemar discordant {mc_hs[0]}/{mc_hs[1]}, p = {mc_hs[2]:.4f}) |")
w(f"| High-consensus verbs (at least 4 of 8 on one level), endorsed | {hi_hits}/{len(HI)} = {hi_hits/len(HI):.0%}; conditional chance {np.mean(ch_hi):.0%}, exact p = {f4(p_hi_exact)} |")
w(f"| Low-consensus verbs, endorsed | {lo_hits}/{len(LO)} = {lo_hits/len(LO):.0%} |")
w(f"| Band gap, Fisher exact (two-sided) | p = {f4(fisher_band)} |")
w(f"| Score-vote Spearman (mean per verb, 6 scores vs 6 vote counts) | rho = {sv_prod26['mean_rho']:.3f}, permutation p (MC) = {f4(sv_prod26['p_mc'])} |")
w(f"| Gate on D29 (None-plurality reference): CA / CR / FA / FR | {g_prod['correct_accept']} / {g_prod['correct_reject']} / {g_prod['false_accept']} / {g_prod['false_reject']}, accuracy {g_prod['correct_accept'] + g_prod['correct_reject']}/29 = {g_prod['accuracy']:.1%}, Fisher exact p = {f4(g_prod['fisher_p'])} |")
w()
w("The endorsed and high-consensus chance tests are exact (Poisson-binomial tails), and the score-vote Spearman p "
  f"is from {N_PERM_MC:,} Monte Carlo draws.")
w()
w("## 2. Like-for-like: archived original-submission model vs production model")
w()
w("Both models are scored on the same normalized verb forms with the same definitions. Rejected or missing verbs are misses.")
w()
w("| Model | Set | Accepted | Strict | Endorsed | J / P / R (misses 0) | P accepted only | Coverage |")
w("|---|---|---|---|---|---|---|---|")
for r in LFL:
    w(f"| {r['model']} | {r['set']} | {r['accepted']}/{r['N']} | {pct(r['strict'], r['N'])} | {pct(r['endorsed'], r['N'])} | "
      f"{r['J']:.4f} / {r['P']:.4f} / {r['R']:.4f} | {r['P_accepted_only']:.4f} | {r['coverage']}/{r['N']} |")
w()
w(f"v1 as stored, hard + soft J / P / R on 30: {xl30s['J']:.2f} / {xl30s['P']:.2f} / {xl30s['R']:.2f}; coverage hard "
  f"{summary(XL, D30)['coverage']}/30, hard + soft {xl30s['coverage']}/30. Score-vote Spearman as stored on 30: rho = "
  f"{sv_xl30['mean_rho']:.3f}, MC p = {f4(sv_xl30['p_mc'])} (v1 printed 0.144, p = 0.072).")
w()
w("Exact paired McNemar, OLD vs PROD:")
w()
w("| Set | Metric | OLD hits | PROD hits | OLD only | PROD only | p |")
w("|---|---|---|---|---|---|---|")
for r in MCN:
    w(f"| {r['set']} | {r['metric']} | {r['OLD_hits']} | {r['PROD_hits']} | {r['OLD_only']} | {r['PROD_only']} | {f4(r['p_exact'])} |")
w()
w(f"Gate on D29: OLD {g_old['correct_accept']}/{g_old['correct_reject']}/{g_old['false_accept']}/{g_old['false_reject']} "
  f"(accuracy {g_old['correct_accept'] + g_old['correct_reject']}/29 = {g_old['accuracy']:.1%}; it accepts every rated verb, "
  f"because the sample was drawn from its accepted verbs, so this is the accept-everything baseline), PROD "
  f"{g_prod['correct_accept']}/{g_prod['correct_reject']}/{g_prod['false_accept']}/{g_prod['false_reject']} ({g_prod['accuracy']:.1%}).")
w()
w(f"Score-vote Spearman on the common C26: OLD rho = {sv_old26['mean_rho']:.3f} (MC p = {f4(sv_old26['p_mc'])}), "
  f"PROD rho = {sv_prod26['mean_rho']:.3f} (MC p = {f4(sv_prod26['p_mc'])}).")
w()
w("## 3. Baselines and label-permutation test")
w()
w("Uniform: one level drawn uniformly from six. Core frequency: one level drawn with the 381-verb core shares of "
  f"verb-level assignments ({', '.join(f'{l} {x:.4f}' for l, x in zip(LEVELS, FR381))}). Always-Ap: constant Application. "
  "Guessers answer every verb. Uniform and core-frequency p-values are exact Poisson-binomial tails P(X >= hits); "
  "the always-Ap comparison is an exact paired McNemar. The label-permutation test shuffles the model's own primaries "
  "across the verbs it accepts in the set (its level mix is kept); p is exact, from the full permutation distribution "
  "(Monte Carlo cross-check in the CSV).")
w()
w("| Model | Set | Metric | Hits | Uniform exp., p | Core-freq exp., p | Always-Ap hits; model only / Ap only, p | Perm null mean, exact p |")
w("|---|---|---|---|---|---|---|---|")
for r in BASE:
    w(f"| {r['model']} | {r['set']} | {r['metric']} | {r['hits']}/{r['N']} | {r['uniform_expected']:.2f}, {f4(r['p_uniform_exact'])} | "
      f"{r['corefreq_expected']:.2f}, {f4(r['p_corefreq_exact'])} | {r['always_Ap_hits']}; {r['model_only_vs_Ap']}/{r['Ap_only']}, {f4(r['p_mcnemar_vs_Ap'])} | "
      f"{r['perm_null_mean']:.2f}, {f4(r['p_perm_exact'])} |")
w()
w("The permutation test shuffles labels among accepted verbs only, so its p-value is the same on D30, D29 and C26 for "
  "the production model (it accepts exactly C26) and on D29 and D30 for OLD.")
w()
w("## 4. Reliability: per-level Fleiss kappa and rater subsampling")
w()
w("| Category | Prevalence | Observed pairwise agreement | Fleiss kappa |")
w("|---|---|---|---|")
for r in KAP:
    w(f"| {r['category']} | {r['prevalence']:.2f} | {r['observed_agreement']:.2f} | {r['fleiss_kappa']:.3f} |")
w()
w(f"Range over the six levels: {lev_k.min():.3f} to {lev_k.max():.3f}, i.e. {lev_k.min():.2f} to {lev_k.max():.2f} "
  "(Comprehension is negative).")
w()
w(f"Rater subsampling (hypergeometric draw of j of the 8 votes per verb and level, 3,000 panels per size, default_rng({SEED}), "
  "mean of the six per-level kappas; same procedure and generator as the panel-size curve in Section 5 of "
  "`BT 5 - Revision analyses.ipynb`):")
w()
w("| Panel size | Mean kappa | 2.5th pct | 97.5th pct |")
w("|---|---|---|---|")
for r in CURVE:
    w(f"| {r['panel_size']} | {r['mean_kappa']:.3f} | {r['pct_2_5']:.3f} | {r['pct_97_5']:.3f} |")
w()
w("## 5. Threshold sweep of the production model on a fixed 29-verb denominator")
w()
w("Per-level hard thresholds scaled by s; a level is an entry when the calibrated score is at least s x threshold. "
  "'After top-2 collapse' applies the production collapse (delta 0.060) to the entries, so s = 1.00 reproduces the "
  "delivered bank. Precision is averaged over the rated verbs accepted at that scale; recall and Jaccard are averaged "
  "over all 29 rated pool verbs with rejected verbs counted as 0. Pool acceptance and levels per accepted verb are on all "
  "1,650 candidates. The 'before collapse' rows are the sweep reported in Section 3.5 of the article. The *_v2_style "
  "columns of the CSV (v2-style P / R in the table) average both precision and recall over the verbs accepted at each "
  "scale, as the first revision of the article (v2) did. This is a description of operating "
  "characteristics, not a tuning procedure.")
w()
w("| Scale | Counting | Pool acceptance | Levels per accepted verb | Rated accepted | Endorsed /29 | Strict /29 | Precision (accepted) | Recall /29 | Jaccard /29 | v2-style P / R |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
for r in SW:
    w(f"| {r['scale']:.2f} | {r['counting']} | {r['pool_acceptance']:.3f} | {r['levels_per_accepted_verb']:.2f} | {r['rated_accepted_of_29']}/29 | "
      f"{r['endorsed_top1_of_29']} | {r['strict_top1_of_29']} | {r['precision_accepted_only']:.3f} | {r['recall_of_29']:.3f} | {r['jaccard_of_29']:.3f} | "
      f"{r['precision_v2_style']:.2f} / {r['recall_v2_style']:.2f} |")
w()
w("The production soft tier (0.85 x threshold) is reported without the collapse; the 0.85 'after top-2 collapse' row is "
  "therefore not the soft tier.")
w()
w("## 6. Pairing of 'rest' and 'copy' in the original submission")
w()
na = norm_audit.set_index("original")
for v in ("rest", "copy"):
    raw = RAW[v]
    act = na.loc[raw, "action"] if raw in na.index else "not in audit"
    x = XL[v]
    o = OLD[v]
    w(f"- '{v}': the votes are recorded for '{v}' (workbook column Verb_clean), but the v1 table paired them with the model output for '{raw}' (column Verb) "
      f"(normalization audit: {raw} -> {na.loc[raw, 'normalized_lemma'] if raw in na.index else '?'}, {act}). Stored v1 output: "
      f"primary {x['primary'] or 'none'}, hard {sorted(x['hard'])}. "
      + (f"The archived model on '{v}': primary {o['primary'] or 'none'}, hard {sorted(o['hard'])} "
         f"(strict {per_verb(OLD, v)['strict']}, endorsed {per_verb(OLD, v)['endorsed']})."
         if o["status"] != "missing" else
         f"'{v}' is a core training verb (in the 358-verb core: {v in core358}; in the 381-verb core: {v in core381}), "
         "so it is not a candidate and counts as a miss for every model on D30."))
x_s, x_e = summary(XL, D30)["strict"], summary(XL, D30)["endorsed"]
rest_fix = {kk: per_verb(OLD, "rest")[kk] - (XL["rest"]["primary"] in (emax("rest") if kk == "strict" else e25("rest")))
            for kk in ("strict", "endorsed")}
w(f"- Effect: v1 as reported {x_s}/30 strict and {x_e}/30 endorsed. Scoring the rated form 'rest' with the same model "
  f"changes these by {rest_fix['strict']:+d} and {rest_fix['endorsed']:+d} ({x_s + rest_fix['strict']}/30, "
  f"{x_e + rest_fix['endorsed']}/30). On normalized forms with 'copy' counted as a miss, OLD scores "
  f"{summary(OLD, D30)['strict']}/30 and {summary(OLD, D30)['endorsed']}/30 (Section 2). The v1 sample therefore "
  "contained one core-verb variant ('copying').")
w()
open(os.path.join(OUT, "expert_report.md"), "w", encoding="utf-8").write("\n".join(REPORT) + "\n")
json.dump(dict(prod_check=prod_check, old_check=old_check, raw_check=raw_check, sets=dict(D29=D29, C26=C26),
               production=dict(C26=pv26, C26_hard_soft=pv26s, high=[hi_hits, len(HI)], low=[lo_hits, len(LO)],
                               fisher_band=fisher_band, p_endorsed_uniform_exact=p26_exact, p_high_exact=p_hi_exact,
                               gate=g_prod, spearman=sv_prod26),
               old=dict(gate=g_old, spearman_C26=sv_old26), v1_stored=dict(spearman_D30=sv_xl30),
               kappa_range=[float(lev_k.min()), float(lev_k.max())]),
          open(os.path.join(OUT, "expert_summary.json"), "w", encoding="utf-8"), indent=1, default=float)
print("\n".join(REPORT))
print("written to", OUT)
