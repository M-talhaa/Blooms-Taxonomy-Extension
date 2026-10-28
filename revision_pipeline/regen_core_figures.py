"""Regenerate Figures 1 and 2 of the paper (core-list structure, Section 2.1) from the audited core.

Core: revision_pipeline/core_merged.csv (381 verbs, 607 verb-level assignments). Embeddings: the
stored all-mpnet-base-v2 vectors in revision_pipeline/all_embeddings.npz (768-d, L2-normalized).
Definitions follow Eq. 5 to 7: a level centroid is the L2-normalized mean embedding of the verbs
assigned to the level, the distance between two levels is 1 - cosine similarity of their centroids,
and d_n is the mean distance over the level pairs that are n steps apart in the order
Kn, Cm, Ap, An, Sn, Ev (the taxonomic gap n). High-frequency entries have a source frequency of at
least 10, low-frequency entries below 10 (135 and 472 assignments).

Fig. 1: heatmaps of the high-frequency (a) and low-frequency (b) centroid distances on one shared
cosine-distance colour scale, with the cell values printed (they equal Tables 3 and 4).
Fig. 2: mean centroid distance d_n against the taxonomic gap n for all, high- and low-frequency core.
(The original submission's Fig. 2 plotted Euclidean distances between unnormalized level means.
The first-revision export, which used cosine distances but labelled them 'nominal' distances, is
superseded and kept in legacy/round2/core_structure/.)

Checks: the three distance matrices are compared with results/core_structure/T1_core_all.csv,
T2_core_high.csv and T3_core_low.csv (Tables 2 to 4) and the script stops if any cell differs by
more than 5e-4.

Outputs (results/core_structure/): Fig1_core_centroid_distance_heatmaps.png,
Fig2_mean_centroid_distance_by_gap.png, core_gap_means.csv.
Run: python -I -B revision_pipeline/regen_core_figures.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "results", "core_structure")
LEVELS = ["Kn", "Cm", "Ap", "An", "Sn", "Ev"]
FREQ_SPLIT = 10

_z = np.load(os.path.join(HERE, "all_embeddings.npz"))
EMB = {str(w).strip().lower(): np.asarray(v, float) for w, v in zip(_z["words"], _z["vectors"])}
core = pd.read_csv(os.path.join(HERE, "core_merged.csv"))
core["verb"] = core["verb"].astype(str).str.strip().str.lower()
core["freq"] = pd.to_numeric(core["freq"], errors="coerce").fillna(0)
assert core["verb"].nunique() == 381 and len(core) == 607


def dist_matrix(rows):
    cents = []
    for l in LEVELS:
        V = np.stack([EMB[v] for v in rows.loc[rows["level"] == l, "verb"]])
        m = V.mean(0)
        cents.append(m / np.linalg.norm(m))
    C = np.stack(cents)
    D = 1.0 - C @ C.T
    np.fill_diagonal(D, 0.0)
    return D


def gap_means(D):
    return [float(np.mean([D[i, i + n] for i in range(6 - n)])) for n in range(1, 6)]


SUBSETS = {
    "all": (core, "All core (381 verbs, 607 assignments)", "T1_core_all.csv"),
    "high": (core[core["freq"] >= FREQ_SPLIT], f"High-frequency core (freq >= {FREQ_SPLIT}, "
             f"{int((core['freq'] >= FREQ_SPLIT).sum())} assignments)", "T2_core_high.csv"),
    "low": (core[core["freq"] < FREQ_SPLIT], f"Low-frequency core (freq < {FREQ_SPLIT}, "
            f"{int((core['freq'] < FREQ_SPLIT).sum())} assignments)", "T3_core_low.csv"),
}
assert len(SUBSETS["high"][0]) == 135 and len(SUBSETS["low"][0]) == 472

D, GAP = {}, {}
for key, (rows, label, table) in SUBSETS.items():
    D[key] = dist_matrix(rows)
    ref = pd.read_csv(os.path.join(OUT, table), index_col=0).loc[LEVELS, LEVELS].to_numpy(float)
    diff = float(np.abs(D[key] - ref).max())
    assert diff < 5e-4, f"{table}: max abs difference {diff:.2e}"
    print(f"{label}: matches {table} (max abs difference {diff:.1e})")
    GAP[key] = gap_means(D[key])

pd.DataFrame([dict(subset=SUBSETS[k][1], **{f"d{n}": GAP[k][n - 1] for n in range(1, 6)}) for k in SUBSETS]
             ).to_csv(os.path.join(OUT, "core_gap_means.csv"), index=False, float_format="%.6f")

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- Fig. 1
vmax = float(np.ceil(max(D["high"].max(), D["low"].max()) * 20) / 20)   # shared scale, rounded up to 0.05
cmap = plt.get_cmap("Blues")
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), constrained_layout=True)
for ax, key, tag in ((axes[0], "high", "(a)"), (axes[1], "low", "(b)")):
    M = D[key]
    im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=vmax)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.55 * vmax else "#1a1a1a")
    ax.set_xticks(range(6), LEVELS)
    ax.set_yticks(range(6), LEVELS)
    ax.set_xlabel("Cognitive level")
    ax.set_ylabel("Cognitive level")
    ax.set_title(f"{tag} {SUBSETS[key][1]}", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
cb = fig.colorbar(im, ax=axes, shrink=0.9)
cb.set_label("Cosine distance between level centroids (1 - cosine similarity)")
cb.outline.set_visible(False)
p1 = os.path.join(OUT, "Fig1_core_centroid_distance_heatmaps.png")
fig.savefig(p1, dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Fig. 2
STYLE = {"all": ("#1f4e79", "o", "-"), "high": ("#c55a11", "s", "--"), "low": ("#5b8c3a", "^", ":")}
fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
x = np.arange(1, 6)
for key in ("high", "all", "low"):
    color, marker, ls = STYLE[key]
    ax.plot(x, GAP[key], color=color, marker=marker, linestyle=ls, linewidth=2, markersize=7,
            label=SUBSETS[key][1])
ax.set_xticks(x)
ax.set_xlim(0.7, 5.3)
ax.set_ylim(0, max(max(v) for v in GAP.values()) * 1.15)
ax.set_xlabel("Taxonomic gap n (levels apart)")
ax.set_ylabel("Mean centroid distance $d_n$ (cosine)")
ax.grid(axis="y", alpha=0.3)
ax.legend(frameon=False, loc="upper left", fontsize=9)
p2 = os.path.join(OUT, "Fig2_mean_centroid_distance_by_gap.png")
fig.savefig(p2, dpi=300)
plt.close(fig)

for k in SUBSETS:
    print(f"d_n {k}: " + ", ".join(f"{v:.3f}" for v in GAP[k]))
print("written:", p1, p2, sep="\n  ")
