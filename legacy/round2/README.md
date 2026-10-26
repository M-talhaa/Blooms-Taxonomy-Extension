# First-revision files replaced in the second revision

Kept so the first-revision state stays inspectable. Nothing here is read by the current
scripts, and the scripts here are not meant to be rerun from this folder.

| File | Replaced by | Why |
| --- | --- | --- |
| `expert_recompute.md` | `results/expert/expert_report.md` (from `revision_pipeline/expert_recompute.py`) | The counts are unchanged, but the old report compared the current model on 26 verbs with the original model on 30, used Monte Carlo p-values, and had no baselines, permutation test or paired tests. |
| `evaluation_upgrade.py`, `evaluation_upgrade.md` | Gate analysis: `results/expert/`. Independent-encoder gap trend: `results/bank_structure/` (from `revision_pipeline/bank_structure.py`). | The stored report describes an interim model (25 accepted, 4 rejected rated verbs), not the production model. Its matched-acceptance comparison used cross-validation results computed at default hyperparameters and P50 for every family, not each family's selected configuration. The script also read the expert workbook from a path that no longer exists. |
| `core_structure/Fig1_nominal_centroid_distance_cosine.png`, `core_structure/Fig_nominal_dn_curves.png` | `results/core_structure/Fig2_mean_centroid_distance_by_gap.png` (from `revision_pipeline/regen_core_figures.py`) | The quantity plotted is the mean centroid distance d_n of Eq. 7; 'nominal' describes the gap, not the distance. |
| `core_structure/Fig_core_high_heatmap.png`, `core_structure/Fig_core_low_heatmap.png` | `results/core_structure/Fig1_core_centroid_distance_heatmaps.png` | The two panels now share one colour scale and carry panel labels (a) and (b). |
