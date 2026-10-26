# Material from earlier versions

This folder keeps files from the original submission (top level) and from the first revision
(`round2/`). Some of them are still read by the current analysis; the table says which.

## Original submission

| File | What it is | Status |
| --- | --- | --- |
| `Final File(with model & Expert Classification).xlsx` | The expert study: per-verb vote counts of the eight raters for the 30 sampled verbs (`cnt_Kn` to `cnt_Ev`, `cnt_None`), the rated form (`Verb_clean`), and the original model's output for the form it was run on (`Verb`). | **In use.** It is the only copy of the vote matrix. `revision_pipeline/expert_recompute.py`, `merge_core.py` and the BT 4 and BT 5 notebooks read it. The model columns are the original-submission outputs, not the current model's. |
| `ExpertSample_single (Model Classified Sample).csv` | The original model's outputs for the 30 sampled verbs, in the surface forms it was run on. | Kept as the record of the original sample. Not replaced by `results/ExpertSample_single.csv`, which is a new 30-verb draw from the current bank and shares no verb with the rated sample. |
| `original-submission-model/` | The original production bundle (SGDClassifier, hinge, alpha 1e-5, P60 thresholds, saved 2025-10-12, pickled with scikit-learn 1.2.2). | Superseded as the production model by `production_model/` (kNN, n = 21, cosine, distance-weighted, P50). It still reproduces the original submission's calibrated scores on the rated verbs (maximum absolute difference about 5e-6), and `expert_recompute.py` uses it for the like-for-like comparison of the two versions. It does not reproduce the revised results, which rest on a different core, candidate pool and model. |
| `BTverblist_new.txt` | 4,391 verbs written by a BT 2 extraction run of the original submission. | Superseded. It is not the 1,824-verb candidate pool of the original submission (7 of those 1,824 verbs are not in it). That pool is `revision_pipeline/raw_candidates_1824.txt` (the `Verb` column of the original bank), which `revision_pipeline/normalize.py` turns into the 1,650 candidates in `revision_pipeline/candidates_final.txt`. |

Why the original model and pool were replaced:

- The training core was audited against its published source and enriched from two further
  published consolidations: 381 verbs and 607 assignments now, against 358 and 559 before.
- The candidate pool was normalized, which removed core-verb variants, within-pool duplicates
  and auxiliaries. `revision_pipeline/normalization_audit.csv` records what happened to each
  of the 1,824 raw candidates.
- Model selection changed from the structural score alone to a rule that also uses
  cross-validated generalization and held-out end-to-end accuracy (see `revision_pipeline/`).

## First revision (`round2/`)

Outputs and scripts of the first revision that the current scripts replace. See
`round2/README.md`.
