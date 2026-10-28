# Revision pipeline

Standalone scripts behind every number in the article and in the comparison with the original
submission (`results/change_attribution/`).
`retrain.py` reproduces the BT 3 notebook pipeline (every constant and every deviation from the
notebook is listed in its header); the other scripts add the evaluation, selection, robustness,
bank-structure and expert analyses, and the comparison with the original submission. All scripts
run from the repository root with `python -I -B revision_pipeline/<script>` and need no network
access once the encoders are cached (see the root README).

## Reproduction order

The exact command sequence, with runtimes and the expected differences between reruns, is in the
root [`README.md`](../README.md#reproduction). In short:

| Step | Script | Writes |
| --- | --- | --- |
| 1 | `normalize.py` | `cleaned_candidates.txt` (1,652 lemmas), `normalization_audit.csv` |
| 2 | `reconcile_table9.py` | nothing (prints the comparison of the two transcription passes with the original core) |
| 3 | `merge_core.py` | `core_merged.csv` (381 verbs, 607 assignments), `merge_audit.csv`, `candidates_final.txt` (1,650) |
| 4 | `regen_core_figures.py` | `results/core_structure/` Fig. 1, Fig. 2, `core_gap_means.csv` |
| 5 | `retrain.py ... --scan-only` | `results/structural_scan_components.csv`, `model_comparison_table5.csv` (Table 5 of the original submission; structural column of Table 8 in the article), `model_comparison_pareto.csv` |
| 6 | `supervised_eval.py all` | `results/selection/` supervised grid, matched table, baselines, McNemar, per-level, encoder ablation |
| 7 | `select_model.py` | `results/selection/selection_table.csv`, `selection_report.md`, `v2_default_config_reference.csv` |
| 8 | `robustness.py` | `results/selection/robustness_*.csv`, `robustness_report.md` |
| 9 | `scs_sensitivity.py` | `scs_sensitivity.md` (short summary of the SCS-weight part of step 8) |
| 10 | `retrain.py ...` (final stage) | `results/Verb List Classified by Model.csv`, `results/winner_and_constants.json`, a winner bundle under `results/BT4_structural_best/` |
| 11 | `methodology_upgrades.py` | `methodology_upgrades.md`, `heldout_results.csv` (round-2 held-out protocol) |
| 12 to 14 | `cv_metrics.py` (three encoders) | `cv_results_*.csv`, `cv_summary_*.md` (round-2 CV protocol) |
| 15 | `bank_structure.py` | `results/bank_structure/` |
| 16 | `expert_recompute.py` | `results/expert/` |
| 17, 18 | `regen_bank_figures.py`, `regen_model_space.py` | `results/figures/` |
| 19 | `change_attribution.py` | `results/change_attribution/` (bank change and refit attribution relative to the original submission) |
| 20 | `paper_results.py` | `results/paper_tables.md`, `results/paper_numbers.json` |

Steps 11 to 14 regenerate the round-2 default-configuration outputs, which are kept for
traceability; the tables of the article do not depend on them (`results/paper_tables.md` quotes
them only as round-2 reference values, in the note under Section 5 and in Section 19).

## Files

Data preparation
- `raw_candidates_1824.txt`: the 1,824 raw candidate verbs of the original submission, in the
  row order of the original bank (input of `normalize.py`).
- `normalize.py`, `normalization_audit.csv`, `cleaned_candidates.txt`: candidate-pool cleanup
  (WordNet lemmatization, British spelling merged only where it resolves a real duplicate,
  auxiliaries and core-verb leaks removed, duplicates collapsed), one audit row per raw candidate.
- `core_published.csv`, `table9_pass1.csv`, `table9_pass2.csv`: the published training table
  (Sharunova et al. 2019, appendix Table 9) and its two transcription passes. The passes were
  produced by AI-assisted transcription and are byte-identical, so they confirm each other but are
  not an independent check against the printed table. `reconcile_table9.py` compares them with
  `core_notebook_original.csv`, the superseded 358-verb working copy of the original submission.
- `merge_core.py`, `merge_audit.csv`, `stanny_extracted.csv`, `newton_extracted.csv`: merge of the
  audited core with Stanny (2016) Table 1 and Newton, Da Silva and Peters (2020) Table 1 through
  the 1956 to 2001 crosswalk; writes `core_merged.csv` and `candidates_final.txt`.
- `all_embeddings.npz`: all-mpnet-base-v2 vectors (L2-normalized, one per word) of every core and
  candidate verb. Every script reads the vectors from here; delete it to re-encode from scratch.

Model training, selection and export
- `retrain.py`: the BT 3 pipeline (model zoo, out-of-fold empirical-CDF calibration, percentile
  threshold scan, structural objective, top-2 margin collapse with delta 0.060, soft tier at
  0.85 x threshold). `--scan-only` stops after the structural scan; the final stage applies the
  selection rule of `select_model.py` to the scan and exports the selected family. Missing
  supervised results stop it with an error (there is no structural-only fallback).
- `supervised_eval.py`: repeated stratified 5 x 5 cross-validation and the held-out evaluation of
  the full production pipeline, for every family at every configuration and percentile of the
  scan, so each family can be evaluated at its own structural best (matched configurations);
  paired percentile bootstrap, exact McNemar tests, baselines, per-level results of the production
  model and the encoder ablation. It imports `retrain.py` and `cv_metrics.py`; rerun it when either
  changes.
- `select_model.py`: the selection rule as one named, switchable function
  (`SELECTION_RULE = "balanced_maximin"`, function `rule_balanced_maximin`), the eligibility
  definition, the selection table and report, and the round-2 two-stage rule for comparison.
- `robustness.py`: robustness of the selection (SCS weights, held-out variants, guardrails and
  penalties, decision margins, aggregation alternatives, Legacy kNN admitted under an
  evaluation-only construction).
- `scs_sensitivity.py`, `scs_sensitivity.md`: thin wrapper around the SCS-weight analysis of
  `robustness.py` (500 joint draws and 20 single-weight changes); writes only the short summary.
  The full per-setting table is `results/selection/robustness_scs_weights.csv`.

Round-2 protocols, kept for traceability
- `cv_metrics.py`, `cv_results_*.csv`, `cv_summary_*.md`: the round-2 CV protocol (default
  configurations, P50) for three encoders. The McNemar tests and per-level F1 in the summaries use
  the production model kNN-21 as the reference.
- `methodology_upgrades.py`, `methodology_upgrades.md`, `heldout_results.csv`: the round-2 held-out
  protocol (default configurations, seed 13).

Bank structure, expert agreement, figures and tables
- `bank_structure.py`: coverage and cardinality of the delivered bank with both density ratios and
  the primary-domain distribution, extended-bank centroid distances and gap means, and the gap
  trend in three encoders (`results/bank_structure/`). The original-submission bank is read from
  its committed copy `legacy/original_submission_bank_v1.csv` (a byte copy of the bank at commit
  2a87a33) or, if that file is missing, from git history (`git show 2a87a33`); without either the
  comparison column is skipped.
- `expert_recompute.py`: expert agreement of the production model, the like-for-like comparison
  with the archived original-submission model, baselines, label-permutation test, Fleiss kappa and
  threshold sweep (`results/expert/`).
- `regen_core_figures.py`: Figures 1 and 2 (core structure), checked against the core tables.
- `regen_bank_figures.py`, `regen_model_space.py`: the bank figures in `results/figures/`
  (repository figures; not in the article).
- `change_attribution.py`: the comparison of the delivered bank and the production model with
  the original submission (`results/change_attribution/`). (a) The original-submission bank
  (`legacy/original_submission_bank_v1.csv`) carried over to the 1,650 normalized candidates
  through `normalization_audit.csv` and compared verb by verb with the delivered bank: acceptance,
  agreement of the accept or reject decision, same primary level, distance between the primary
  levels and transitions. A lemma takes the v1 row of the raw form equal to it (else the first kept
  or lemmatized raw form); the first-form rule is reported as the alternative. (b) Refits of
  SGDClassifier (the archived original configuration and alpha 1e-4) and kNN-21 on the original
  358-verb core and on the corrected 381-verb core with the production procedure of `retrain.py`
  (imported), scored against the expert ratings on four verb sets (all 30 rated verbs, the 29 rated
  verbs in the candidate pool, the verbs of those 29 that each model accepts, and the 26 of them that
  the production model accepts), with exact McNemar tests against the production model and the
  core x family 2 x 2. The refit of the production configuration must reproduce the delivered bank,
  and the refits on the corrected core must reproduce the penalized objectives of
  `results/model_comparison_table5.csv`; the script stops otherwise.
- `paper_results.py`: every table and number of the article, read from the outputs above and
  cross-checked; writes `results/paper_tables.md` and `results/paper_numbers.json` (Section 20
  holds the comparison with the original submission from `change_attribution.py`).

## Notes

- Row weights: under scikit-learn 1.9.0 every weighted one-vs-rest fit raises and falls back to
  the unweighted fit, so the row weights reach only the weighted threshold percentiles. The code
  keeps this behavior, counts the fallbacks and records them (`row_weight_fit` in
  `results/winner_and_constants.json`, `grid_meta.json` in `results/selection/`).
- BT 3's committed state is missing the cell that defined the top-2 margin delta; `retrain.py`
  restores the published value (0.060) and documents it (deviation D14).
- Each step writes into the repository in place. Run them in the order above; a later step reads
  what an earlier one wrote.
