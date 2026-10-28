# Run log: results/selection

## supervised_eval.py (supervised evaluation, round 3)

Command, run from the repository root on branch round3-revision (from main 2c0e25b):

```
python -I -B revision_pipeline/supervised_eval.py all --workers 8
```

Equivalent step by step: `grid --workers 8`, then `analyze`, then `encoders --workers 8`.
No network access: the script sets HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 and loads the
encoders with local_files_only=True from the local Hugging Face cache.

Run on 2026-10-24, start 17:27:47, end 17:40:25 (UTC-6); exit code 0; 0 errors; 0 warnings captured.

| step | runtime (s) | outputs |
|---|---|---|
| grid (30 configurations x 9 percentiles x 5 repeats, 8 workers) | 153.5 | supervised_grid_per_seed.csv (1,350 rows), oof_decisions.npz, grid_meta.json |
| analyze (10,000 bootstrap resamples, exact McNemar) | 268.2 | supervised_grid.csv (270 rows x 428 columns), supervised_grid_bootstrap.csv, supervised_grid_mcnemar.csv, matched_supervised.csv, mcnemar_vs_production.csv, baselines.csv, production_per_level.csv, v2_reproduction_check.csv |
| encoders (30 jobs, 8 workers, includes encoding) | 333.1 | encoder_ablation.csv, encoder_vectors.npz, encoder_ablation_meta.json |
| total | 754.9 (wall 12 min 38.6 s) | |

Environment: Python 3.12.10 (Windows 10, 8 cores), numpy 2.5.1, pandas 3.0.5, scikit-learn 1.9.0,
iterative-stratification 0.1.9, sentence-transformers 5.6.1, torch 2.13.0+cpu.

### Protocol facts recorded by the run (grid_meta.json)

* Configurations: the 27 one-vs-rest configurations of results/winner_and_constants.json model_zoo
  (checked equal to retrain.MODEL_GRID) plus 3 supplementary LDA/QDA rows. Legacy kNN is not a
  one-vs-rest classifier and is not evaluated here.
* LDA/QDA primary rows use the projection of the structural scan (retrain.make_estimator:
  whitened PCA with retrain._pca_dim_for_lda_qda(X_fit, n_classes=2) = 2 components). The
  supplementary rows with suffix `|PCA24` use the 24-component whitened PCA of round-2
  cv_metrics.py and exist only to reproduce the round-2 CV table.
* Row weights: all 4,500 weighted one-vs-rest fits raised under scikit-learn 1.9.0 and fell back
  to the unweighted fit (weighted_fit_ok = 0, weighted_fit_fallback_unweighted = 4,500). Row
  weights reach only the weighted threshold percentiles, as in round 2.
* Held-out pipeline: retrain._apply_top2_margin with delta 0.060, primary = argmax of calibrated
  scores among accepted levels, soft tier at 0.85 x threshold; per-fold metrics averaged over
  the five outer folds.
* Bootstrap: 95% percentile bootstrap, 10,000 resamples of the 381 verbs, numpy default_rng(13),
  the same resamples for every configuration and repeat. The vectorized bootstrap equals
  cv_metrics.bootstrap_ci exactly for kNN-21 P50 seed 13 (macro F1 CI [0.4701540627, 0.5370211025]);
  first five resamples agree with sklearn to 1.1e-16 (AUC) and 0 (F1).

### Reproduction of round-2 outputs (v2_reproduction_check.csv)

* CV metrics at the round-2 configurations (P50) equal revision_pipeline/cv_results_mpnet.csv to
  5.6e-17 for every configuration except SGD-hinge-1e4 (seed 15 only), SGD-log-1e5 and
  PassiveAggressive-C1. Cause: round-2 cv_metrics.py encoded the verbs live in one batch, this
  script uses the production per-word vectors (all_embeddings.npz); the vectors differ by at most
  1.6e-7, which changes only these stochastic-gradient families.
* Held-out seed 13 at the round-2 configurations equals revision_pipeline/heldout_results.csv
  after rounding (difference 0) for kNN-21, LogReg-C1, NearestCentroid and CalibratedSVC.
* Baselines equal the round-2 majority-level and freq-random rows (max difference 9.0e-17).

### Encoder ablation vs round-2 Table 7 (encoder_ablation.csv)

kNN-21 P50 CV macro AUC (5 repeats): all-mpnet-base-v2 0.713486, all-MiniLM-L6-v2 0.703321,
all-distilroberta-v1 0.659825, which round to the Table 7 values 0.713 / 0.703 / 0.660 (difference
0). Per-word and one-batch encodings give the same kNN-21 values to 6 decimals. The MPNet rows of
the ablation equal the main grid exactly (max abs difference 0).

## End-to-end run of the round-3 sequence

Run on 2026-10-24 (UTC-6) from the repository root on branch round3-revision, with
HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 set; the commands below were run one after the other
(the logs in results/logs/ are from the later full run of the README sequence, not from this
run). Python 3.12.10, numpy 2.5.1, pandas 3.0.5, scipy 1.18.0, scikit-learn 1.9.0, joblib 1.5.3
(as recorded in results/winner_and_constants.json "versions").

| step | command | start | end | runtime (s) | exit |
|---|---|---|---|---|---|
| 1 structural scan | `python -I -B revision_pipeline/retrain.py --core-csv revision_pipeline/core_merged.csv --candidates revision_pipeline/candidates_final.txt --outdir . --scan-only` | 22:17:23 | 22:25:35 | 491.5 | 0 |
| 2 supervised evaluation | `python -I -B revision_pipeline/supervised_eval.py all --workers 8` | 22:25:35 | 22:37:53 | 738.7 | 0 |
| 3 selection | `python -I -B revision_pipeline/select_model.py` | 22:37:53 | 22:37:58 | 4.8 | 0 |
| 4 robustness | `python -I -B revision_pipeline/robustness.py` | 22:37:58 | 22:39:19 | 81.2 | 0 |
| 5 selection + export | `python -I -B revision_pipeline/retrain.py --core-csv revision_pipeline/core_merged.csv --candidates revision_pipeline/candidates_final.txt --outdir <out>` (`<out>` = a separate output folder, so that the committed results/ files are not overwritten) | 22:39:19 | 22:47:24 | 484.2 | 0 |
| 6 round-2 held-out protocol | `python -I -B revision_pipeline/methodology_upgrades.py` | 22:47:33 | 22:47:52 | 19.1 | 0 |
| 7 round-2 CV protocol, MPNet | `python -I -B revision_pipeline/cv_metrics.py` | 22:47:52 | 22:57:32 | 579.8 | 0 |
| 8 round-2 CV protocol, MiniLM | `python -I -B revision_pipeline/cv_metrics.py --encoder sentence-transformers/all-MiniLM-L6-v2 --tag minilm` | 22:57:32 | 23:06:11 | 518.9 | 0 |
| 9 round-2 CV protocol, DistilRoBERTa | `python -I -B revision_pipeline/cv_metrics.py --encoder sentence-transformers/all-distilroberta-v1 --tag droberta` | 23:06:11 | 23:15:31 | 560.3 | 0 |

Steps 1 to 5 are the reproduction order in the retrain.py header. Steps 6 to 9 regenerate the
round-2 default-configuration outputs that are kept for traceability.

### Step 1 (structural scan)

* results/structural_scan_components.csv: byte-identical to the committed file.
* results/model_comparison_table5.csv: the 11 one-vs-rest rows are identical (max abs difference
  0). Only the Legacy kNN row changed, because tune_legacy_knn now ranks by the penalized
  objective (retrain.py D11):

  | | configuration | Eval_SCS | Eval_Acceptance | Eval_en_over_ac | PenalizedObj | Core_Acceptance |
  |---|---|---|---|---|---|---|
  | committed (round-2 rule: largest raw SCS within 0.02) | freq-weighted K=5/th=0.32/c=32 | 0.9674580541 | 0.1921212121 | 1.4984227129 | -1.1967902490 | 0.4435695538 |
  | this run (largest PenalizedObj) | unweighted K=15/th=0.28/c=6 | 1.0036696525 | 0.8490909091 | 1.6074232691 | 1.0036655202 | 0.9527559055 |

  The table is sorted by PenalizedObj, so the Legacy kNN row moves from last to first place,
  above SGDClassifier P55 (0.9834924301). Legacy kNN stays a structural reference row that is not
  eligible for selection.
* results/model_comparison_pareto.csv (front of Eval_SCS and acceptance over the rows of
  model_comparison_table5.csv): committed SGDClassifier P55, PassiveAggressive P50,
  KNeighborsClassifier P50, GaussianNB P40; now Legacy kNN (K=15/th=0.28/c=6),
  KNeighborsClassifier P50, GaussianNB P40. The bug-fixed Legacy row (SCS 1.00367, acceptance
  0.849) dominates SGD (0.99766, 0.804) and PassiveAggressive (0.92145, 0.833).
* Row weights: 162 weighted one-vs-rest fits attempted, 0 accepted, 162 fit unweighted.

### Steps 2 to 4

* supervised_eval.py: every output is byte-identical to the 17:27 run above except the timing
  fields (column runtime_s of supervised_grid.csv and encoder_ablation.csv, runtime fields of
  grid_meta.json and encoder_ablation_meta.json).
* select_model.py: selection_table.csv and v2_default_config_reference.csv byte-identical.
  selection_report.md differs in one line only: the sentence on the Legacy kNN row of
  results/model_comparison_table5.csv quoted the new row (`Legacy kNN (K=15/th=0.28/c=6)`,
  PenalizedObj 1.0036655202) but still attributed it to the round-2 raw-SCS rule.
  select_model.write_report was corrected afterwards, and the committed selection_report.md
  states that the regenerated table applies the bug-fixed rule.
* robustness.py: all nine robustness_* outputs byte-identical (robustness_report.md included);
  revision_pipeline/scs_sensitivity.md (not part of this sequence) unchanged.

### Step 5 (selection and export)

* Rule: select_model.rule_balanced_maximin (SELECTION_RULE = "balanced_maximin", adequacy floor
  0.5). Selected KNeighborsClassifier P50 (n_neighbors 21, cosine, distance-weighted), score
  0.7134261320; runner-up CalibratedSVC 0.5520128218; margin 0.1614133102. The round-2 two-stage
  rule on the same matched inputs (held-out seed 13) would select CalibratedSVC with 0.1704371560
  (reported in winner_and_constants.json, not used).
* Cross-checks inside retrain.py: the re-derivation by select_model from the scan file gives the
  same configurations, PenalizedObj and SCS for all 12 families; results/selection/selection_table.csv
  gives the same family, rule and score.
* Export versus the packaged model and the delivered bank (retrain.py D15, stored under
  "reproduction_checks" in winner_and_constants.json): model.pkl, calibration_cdfs.json,
  thresholds.json, row_weights.npy and bt4_scorer.py byte-identical to production_model/;
  metadata.json equal apart from created_at and the artifact paths; the exported
  "Verb List Classified by Model.csv" is byte-identical to the repository bank, so hard labels,
  primaries, other domains, soft patterns and soft primaries agree for 1,650 of 1,650 verbs.
  Production summary: 1,420 accepted, 230 rejected, 1,243 single label, 177 multi-label, 179
  suggested only (equal to the committed winner_and_constants.json).
* production_model/ and results/Verb List Classified by Model.csv show no change against the
  committed versions.
* results/winner_and_constants.json is the step 5 output copied unchanged. Its winner.bundle_dir
  is relative to the step 5 --outdir and is not archived; the packaged copy of that bundle is
  production_model/ (byte-identical, above).
* Missing supervised results stop the selection: with an empty results folder,
  retrain.select_production_family raises RuntimeError ("Model selection needs the supervised
  evaluation results, which are missing"); there is no structural-only fallback.

### Steps 6 to 9 (round-2 default-configuration outputs, kept for traceability)

* methodology_upgrades.md: byte-identical to the committed file.
* heldout_results.csv is now written by methodology_upgrades.py. Against the committed file the
  four single-model rows read by the round-2 code (kNN-21, LogReg-C1, NearestCentroid,
  CalibratedSVC) are unchanged; the ensemble row changes from 0.911 / 0.618 / 0.778 / 0.74 /
  0.494 / 0.729 to 0.890 / 0.618 / 0.773 / 0.73 / 0.479 / 0.729 (gate recall, primary in labels,
  within one level, mean ordinal distance, macro F1, macro AUC; now equal to
  methodology_upgrades.md and round-2 Table 11), and a LinearSVC-C1 row is added (0.908 / 0.552 /
  0.723 / 0.88 / 0.443 / 0.694).
* cv_results_mpnet.csv: byte-identical to the committed file. cv_summary_mpnet.md: the model table
  is unchanged; the per-level F1 block and the exact McNemar tests now use kNN-21 (the production
  model) as the reference instead of the hard-coded SGD-hinge-1e4. Seed-13 repeat, kNN-21 versus:
  SGD-hinge-1e4 p = 8.315e-05 (kNN-21 correct alone 321, SGD alone 228), LogReg-C1 p = 0.2985,
  Ridge-a1 p = 1, NearestCentroid p = 0.3877, CalibratedSVC p = 0.0725 (152 vs 186).
* cv_results_minilm.csv and cv_results_droberta.csv: the 56 committed rows of each are reproduced
  exactly (max abs difference 0, same order); 20 rows are added per encoder for the four families
  that joined the zoo after these files were first written (CalibratedSVC, GaussianNB, LDA, QDA).
  The summaries are regenerated with kNN-21 as the reference. kNN-21 macro AUC (mean of 5
  repeats): MPNet 0.713486, MiniLM 0.703321, DistilRoBERTa 0.659825 (Table 7 values unchanged).
  MPNet has the highest macro AUC of the three encoders for all 14 default configurations,
  including the four added families (CalibratedSVC 0.724592 / 0.699369 / 0.681260, GaussianNB
  0.711224 / 0.689975 / 0.676611, LDA 0.705241 / 0.686090 / 0.684094, QDA 0.677489 / 0.664123 /
  0.646961 for MPNet / MiniLM / DistilRoBERTa).
