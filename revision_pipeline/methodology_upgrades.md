# Round-2 held-out evaluation (default configurations, outer split seed 13; kept for traceability)

The held-out results reported in the article (Table 11) are means over outer seeds 13 to 17 at
matched configurations (`results/selection/matched_supervised.csv`, written by `supervised_eval.py`).

## Held-out full-pipeline evaluation with ordinal metrics

Held-out core verbs are passed through the full production pipeline (calibration,
P50 thresholds, top-2 collapse). All are known Bloom verbs, so the gate should
accept them. Each model is at its round-2 default configuration; 5-fold multi-label
stratified split with seed 13 (one repeat), means over the five folds:

| Pipeline | Gate recall | Primary-in-labels | Within-1-level | Mean ordinal dist | macro F1 | macro AUC |
|---|---|---|---|---|---|---|
| CalibratedSVC | 0.895 | 0.587 | 0.744 | 0.81 | 0.486 | 0.724 |
| Ensemble(3, vote>=2) | 0.890 | 0.618 | 0.773 | 0.73 | 0.479 | 0.729 |
| LinearSVC-C1 | 0.908 | 0.552 | 0.723 | 0.88 | 0.443 | 0.694 |
| LogReg-C1 | 0.895 | 0.593 | 0.736 | 0.81 | 0.489 | 0.720 |
| NearestCentroid | 0.882 | 0.597 | 0.736 | 0.81 | 0.479 | 0.722 |
| kNN-21 | 0.924 | 0.600 | 0.784 | 0.74 | 0.477 | 0.719 |

'Gate recall' is the share of held-out core verbs (all known Bloom verbs) that pass
the acceptance gate. 'Primary-in-labels'
scores the primary domain against the verb's published level set. 'Within-1-level'
credits adjacent-level predictions, reflecting the ordinal nature of the taxonomy.