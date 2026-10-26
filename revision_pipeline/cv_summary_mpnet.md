# CV summary (mpnet, 381 core verbs, repeated 5x5 stratified CV, winner percentile P50)

| Model | macro AUC | macro F1 (mean over repeats) | 95% CI (seed 13 repeat) | micro F1 | subset acc |
|---|---|---|---|---|---|
| SGD-hinge-1e4 | 0.666 | 0.447 | [0.412, 0.481] | 0.450 | 0.093 |
| SGD-log-1e5 | 0.674 | 0.459 | [0.417, 0.484] | 0.460 | 0.097 |
| SGD-hinge-1e5 | 0.665 | 0.455 | [0.430, 0.495] | 0.460 | 0.097 |
| LinearSVC-C1 | 0.701 | 0.484 | [0.429, 0.501] | 0.485 | 0.128 |
| LogReg-C1 | 0.720 | 0.508 | [0.476, 0.548] | 0.508 | 0.156 |
| Ridge-a1 | 0.713 | 0.501 | [0.464, 0.533] | 0.500 | 0.135 |
| PassiveAggressive-C1 | 0.672 | 0.457 | [0.410, 0.475] | 0.458 | 0.098 |
| kNN-21 | 0.713 | 0.492 | [0.470, 0.537] | 0.496 | 0.142 |
| kNN-11 | 0.706 | 0.473 | [0.454, 0.519] | 0.478 | 0.120 |
| NearestCentroid | 0.716 | 0.491 | [0.474, 0.541] | 0.494 | 0.148 |
| GaussianNB | 0.711 | 0.501 | [0.481, 0.552] | 0.502 | 0.133 |
| LDA | 0.705 | 0.480 | [0.459, 0.525] | 0.484 | 0.127 |
| QDA | 0.677 | 0.459 | [0.439, 0.504] | 0.464 | 0.104 |
| CalibratedSVC | 0.725 | 0.512 | [0.481, 0.552] | 0.512 | 0.159 |
| majority-level | nan | 0.090 | [0.081, 0.098] | 0.283 | 0.150 |
| freq-random | nan | 0.271 | [0.243, 0.311] | 0.297 | 0.043 |

Per-level F1 (reference kNN-21, mean over 5 repeats):
| Kn | Cm | Ap | An | Sn | Ev |
|---|---|---|---|---|---|
| 0.476 | 0.499 | 0.486 | 0.524 | 0.510 | 0.458 |

Paired exact McNemar (reference kNN-21 vs competitor, per verb-level decision, seed-13 repeat):
- vs SGD-hinge-1e4: reference-only correct 321, competitor-only correct 228, exact p = 8.315e-05
- vs SGD-log-1e5: reference-only correct 302, competitor-only correct 211, exact p = 6.798e-05
- vs SGD-hinge-1e5: reference-only correct 345, competitor-only correct 217, exact p = 7.466e-08
- vs LinearSVC-C1: reference-only correct 236, competitor-only correct 188, exact p = 0.02235
- vs LogReg-C1: reference-only correct 157, competitor-only correct 177, exact p = 0.2985
- vs Ridge-a1: reference-only correct 186, competitor-only correct 185, exact p = 1
- vs PassiveAggressive-C1: reference-only correct 314, competitor-only correct 207, exact p = 3.17e-06
- vs kNN-11: reference-only correct 171, competitor-only correct 134, exact p = 0.03909
- vs NearestCentroid: reference-only correct 163, competitor-only correct 180, exact p = 0.3877
- vs GaussianNB: reference-only correct 161, competitor-only correct 177, exact p = 0.4146
- vs LDA: reference-only correct 222, competitor-only correct 195, exact p = 0.2029
- vs QDA: reference-only correct 264, competitor-only correct 176, exact p = 3.176e-05
- vs CalibratedSVC: reference-only correct 152, competitor-only correct 186, exact p = 0.0725
- vs majority-level: reference-only correct 418, competitor-only correct 329, exact p = 0.001268
- vs freq-random: reference-only correct 557, competitor-only correct 332, exact p = 4.264e-14