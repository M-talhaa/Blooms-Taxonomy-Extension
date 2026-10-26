# CV summary (droberta, 381 core verbs, repeated 5x5 stratified CV, winner percentile P50)

| Model | macro AUC | macro F1 (mean over repeats) | 95% CI (seed 13 repeat) | micro F1 | subset acc |
|---|---|---|---|---|---|
| SGD-hinge-1e4 | 0.649 | 0.427 | [0.399, 0.458] | 0.432 | 0.064 |
| SGD-log-1e5 | 0.658 | 0.439 | [0.408, 0.468] | 0.442 | 0.066 |
| SGD-hinge-1e5 | 0.640 | 0.431 | [0.400, 0.459] | 0.438 | 0.072 |
| LinearSVC-C1 | 0.669 | 0.455 | [0.425, 0.487] | 0.459 | 0.078 |
| LogReg-C1 | 0.683 | 0.469 | [0.438, 0.501] | 0.472 | 0.103 |
| Ridge-a1 | 0.677 | 0.465 | [0.437, 0.498] | 0.468 | 0.097 |
| PassiveAggressive-C1 | 0.655 | 0.432 | [0.394, 0.457] | 0.436 | 0.072 |
| kNN-21 | 0.660 | 0.438 | [0.415, 0.476] | 0.443 | 0.081 |
| kNN-11 | 0.664 | 0.442 | [0.401, 0.467] | 0.446 | 0.077 |
| NearestCentroid | 0.681 | 0.472 | [0.440, 0.505] | 0.476 | 0.118 |
| GaussianNB | 0.677 | 0.473 | [0.435, 0.497] | 0.478 | 0.108 |
| LDA | 0.684 | 0.466 | [0.444, 0.505] | 0.470 | 0.097 |
| QDA | 0.647 | 0.431 | [0.399, 0.461] | 0.436 | 0.069 |
| CalibratedSVC | 0.681 | 0.471 | [0.438, 0.501] | 0.474 | 0.092 |
| majority-level | nan | 0.090 | [0.081, 0.098] | 0.283 | 0.150 |
| freq-random | nan | 0.271 | [0.243, 0.311] | 0.297 | 0.043 |

Per-level F1 (reference kNN-21, mean over 5 repeats):
| Kn | Cm | Ap | An | Sn | Ev |
|---|---|---|---|---|---|
| 0.419 | 0.394 | 0.452 | 0.455 | 0.482 | 0.426 |

Paired exact McNemar (reference kNN-21 vs competitor, per verb-level decision, seed-13 repeat):
- vs SGD-hinge-1e4: reference-only correct 361, competitor-only correct 287, exact p = 0.0041
- vs SGD-log-1e5: reference-only correct 314, competitor-only correct 303, exact p = 0.6873
- vs SGD-hinge-1e5: reference-only correct 321, competitor-only correct 304, exact p = 0.5222
- vs LinearSVC-C1: reference-only correct 259, competitor-only correct 279, exact p = 0.4127
- vs LogReg-C1: reference-only correct 187, competitor-only correct 269, exact p = 0.0001432
- vs Ridge-a1: reference-only correct 220, competitor-only correct 276, exact p = 0.01345
- vs PassiveAggressive-C1: reference-only correct 312, competitor-only correct 317, exact p = 0.8733
- vs kNN-11: reference-only correct 180, competitor-only correct 201, exact p = 0.3055
- vs NearestCentroid: reference-only correct 189, competitor-only correct 264, exact p = 0.000494
- vs GaussianNB: reference-only correct 214, competitor-only correct 251, exact p = 0.09492
- vs LDA: reference-only correct 218, competitor-only correct 300, exact p = 0.0003629
- vs QDA: reference-only correct 298, competitor-only correct 267, exact p = 0.2069
- vs CalibratedSVC: reference-only correct 210, competitor-only correct 261, exact p = 0.02113
- vs majority-level: reference-only correct 402, competitor-only correct 469, exact p = 0.02527
- vs freq-random: reference-only correct 530, competitor-only correct 461, exact p = 0.03071