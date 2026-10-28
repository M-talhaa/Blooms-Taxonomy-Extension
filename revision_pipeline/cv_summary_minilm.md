# CV summary (minilm, 381 core verbs, repeated 5x5 stratified CV, P50; round-2 protocol at the fixed configurations listed, kept for traceability)

The CV results reported in the article are at matched configurations: `results/selection/matched_supervised.csv` (Table 6, all-mpnet-base-v2) and `results/selection/encoder_ablation.csv` (Table 7, all three encoders).

| Model | macro AUC | macro F1 (mean over repeats) | 95% CI (seed 13 repeat) | micro F1 | subset acc |
|---|---|---|---|---|---|
| SGD-hinge-1e4 | 0.644 | 0.437 | [0.422, 0.482] | 0.435 | 0.065 |
| SGD-log-1e5 | 0.649 | 0.437 | [0.393, 0.457] | 0.432 | 0.077 |
| SGD-hinge-1e5 | 0.634 | 0.437 | [0.405, 0.464] | 0.433 | 0.062 |
| LinearSVC-C1 | 0.674 | 0.460 | [0.424, 0.488] | 0.456 | 0.087 |
| LogReg-C1 | 0.698 | 0.489 | [0.459, 0.529] | 0.486 | 0.113 |
| Ridge-a1 | 0.687 | 0.473 | [0.432, 0.499] | 0.468 | 0.100 |
| PassiveAggressive-C1 | 0.647 | 0.436 | [0.389, 0.455] | 0.431 | 0.075 |
| kNN-21 | 0.703 | 0.481 | [0.444, 0.511] | 0.485 | 0.110 |
| kNN-11 | 0.701 | 0.478 | [0.462, 0.525] | 0.483 | 0.111 |
| NearestCentroid | 0.702 | 0.490 | [0.461, 0.528] | 0.490 | 0.117 |
| GaussianNB | 0.690 | 0.475 | [0.434, 0.503] | 0.477 | 0.095 |
| LDA | 0.686 | 0.470 | [0.438, 0.504] | 0.472 | 0.088 |
| QDA | 0.664 | 0.436 | [0.407, 0.467] | 0.443 | 0.071 |
| CalibratedSVC | 0.699 | 0.496 | [0.469, 0.535] | 0.492 | 0.109 |
| majority-level | nan | 0.090 | [0.081, 0.098] | 0.283 | 0.150 |
| freq-random | nan | 0.271 | [0.243, 0.311] | 0.297 | 0.043 |

Per-level F1 (reference kNN-21, mean over 5 repeats):
| Kn | Cm | Ap | An | Sn | Ev |
|---|---|---|---|---|---|
| 0.507 | 0.433 | 0.506 | 0.494 | 0.505 | 0.439 |

Paired exact McNemar (reference kNN-21 vs competitor, per verb-level decision, seed-13 repeat):
- vs SGD-hinge-1e4: reference-only correct 373, competitor-only correct 227, exact p = 2.711e-09
- vs SGD-log-1e5: reference-only correct 360, competitor-only correct 220, exact p = 6.67e-09
- vs SGD-hinge-1e5: reference-only correct 381, competitor-only correct 212, exact p = 3.772e-12
- vs LinearSVC-C1: reference-only correct 276, competitor-only correct 207, exact p = 0.001944
- vs LogReg-C1: reference-only correct 177, competitor-only correct 206, exact p = 0.1524
- vs Ridge-a1: reference-only correct 230, competitor-only correct 201, exact p = 0.1774
- vs PassiveAggressive-C1: reference-only correct 340, competitor-only correct 227, exact p = 2.377e-06
- vs kNN-11: reference-only correct 136, competitor-only correct 154, exact p = 0.3181
- vs NearestCentroid: reference-only correct 165, competitor-only correct 195, exact p = 0.1263
- vs GaussianNB: reference-only correct 219, competitor-only correct 187, exact p = 0.1238
- vs LDA: reference-only correct 237, competitor-only correct 230, exact p = 0.7813
- vs QDA: reference-only correct 329, competitor-only correct 216, exact p = 1.48e-06
- vs CalibratedSVC: reference-only correct 178, competitor-only correct 198, exact p = 0.3272
- vs majority-level: reference-only correct 403, competitor-only correct 367, exact p = 0.2072
- vs freq-random: reference-only correct 548, competitor-only correct 376, exact p = 1.688e-08