# Evaluation upgrades

## 1. Rejection-aware expert evaluation

Rated verbs present in the bank: 29 (accepted 25, rejected 4).

A rejection is a decision the experts can also judge, because they could vote
'None of the above'. Scoring only accepted verbs discards that evidence.

| | experts: None is plurality | experts: a level leads |
|---|---|---|
| model accepts | 4 | 21 |
| model rejects | 1 | 3 |

- Correct acceptances (model assigns, experts back a level): 21
- Correct rejections   (model declines, experts say None):   1
- False acceptances    (model assigns, experts say None):    4
- False rejections     (model declines, experts back a level): 3
- **Gate accuracy (accept/reject decision): 22/29 = 75.9%**
- Fisher exact on the 2x2: p = 0.5526

None-vote rates by model decision:
- accepted verbs: mean 2.24/8
- rejected verbs: mean 2.25/8
- Mann-Whitney (rejected have more None votes): p = 0.6026

Rejected verbs in detail:
  authorize      None 1/8, top level vote 3/8, expert-endorsed levels ['Ap', 'Ev', 'Sn']
  rest           None 5/8, top level vote 2/8, expert-endorsed levels ['Ap', 'Sn']
  offload        None 2/8, top level vote 3/8, expert-endorsed levels ['Ap', 'Cm', 'Ev', 'Sn']
  renormalize    None 1/8, top level vote 4/8, expert-endorsed levels ['An', 'Ap', 'Cm', 'Sn']

## 2. Matched-acceptance structural comparison

The structural objective penalizes deviation from a target acceptance band, but
acceptance is set by the threshold percentile, not by model quality. Comparing
families at a COMMON acceptance removes that confound.

### At acceptance ~0.80

| family | percentile | acceptance | SCS | CV AUC |
|---|---|---|---|---|
| SGDClassifier | P55 | 0.804 | 0.998 | 0.666 |
| PassiveAggressive | P55 | 0.787 | 0.928 | 0.672 |
| LogisticRegression | P45 | 0.764 | 0.894 | 0.720 |
| KNeighborsClassifier | P55 | 0.779 | 0.876 | 0.713 |
| NearestCentroid | P40 | 0.825 | 0.863 | 0.716 |
| LinearSVC | P50 | 0.798 | 0.849 | 0.701 |
| RidgeClassifier | P45 | 0.802 | 0.712 | 0.713 |

Spearman(SCS at matched acceptance, CV AUC) = -0.414, p = 0.355

### At acceptance ~0.86

| family | percentile | acceptance | SCS | CV AUC |
|---|---|---|---|---|
| SGDClassifier | P50 | 0.870 | 0.933 | 0.666 |
| KNeighborsClassifier | P50 | 0.861 | 0.909 | 0.713 |
| LinearSVC | P45 | 0.856 | 0.899 | 0.701 |
| PassiveAggressive | P50 | 0.852 | 0.882 | 0.672 |
| NearestCentroid | P40 | 0.825 | 0.863 | 0.716 |
| LogisticRegression | P40 | 0.847 | 0.699 | 0.720 |
| RidgeClassifier | P40 | 0.861 | 0.669 | 0.713 |

Spearman(SCS at matched acceptance, CV AUC) = -0.667, p = 0.102

For comparison, the penalized objective (with the acceptance penalty) against CV AUC: Spearman = -0.714, p = 0.071

## 3. Independent-encoder structural check

The structural metrics are computed in the same embedding space the classifier
uses, so they partly measure that geometry rather than the assignments. Recomputing
the ordinal-separation trend in an encoder that played no part in training or
selection gives evidence that is not circular.

- mpnet (used in training): gap means [np.float64(0.1226), np.float64(0.1088), np.float64(0.0992), np.float64(0.1612), np.float64(0.1595)], Spearman rho = 0.50, exact one-tailed p = 0.2250
- MiniLM (independent): gap means [np.float64(0.0739), np.float64(0.0671), np.float64(0.0568), np.float64(0.1046), np.float64(0.0999)], Spearman rho = 0.50, exact one-tailed p = 0.2250
- distilroberta (independent): gap means [np.float64(0.0616), np.float64(0.055), np.float64(0.0446), np.float64(0.0876), np.float64(0.0834)], Spearman rho = 0.50, exact one-tailed p = 0.2250

If the monotone trend survives in encoders that were never used for training or
model selection, the ordinal structure of the extended bank is a property of the
assignments rather than an artifact of the embedding space used to produce them.