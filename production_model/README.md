# Production model

This folder holds the model that produced the published verb bank, packaged so it can be
used on its own without running the training pipeline.

The model is a one-vs-rest k-nearest-neighbour classifier (k = 21, cosine distance,
distance weighting) trained on 381 core Bloom verbs carrying 607 verb-level assignments,
with per-level thresholds set at the 50th percentile of the calibrated core scores.
Verbs are represented by `all-mpnet-base-v2` sentence embeddings (768 dimensions,
L2 normalised).

## Files

| File | What it is |
| --- | --- |
| `model.pkl` | the fitted one-vs-rest classifier (joblib) |
| `calibration_cdfs.json` | per-level empirical CDFs that map raw scores to [0, 1] |
| `thresholds.json` | per-level acceptance thresholds (P50 of the calibrated core scores) |
| `row_weights.npy` | source-frequency weights of the core verbs, used for the weighted threshold percentiles (the classifier itself is fit unweighted) |
| `metadata.json` | model parameters, level order, provenance |
| `bt4_scorer.py` | loader and scoring class used by the pipeline |
| `classify.py` | command line tool for labelling new verbs |

## Quick use

```
pip install -r ../requirements.txt
python classify.py weld debug summarise appraise
```

Output is one line per verb with the primary level, the accepted levels and the
calibrated score for each of the six levels:

```
verb       primary  accepted     score_Kn  score_Cm  score_Ap  score_An  score_Sn  score_Ev
weld       Ap       Ap           0.267     0.34      0.871     0.195     0.552     0.481
debug      Sn       Sn           0.149     0.026     0.003     0.003     0.997     0.008
summarise  Cm       Cm An Sn Ev  0.33      0.997     0.003     0.977     0.992     0.997
appraise   Ap       Ap An Ev     0.149     0.026     0.997     0.997     0.003     0.997
```

A level is accepted when its calibrated score reaches the level's threshold. As in the
published verb bank, when two or more levels pass and the highest score leads the second
by at least 0.060, only the highest level is kept; `--no-margin` lists every level that
passes. With the margin, `classify.py` reproduces the hard labels, primary levels and soft
suggestions of `results/Verb List Classified by Model.csv` for all 1,650 candidate verbs.
A verb with no level above its threshold is reported as `None`, which is the model
declining to place it rather than a failure. Add `--soft` to also show the wider
suggestion tier (thresholds scaled by 0.85), which trades precision for recall.

To label a file of verbs, one per line:

```
python classify.py --file my_verbs.txt --out labelled.csv
```

## Using it from Python

```python
from bt4_scorer import BT4Scorer
from sentence_transformers import SentenceTransformer
import numpy as np

enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
X = enc.encode(["weld", "debug"], normalize_embeddings=True)

scorer = BT4Scorer(".")
Y, scores = scorer.predict_binary(X)      # levels that pass their thresholds
Ysoft, _ = scorer.suggest_binary(X, 0.85) # soft suggestions
```

`Y` is an n x 6 binary matrix over the levels in `metadata.json` order
(Kn, Cm, Ap, An, Sn, Ev) and `scores` holds the calibrated scores. `predict_binary` does
not apply the top-2 margin; `classify.apply_top2_margin(scores, Y)` does, and gives the
hard labels of the published bank.

## What to expect from it

On held-out core verbs (5-fold cross-validation repeated with five random splits) the
acceptance gate keeps 93.3 per cent of known Bloom verbs, and 74.2 per cent of the primary
predictions of accepted verbs lie within one level of a published level. Against
an eight-expert panel the primary level fell inside the expert-endorsed set for 69.2 per
cent of accepted verbs. Verb-to-level mapping depends partly on context, so treat the
output as a ranked prior to be read alongside the full learning outcome, not as a
context-free verdict. The precision side of the accept/reject decision has not been
validated against labelled negatives, so unassigned verbs are best sent to human review.

The model is stored as a pickle, which means it is tied to the scikit-learn version in
`requirements.txt`. If you load it under a different version and see a warning, retrain
with `revision_pipeline/retrain.py` instead of trusting the loaded object.
