# Expert agreement recomputed against the regenerated bank

Sampled verbs matched in the new bank: 29/30.
Not in the extension any more: ['copy'] ('copy' resolved to a core verb under normalization).
Matched but no longer hard-accepted by the new model: ['input', 'authorize', 'rest']
Agreement metrics computed over the 26 hard-accepted matched verbs.

- Strict top-1: 11/26 = 42.3% (old model on 30 verbs: 20.0%)
- Endorsed top-1: 18/26 = 69.2% (old: 46.7%)
- Hard Jaccard/precision/recall: 0.32 / 0.71 / 0.38 (old: 0.17/0.46/0.18)
- Hard+soft: 0.39 / 0.67 / 0.45 (old: 0.31/0.53/0.42)
- Coverage of at least one endorsed level: hard 20/26, hard+soft 20/26 (old: 15/30, 23/30)

- McNemar hard vs hard+soft coverage: discordant 0 vs 0, exact p = 1.0000

- Consensus decomposition: moderate/high consensus 13/14, low consensus 5/12, Fisher exact p = 0.0093
- Moderate/high band vs conditional chance (51%): simulation p = 0.0007

- Score-vote correlation: mean rho = 0.171, permutation p = 0.0140 (old model: 0.144, p = 0.072)