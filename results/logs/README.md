# Logs of the documented reproduction run

The README reproduction sequence was run end to end on 24 and 25 October 2026 on a copy of the
repository without its git history and without `Papers & Books Used for Extraction/`, with
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`: Windows 10, 8 cores, Python 3.12.10, numpy 2.5.1,
pandas 3.0.5, scipy 1.18.0, scikit-learn 1.9.0, torch 2.13.0+cpu, sentence-transformers 5.6.1.

- `01_normalize.log` to `19_paper_results.log`: standard output and error of each step of the
  documented run, which did not include `change_attribution.py` (README step 19, run later);
  `paper_results.py` is therefore step 19 here and step 20 in the README. `run_timings.jsonl`:
  command, start, end, runtime and exit code of each step (all exit codes 0; total 3,629.6 s).
- `check_supervised_eval_encoders_reencode.log`: the encoder-ablation step rerun with
  `--reencode` (vectors re-encoded from the cached encoders; identical vectors and results).
- `check_bank_structure_with_git_history.log`, `check_paper_results_with_git_history.log`:
  `bank_structure.py` and `paper_results.py` (steps 15 and 19 of the documented run) rerun with
  the git history available, which restores the original-submission column of the bank
  comparison (outputs byte-identical to the files committed at the time of the run).

Machine-specific paths are shortened: the folder the run used as repository root is written
`<repo>` and the user's home folder `~`. Nothing else is edited. The wording of some reports was
revised after this run, in the scripts and in the committed reports together, with no number
changed, so a few logs (for example `16_expert_recompute.log`) print the earlier wording. The many `FutureWarning` lines
are scikit-learn 1.9.0 deprecation notices (`penalty` of LogisticRegression,
`PassiveAggressiveClassifier`, `SVC(probability=True)`); they do not change any result.
