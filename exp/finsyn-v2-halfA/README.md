# Audit half excluded from synthesis (external review W2)

The audit in Section 5.4 selects a release on one half (A) of the private test period and scores it on the other (B). In the main experiment the synthetic test split of every release was generated from the whole test period, so B was never used for selection but was seen by the generator. This experiment removes that overlap: every release's synthetic test split is regenerated from half A only, and selection on A is then scored on B.

## Design

- **Split.** The first split of `StratifiedShuffleSplit(n_splits=200, test_size=0.5, random_state=0)` on the private test labels, the same generator the audit in `scripts/release_selection_v2.py` uses. A has 7,883 transfers (87 positive) and B has 7,884 (88 positive). The indices are in `split_idx_A.npy` and `split_idx_B.npy`.
- **Generation.** Every generator and seed is rerun on A with the original test-split configuration. Only the output size changes, to |A| (twice that for TabPFGen-prior, as before). The configurations are in `gen/<model>/test[_seed<k>]/config.toml`. The synthetic train and validation splits of each release are unchanged.
- **Post-processing.** The same publication filter as the releases removes synthetic rows identical to any private record; only SMOTE produced such rows (20 to 30 per release). TabPFGen-prior is subsampled to the positive rate of A.
- **Leaderboards.** Each release's detectors are refit from their stored S->S tuning (seeds 0 to 4) on the release's own train and validation splits, then scored on the new test split.
- **Analysis.** The reference leaderboard is computed on A and on B (mean over seeds of per-seed PR-AUC). The release with the highest tau on A is scored on B, within each generator and among all sixty, next to the same computation with the original synthetic test splits.

## Where it ran

- **SMOTE** was generated on the project workstation (CPU).
- **The other eleven generators** were generated on a separate server with three NVIDIA RTX A6000 GPUs. Its environment is in `server_env/`: Python 3.11.15, torch 2.14.0+cu126, and the same package versions as the workstation (`pip_freeze.txt`). The server ran the public repository at the commit in `repo_commit.txt`, plus the CTAB-GAN column entries in `columns_patch.diff`.
- **What went to the server.** Only the identifier-free feature arrays of half A. B, the training and validation periods, and the identifier file were not copied.
- **Everything after generation** (assembly, filtering, leaderboards, analysis) ran on the workstation, in the environment that produced the main results.

## Files

| Path | Contents |
|---|---|
| `split_idx_A.npy`, `split_idx_B.npy` | Row indices of the two halves in the private test split |
| `gen/<model>/test[_seed<k>]/` | Generator output (`X_num_train.npy`, `X_cat_train.npy`, `y_train.npy`), configuration, and `run.log` from the server |
| `gen_runs.jsonl` | One record per generation run: return code, minutes, device |
| `synth_test/<model>/seed<k>/` | Assembled and filtered synthetic test split used for the leaderboards |
| `assemble_report.json` | Rows, removed copies, and positive rate per release |
| `boards/<model>_seed<k>.json` | Per-detector PR-AUC of each release on its new test split |
| `halfA_analysis.json` | Selection results, B-excluded and original |
| `logs/server/` | Per-job driver logs from the server; `logs/board_*.txt` are the workstation leaderboard logs |
| `server_env/` | Server hardware, Python and package versions, driver and install logs |

Generated arrays, logs and the server's model checkpoints are not committed (`.gitignore`). The checkpoints were not copied back.

## Rerunning

```bash
# on a GPU machine with the public repository and half A in data/finsyn-v2-halfA-part-test/
python scripts/halfA_driver.py --gpus 0,1,2 --per-gpu 2
# SMOTE, on the workstation
python scripts/synth_v2.py --model smote --real data/finsyn-v2-halfA-part-test --out <tmp> --seed <k>
# on the workstation
python scripts/halfA_analysis_v2.py assemble
python scripts/halfA_analysis_v2.py leaderboards --jobs 5
python scripts/halfA_analysis_v2.py analyse
```

## Incidents during the run

- **Missing sources.** TabSyn's `vae/` sources and TabDiff's `eval/` sources were missing from the public repository because of `.gitignore` rules, so the first TabSyn runs failed. They were restored (repository commit `a06c48d`) and TabSyn was rerun from scratch.
- **CPU oversubscription.** The first launch let every job use all cores. It was stopped after five minutes and relaunched with two threads per job.
- **GReaT on three GPUs.** GReaT's HuggingFace Trainer used all three visible GPUs, which triples the effective batch size relative to the single-GPU releases. Those runs were discarded. `scripts/halfA_driver.py` now exposes one GPU to every generator that is not a TabSyn, TabDiff or FinDiff wrapper, and GReaT was rerun.
