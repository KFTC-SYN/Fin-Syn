# Fin-Syn: sixty synthetic releases of a private interbank fraud benchmark

Anonymous supplement to the submission *Leaderboard Fidelity: What a Synthetic Release Preserves, and How to Measure It*.

This repository contains the sixty synthetic releases the paper evaluates, the evaluation protocol as code, and the stored results from which every table and figure in the paper is generated.

The private transfers are **not** here, and cannot be: that is the premise of the paper rather than an omission. What you can re-run from this repository is everything an outside user of a public benchmark could do. What requires the private data is the reference leaderboard, the privacy calibration, and the augmentation experiment.

## What is a release

A *release* is one draw from one generator: three synthetic datasets, one per temporal split, produced by a generator trained only on the corresponding private split and sampled to the same size. Twelve generators times five seeds gives sixty releases.

```
data/<generator>/seed<k>/{train,val,test}.parquet
```

| | rows |
|---|---|
| `train.parquet` | 63,707 (September 2021 to December 2023) |
| `val.parquet` | 15,427 (January to June 2024) |
| `test.parquet` | 15,767 (July to December 2024) |

Generators: `smote`, `tabsyn`, `tabdiff`, `findiff`, `tabpfgen`, `tabpfgen-prior`, `tabddpm`, `tvae`, `great`, `ctgan`, `ctabgan`, `ctabgan-plus`. Seeds 0 to 4.

Some releases contain fewer rows than the split they imitate, because rows identical to a private record are deleted before publication (see *Privacy* below).

## Schema

25 columns: 18 numeric features, 6 categorical features, and the label.

| Group | Columns |
|---|---|
| Transaction | `amount`, `hour_band`, `dow`, `media_type`, `fund_type` |
| Bank | `withdraw_bank`, `deposit_bank`, `bankpair_n_30d` |
| Sender history | `s_n_prev`, `s_n_7d`, `s_n_30d`, `s_amt_sum_30d`, `s_amt_mean_prev`, `s_amt_max_prev`, `s_hours_since_last`, `s_days_since_first`, `s_n_payees_prev`, `s_n_dbanks_prev`, `pair_n_prev` |
| Receiver cross-bank history | `r_n_prev_in`, `r_n_in_30d`, `r_n_payers_prev`, `r_n_sendbanks_prev`, `r_days_since_first` |
| Label | `label`, 1 if a bank flagged the transfer as suspicious |

There is no identifier column. History features are counts and amounts over transfers strictly earlier than the one described; `-1` marks a value that is undefined because the account has no prior activity. Categorical columns are integer codes stored as strings; the bank codes are consistent within the benchmark but carry no institutional meaning.

## manifest.csv

One row per release, with its row counts, its positive rate, and the two rank agreements the paper measures against the private leaderboard:

- `tau_s2s`: Kendall tau between the leaderboard obtained by training, tuning, and testing on the release, and the private reference leaderboard. This is what an outside user's leaderboard is worth. It ranges from -0.309 to 0.927 across the sixty releases.
- `tau_s2r`: the same for the TSTR leaderboard, which trains and tunes on the release but tests on the private test period.

Read `tau_s2s` against the noise scale of the private data: resampling the private test period alone moves the ranking by 0.910 on average, with a 5th percentile of 0.818.

## Reproducing the public-user leaderboard

```bash
python ../scripts/leaderboard.py \
  --train release/data/tabsyn/seed0 \
  --tune  release/data/tabsyn/seed0:val \
  --test  release/data/tabsyn/seed0:test \
  --out   /tmp/lb_tabsyn_seed0 --models all --trials 20 --seeds 5
```

Eleven detectors, each tuned with 20 trials of TPE search maximizing PR-AUC on the validation split, then retrained with five seeds; the reported score is the seed mean. Comparing the resulting ordering with `../exp/finsyn-v2/leaderboard_real/results.json` reproduces `tau_s2s` for that release.

## Results

`../exp/finsyn-v2/` holds the stored outputs the paper's tables and figures are generated from, including `lf_analysis.json` (leaderboard fidelity per release, variance decomposition, correlations), `within_generator.json`, `release_selection.json`, `standard_metrics.json`, `privacy.json`, and the per-detector leaderboards under `../exp/finsyn-v2/leaderboard_*/`.

## Privacy

Every synthetic row identical to a private training record is deleted before publication. This affected all fifteen SMOTE splits, about 1% of each and none of them flagged, three TabDDPM runs, and two GReaT runs. All results in the paper are computed on the filtered releases.

The SMOTE releases sit closer to the private data than a fresh real sample does, with a distance ratio of 0.42 and a nearest-neighbour membership-inference AUC of 0.584. We disclose this rather than presenting the releases as risk-free; Section 5.5 and the Ethics Statement of the paper give the measurements.

## Third-party code

The generators are run from their official implementations at the commits and configurations documented in Appendix C of the paper. This repository ships our evaluation code under `../scripts/`, not those implementations; `THIRD_PARTY.md` lists each upstream repository, the commit we used, its license, and the deviations we applied.
