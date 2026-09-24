# Fin-Syn

Anonymous supplement to the submission *Leaderboard Fidelity: What a Synthetic Release Preserves, and How to Measure It*.

## Start here

| What | Where |
|---|---|
| The sixty synthetic releases, and how to read them | [`release/`](release/) and [`release/README.md`](release/README.md) |
| Measured leaderboard fidelity of every release | [`release/manifest.csv`](release/manifest.csv) |
| Datasheet, licenses, upstream generators | [`release/THIRD_PARTY.md`](release/THIRD_PARTY.md) |
| Evaluation protocol as code | [`scripts/`](scripts/) |
| Stored results behind every table and figure | [`exp/finsyn-v2/`](exp/finsyn-v2/) |

The private transfers are not in this repository and will not be: that is the premise of the paper. Everything an outside user of a public benchmark could do is reproducible here.

The pipeline used in the paper is the `*_v2.py` family under `scripts/`, which builds the leakage-free benchmark, runs the generators, computes the leaderboards, and emits the tables and figures. `scripts/build_dataset_v2.py` and `scripts/leaderboard.py` are the entry points.
