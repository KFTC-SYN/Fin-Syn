# Third-party generator implementations

Every generator is run from its official implementation. Four of them (TabSyn, TabDiff, FinDiff, TabM) are included in this repository at the commits we ran, under their own licenses and with their license files intact, so that the pipeline is runnable as published; only their source is included, not the checkpoints and demo assets they ship. The remaining implementations are vendored from earlier work in this repository. The table records what we ran and what we changed; Appendix C of the paper gives the full configuration of each generator.

| Generator | Upstream implementation | Version we ran | License as stated upstream |
|---|---|---|---|
| SMOTE | `imbalanced-learn` (SMOTENC) | release used in `requirements.txt` | MIT |
| CTGAN, TVAE | `sdv-dev/CTGAN` | vendored copy | MIT |
| CTAB-GAN | `Team-TUD/CTAB-GAN` | vendored copy | Apache 2.0 |
| CTAB-GAN+ | `Team-TUD/CTAB-GAN-Plus` | vendored copy | not stated in the copy we obtained |
| TabDDPM | `rotot0/tab-ddpm` | vendored copy | not stated in the copy we obtained |
| GReaT | `kathrinse/be_great` | vendored copy | MIT |
| TabPFGen | `carte-lab/TabPFGen` | vendored copy | Apache 2.0 |
| TabSyn | `amazon-science/tabsyn`, included at [`tabsyn/`](../tabsyn/) | commit `cb5ac0f` | Apache 2.0 |
| TabDiff | `MinkaiXu/TabDiff`, included at [`TabDiff/`](../TabDiff/) | commit `5ecdb33` | MIT-style, Copyright 2024 Minkai Xu |
| FinDiff | `sattarov/FinDiff`, included at [`FinDiff/`](../FinDiff/) | commit `45e9563` | MIT, Copyright 2024 Timur Sattarov |
| TabM (12th detector, Appendix G) | `yandex-research/tabm`, included at [`tabm/`](../tabm/) | commit `28e47ae3` | Apache 2.0 |

Two entries above, CTAB-GAN+ and TabDDPM, have no license file in the copy we obtained. We make no claim about the terms under which they may be redistributed, and a user reproducing our results with those two should obtain them from their own repositories.

The four included implementations are unmodified. Our changes are applied at load time by `scripts/tabgen/patch.py`, which rewrites named lines of the upstream source in memory and fails loudly if a target line is missing, so that the source files here remain identical to those commits.

## Deviations we applied

Each of these is a judgement call another author might make differently, so we record it rather than folding it into the defaults.

- **Training budget.** Where the defaults would not finish in reasonable time we fixed a budget of about three GPU-hours per split and set the epoch or step count from the method's measured per-epoch cost: GReaT trains 14 epochs instead of the default 100, TabSyn 1,300 VAE and 2,500 diffusion epochs instead of 4,000 and 10,001 on the training split, and TabDiff 1,100 steps instead of 8,000. The budget bounds training only, not sampling.
- **CTAB-GAN.** Trained for 150 rather than the code default of 10 epochs, which does not converge.
- **GReaT.** The reference pipeline passes column names as the strings `"0"` to `"23"`, under which the model produced no parseable row. Passing the real feature names instead yields a valid-row rate above 99.99%.
- **Seeding.** Four implementations (CTGAN, TVAE, CTAB-GAN, CTAB-GAN+) seed only their sampling step. Our wrappers (`train_sample_*.py`) additionally seed Python, NumPy, and PyTorch at the start of training. `scripts/check_generator_seeding.py` trains each generator for two epochs on 2,000 rows, every run in a fresh process, and compares numeric features, categorical features, and labels. Two same-seed runs are identical on the CPU for all four and on the GPU for CTGAN and TVAE (`exp/generator_seeding_check_cpu.json`, `exp/generator_seeding_check_gpu.json`). On the GPU, CTAB-GAN and CTAB-GAN+ reproduce categorical features and labels but not numeric features (72-77% of values differ, by at most 0.003 standard deviations); with cuDNN's deterministic mode they are identical (`exp/generator_seeding_check_gpu_cudnn_deterministic.json`). Our released CTAB-GAN and CTAB-GAN+ data were trained on the GPU without that mode, so rerunning them from their seed need not reproduce them bit for bit; we did not check reproducibility at full training length, which is why we publish the data themselves and the spread across seeds.
- **TabPFGen.** One-line fix in the released code, documented in Appendix C. The `tabpfgen-prior` variant is ours: the generator runs at its default class balancing, and we subsample the output to the private class ratio.

## Our code

Everything under `scripts/` is ours and is released under the MIT License, except where a file states otherwise.

## The releases

The synthetic datasets under `data/` are released under CC BY 4.0. They contain no record of any real person or account: every row is generated, and rows identical to a private record are deleted before publication.
