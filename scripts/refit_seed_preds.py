"""
기준 리더보드의 시드별 평가 예측을 저장한다(점수 정의 통일용, 9/23).

leaderboard.py는 요약 점수를 '시드별 PR-AUC의 평균'으로 내면서 부트스트랩 노이즈와 감사 실험에는
'시드 평균 예측의 PR-AUC'를 썼다. 두 정의를 하나로 맞추려면 시드별 예측이 필요한데 평균만 저장돼 있었다.
여기서는 results.json에 저장된 튜닝 결과(best_params)로 같은 시드를 다시 학습해
pred_test_<탐지기>_seed<s>.npy를 쓰고, 다시 잰 시드별 PR-AUC가 저장값과 같은지 확인한다.
튜닝은 다시 하지 않으므로 리더보드 자체는 바뀌지 않는다.

Usage:
    python scripts/refit_seed_preds.py --train data/finsyn-v2 --out exp/finsyn-v2/leaderboard_real
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leaderboard import Prep, fit_predict, load_split, space  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--tune", default=None)
    ap.add_argument("--test", default=None, help="dir[:split], default <train>:test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=None)
    args = ap.parse_args()

    Xtr, ytr, info = load_split(args.train, "train")
    Xva, yva, _ = load_split(args.tune or args.train, "val")
    Xte, yte, _ = load_split(args.test or args.train, "test")
    prep = Prep(info, Xtr)
    out = Path(args.out)
    res = json.loads((out / "results.json").read_text())
    models = args.models.split(",") if args.models else list(res)
    check = {}
    for name in models:
        r = res[name]
        seeds = [run["seed"] for run in r["runs"]]
        if all((out / f"pred_test_{name}_seed{s}.npy").exists() for s in seeds):
            print(f"skip {name}", flush=True)
            continue
        t0 = time.time()
        assert r["best_params"] is not None and not r.get("degenerate", False), name
        best = space(name, optuna.trial.FixedTrial(r["best_params"]))
        diffs = []
        for run in r["runs"]:
            s = run["seed"]
            _, pt = fit_predict(name, best, prep, Xtr, ytr, Xva, yva, Xte, seed=s)
            np.save(out / f"pred_test_{name}_seed{s}.npy", pt)
            diffs.append(abs(average_precision_score(yte, pt) - run["pr_auc"]))
        check[name] = max(diffs)
        print(f"{name:9s} seeds {seeds}  max |PR-AUC - stored| {max(diffs):.2e}  ({(time.time() - t0) / 60:.1f} min)",
              flush=True)
    prev = out / "refit_check.json"
    old = json.loads(prev.read_text()) if prev.exists() else {}
    prev.write_text(json.dumps({**old, **check}, indent=1))


if __name__ == "__main__":
    main()
