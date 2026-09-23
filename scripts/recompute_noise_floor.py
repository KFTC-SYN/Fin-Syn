"""
기준 리더보드의 부트스트랩 노이즈를 시드별 예측으로 다시 계산한다(9/23 점수 정의 통일).

이전 noise_floor.json은 시드 평균 예측의 PR-AUC로 계산했다. 요약 리더보드(시드별 PR-AUC의 평균)와 같은 정의로
다시 내고, 이전 파일은 noise_floor_meanpred.json으로 남긴다. 시드 간 τ(seed_tau_mean)는 정의와 무관해 그대로 둔다.

Usage:
    python scripts/recompute_noise_floor.py --test data/finsyn-v2 --out exp/finsyn-v2/leaderboard_real
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leaderboard import bootstrap_noise, load_split, seed_preds  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--test", required=True, help="dir[:split], default split test")
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)
_, y, _ = load_split(a.test, "test")
res = json.loads((out / "results.json").read_text())
preds = {m: seed_preds(out, m) for m in res}
assert all(p.ndim == 2 for p in preds.values()), "시드별 예측이 없다: refit_seed_preds.py 먼저"
old = json.loads((out / "noise_floor.json").read_text())
if not (out / "noise_floor_meanpred.json").exists():
    (out / "noise_floor_meanpred.json").write_text(json.dumps(old, indent=1))
new = bootstrap_noise(y, preds)
new["seed_tau_mean"] = old.get("seed_tau_mean")
new["score_definition"] = "mean over seeds of per-seed PR-AUC"
(out / "noise_floor.json").write_text(json.dumps(new, indent=1))
sep = lambda n: sum(1 for v in n["pairwise_win_prob"].values() if v >= 0.975 or v <= 0.025)
print(f"{out}: mean {old['tau_vs_full_mean']:.3f} -> {new['tau_vs_full_mean']:.3f}, "
      f"q05 {old['tau_vs_full_q05']:.3f} -> {new['tau_vs_full_q05']:.3f}, separable {sep(old)} -> {sep(new)}")
