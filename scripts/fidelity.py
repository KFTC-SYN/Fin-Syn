"""
리더보드 순위 보존력(leaderboard fidelity) 지표: 실제 리더보드 대비 후보 리더보드.

  1. Kendall τ (PR-AUC 기준) — 실제 데이터 bootstrap τ 상한(noise_floor.json)과 함께 보고
  2. 유의 쌍 순서 보존율 — 실제 데이터에서 95% 수준으로 순서가 갈린 모델 쌍 중 후보에서도 같은 순서인 비율 (주 지표)
  3. 모델 선택 손실 — 후보 1위 모델이 실제 1층(1위와 구분 불가)에 드는지, 실제 PR-AUC 기준 regret

Usage:
    python scripts/fidelity.py --real exp/finsyn-v2/leaderboard_real --cand exp/finsyn-v2/leaderboard_smote_s2r [...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau


def board(d: Path, metric="pr_auc"):
    r = json.loads((d / "results.json").read_text())
    return {m: v["summary"][metric][0] for m, v in r.items()}


def fidelity(real_dir: Path, cand_dir: Path, metric="pr_auc"):
    real, cand = board(real_dir, metric), board(cand_dir, metric)
    noise = json.loads((real_dir / "noise_floor.json").read_text())
    ms = [m for m in real if m in cand]
    tau = kendalltau([real[m] for m in ms], [cand[m] for m in ms]).statistic
    sig = {k: v for k, v in noise["pairwise_win_prob"].items() if v >= 0.975 or v <= 0.025}
    kept = total = 0
    for k, p in sig.items():
        a, b = k.split(">")
        if a in ms and b in ms:
            total += 1
            kept += int((real[a] > real[b]) == (cand[a] > cand[b]))
    best_real = max(ms, key=real.get)
    top_tier = {best_real} | {m for m in ms if m != best_real and (
        0.025 < noise["pairwise_win_prob"].get(f"{best_real}>{m}", 1 - noise["pairwise_win_prob"].get(f"{m}>{best_real}", 0)) < 0.975)}
    pick = max(ms, key=cand.get)
    return {"n_models": len(ms), "kendall_tau": float(tau), "tau_ceiling_real_bootstrap": noise["tau_vs_full_mean"],
            "sig_pair_order_kept": kept / max(total, 1), "n_sig_pairs": total,
            "cand_top1": pick, "cand_top1_in_real_top_tier": pick in top_tier, "real_top_tier": sorted(top_tier),
            "selection_regret_pr_auc": float(real[best_real] - real[pick]),
            "cand_scores": {m: round(cand[m], 4) for m in sorted(ms, key=cand.get, reverse=True)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--metric", default="pr_auc")
    args = ap.parse_args()
    for c in args.cand:
        f = fidelity(Path(args.real), Path(c), args.metric)
        (Path(c) / f"fidelity_vs_{Path(args.real).name}.json").write_text(json.dumps(f, indent=1))
        print(f"{Path(c).name:32s} τ={f['kendall_tau']:.3f} (ceiling {f['tau_ceiling_real_bootstrap']:.3f})  "
              f"sig-pairs kept {f['sig_pair_order_kept']:.2%} of {f['n_sig_pairs']}  top1={f['cand_top1']} "
              f"in-real-top-tier={f['cand_top1_in_real_top_tier']}  regret={f['selection_regret_pr_auc']:.3f}")


if __name__ == "__main__":
    main()
