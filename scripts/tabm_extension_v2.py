"""
TabM(12번째 탐지기) 견고성 점검: 본문의 11개 탐지기 결과에 TabM을 더해도 결론이 유지되는가(부록용).

입력
  exp/finsyn-v2/<리더보드>/            : 기존 11개 탐지기 결과(본문)
  exp/finsyn-v2/tabm/<리더보드>/       : 같은 설정의 TabM 결과(leaderboard.py --models tabm)
처리
  두 결과를 exp/finsyn-v2/tabm12/<리더보드>/ 에 합치고(원본은 건드리지 않음), 실제 리더보드는 12개 기준
  bootstrap 노이즈를 다시 계산한 뒤 fidelity.py와 같은 함수로 11개 기준/12개 기준 지표를 나란히 계산한다.
  대상은 시드 0 릴리스(부록 범위). 추가로 'TabM 대 최고 GBDT'의 우열이 릴리스에서 보존되는지 본다
  — 공개 벤치마크의 대표적 용도(딥러닝 대 트리 비교)를 릴리스가 보존하는지에 대한 직접 점검.

Usage:
    python scripts/tabm_extension_v2.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fidelity import fidelity  # noqa: E402
from leaderboard import bootstrap_noise, load_split  # noqa: E402

E = ROOT / "exp/finsyn-v2"
T, M = E / "tabm", E / "tabm12"
RELEASES = ["smote", "tabpfgen", "tabpfgen-prior", "tabddpm", "great", "tvae", "ctgan", "ctabgan", "ctabgan-plus",
            "tabsyn", "tabdiff", "findiff"]
GBDT = ["lgbm", "xgb", "catboost", "hgb"]


def merge(name):
    base, ext, out = E / name, T / name, M / name
    if not ((base / "results.json").exists() and (ext / "results.json").exists()):
        return None
    out.mkdir(parents=True, exist_ok=True)
    res = json.loads((base / "results.json").read_text())
    res["tabm"] = json.loads((ext / "results.json").read_text())["tabm"]
    (out / "results.json").write_text(json.dumps(res, indent=1))
    for d in (base, ext):
        for f in d.glob("pred_test_*.npy"):
            shutil.copyfile(f, out / f.name)
    return out


def main():
    real = merge("leaderboard_real")
    assert real is not None, "TabM real leaderboard missing"
    _, yte, _ = load_split(str(ROOT / "data/finsyn-v2"), "test")
    res = json.loads((real / "results.json").read_text())
    preds = {m: np.load(real / f"pred_test_{m}.npy") for m in res}
    noise = bootstrap_noise(yte, preds)
    (real / "noise_floor.json").write_text(json.dumps(noise, indent=1))
    ref = {m: v["summary"]["pr_auc"][0] for m, v in res.items()}
    order = sorted(ref, key=ref.get, reverse=True)
    sep = sum(1 for v in noise["pairwise_win_prob"].values() if v >= 0.975 or v <= 0.025)
    out = {"reference": {"pr_auc": {m: round(ref[m], 4) for m in order}, "tabm_rank": order.index("tabm") + 1,
                         "noise_ceiling_mean": noise["tau_vs_full_mean"], "noise_band_q05": noise["tau_vs_full_q05"],
                         "separable_pairs": sep, "n_pairs": len(noise["pairwise_win_prob"])},
           "releases": {}}
    best_gbdt_real = max(GBDT, key=ref.get)
    for m in RELEASES:
        for tag in ("s2s", "s2r"):
            name = f"leaderboard_{m}_{tag}"
            merged = merge(name)
            if merged is None:
                continue
            f11 = fidelity(E / "leaderboard_real", E / name)
            f12 = fidelity(real, merged)
            cand = f12["cand_scores"]
            best_gbdt_c = max(GBDT, key=lambda g: cand[g])
            out["releases"][f"{m}|{tag}"] = {
                "tau11": f11["kendall_tau"], "tau12": f12["kendall_tau"],
                "pairs11": f11["sig_pair_order_kept"], "pairs12": f12["sig_pair_order_kept"],
                "regret11": f11["selection_regret_pr_auc"], "regret12": f12["selection_regret_pr_auc"],
                "top1_11": f11["cand_top1"], "top1_12": f12["cand_top1"],
                "tabm_rank_release": sorted(cand, key=cand.get, reverse=True).index("tabm") + 1,
                "tabm_minus_best_gbdt_real": ref["tabm"] - ref[best_gbdt_real],
                "tabm_minus_best_gbdt_release": cand["tabm"] - cand[best_gbdt_c]}
    for tag in ("s2s", "s2r"):
        rows = [v for k, v in out["releases"].items() if k.endswith(tag)]
        if len(rows) >= 3:
            t11, t12 = [r["tau11"] for r in rows], [r["tau12"] for r in rows]
            out[f"agreement_{tag}"] = {"n": len(rows), "spearman_tau11_tau12": float(spearmanr(t11, t12).statistic),
                                       "kendall_tau11_tau12": float(kendalltau(t11, t12).statistic),
                                       "max_abs_change": float(np.max(np.abs(np.subtract(t12, t11))))}
    (E / "tabm_extension.json").write_text(json.dumps(out, indent=1))
    r = out["reference"]
    print(f"reference 12: TabM rank {r['tabm_rank']} (PR-AUC {r['pr_auc']['tabm']}); ceiling {r['noise_ceiling_mean']:.3f} "
          f"band {r['noise_band_q05']:.3f}; separable {r['separable_pairs']}/{r['n_pairs']}")
    for k, v in out["releases"].items():
        print(f"  {k:22s} tau {v['tau11']:+.3f} -> {v['tau12']:+.3f}  pairs {v['pairs11']:.2f}->{v['pairs12']:.2f}  "
              f"TabM rank {v['tabm_rank_release']:2d}  TabM-GBDT real {v['tabm_minus_best_gbdt_real']:+.3f} "
              f"release {v['tabm_minus_best_gbdt_release']:+.3f}")
    for tag in ("s2s", "s2r"):
        if f"agreement_{tag}" in out:
            print(tag, out[f"agreement_{tag}"])


if __name__ == "__main__":
    main()
