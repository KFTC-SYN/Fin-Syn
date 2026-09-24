"""
원천 추출본이 '송금 계좌 패널'이라는 점이 벤치마크에 주는 영향을 수치로 남긴다(본문 3.2절).

  - 패널 크기: 송금 계좌 수, 수취 계좌 수, 이체 수, 플래그 비율
  - 표본 성격: 플래그 이체를 1건 이상 보낸 송금 계좌의 비율 (무작위 표본이면 0.43% 기저율에서 이렇게 높을 수 없다)
  - 수취 이력의 절단: 수취 계좌 중 패널에 속한 계좌의 비율, 그리고 벤치마크 행 중
    수취 계좌의 과거 입금 건수(r_n_prev_in)가 같은 송금 계좌의 과거 송금 건수(pair_n_prev)와 같은 행의 비율
    → 같으면 수취 이력은 그 송금인 자신의 이력일 뿐, 다른 송금인의 정보는 없다.

Usage:
    python scripts/panel_stats_v2.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/finsyn-v2"


def main():
    o = pd.read_parquet(ROOT / "_datasets/panel.parquet")
    y = o["이상거래여부"].notna().astype(int)
    payer = o["출금금융회사일련번호"].astype(str) + "_" + o["출금계좌일련번호"].astype(str)
    payee = o["입금금융회사일련번호"].astype(str) + "_" + o["입금계좌일련번호"].astype(str)
    flagged_by_payer = y.groupby(payer).sum()
    receivers = pd.Series(payee.unique())

    info = json.loads((D / "info.json").read_text())
    X = pd.DataFrame(np.concatenate([np.load(D / f"X_num_{s}.npy") for s in ("train", "val", "test")]),
                     columns=info["num_cols"])
    same = (X["r_n_prev_in"] == X["pair_n_prev"]).to_numpy()

    res = {
        "transfers": int(len(o)),
        "flag_rate": float(y.mean()),
        "sending_accounts": int(payer.nunique()),
        "receiving_accounts": int(len(receivers)),
        "share_senders_with_flag": float((flagged_by_payer > 0).mean()),
        "share_receivers_in_panel": float(receivers.isin(set(payer)).mean()),
        "share_rows_receiver_history_is_pair_history": float(same.mean()),
    }
    out = ROOT / "exp/finsyn-v2/panel_stats.json"
    out.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"{k:45s} {v}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
