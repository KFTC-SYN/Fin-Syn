"""
Fin-Syn v2 데이터셋 구축: 벤치마크 표본 행 + 전체 이체 패널에서 계산한 과거-only 계좌 이력 피처, 시간 기준 분할.

초기 구성은 고유 2,157행을 91,005행으로 복제한 뒤 무작위 분할해 test의 99.65%가 train과 겹쳤다(논문 5.6절).
v2 원칙
  - 행: _datasets/sample.parquet (벤치마크 표본, 완전 중복 제거, 복제 없음)
  - 이력 피처: _datasets/panel.parquet 전체에서, 해당 거래의 (거래일자, 거래시간대)보다 엄격히 이전 거래만 사용
    (같은 날·같은 시간대는 선후를 알 수 없으므로 제외). 레이블 유래 피처는 만들지 않는다.
  - 계좌 ID·거래일자 값 자체는 피처에서 제외
  - 분할: train 2021-09~2023-12 / val 2024-01~06 / test 2024-07~12

Usage:
    python scripts/build_dataset_v2.py --out data/finsyn-v2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = ["거래일자", "거래시간대", "출금금융회사일련번호", "출금계좌일련번호", "입금금융회사일련번호",
       "입금계좌일련번호", "거래금액", "매체구분", "자금구분"]
DAY_H = 24


def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.drop_duplicates(subset=RAW + ["이상거래여부"]).reset_index(drop=True)
    for c in RAW:
        if c != "거래금액":
            df[c] = df[c].astype(str)
    df["거래금액"] = df["거래금액"].astype(float)
    df["y"] = df["이상거래여부"].notna().astype(np.int64)
    date = pd.to_datetime(df["거래일자"], format="%Y%m%d")
    df["t"] = ((date - pd.Timestamp("2021-01-01")).dt.days * DAY_H + df["거래시간대"].astype(int)).astype(np.int64)
    df["payer"] = df["출금금융회사일련번호"] + "_" + df["출금계좌일련번호"]
    df["payee"] = df["입금금융회사일련번호"] + "_" + df["입금계좌일련번호"]
    df["bankpair"] = df["출금금융회사일련번호"] + ">" + df["입금금융회사일련번호"]
    df["pair"] = df["payer"] + "|" + df["payee"]
    return df


class History:
    """한 키(계좌 등)에 대한 이벤트 시각 배열. 질의 시각보다 엄격히 이전인 이벤트만 센다."""

    SHIFT = 1 << 32  # t(시간 단위) < 2^32

    def __init__(self, key_ids: np.ndarray, t: np.ndarray, amount: np.ndarray | None = None):
        comp = key_ids.astype(np.int64) * self.SHIFT + t
        order = np.argsort(comp, kind="stable")
        self.comp = comp[order]
        self.t = t[order]
        self.key = key_ids[order]
        if amount is not None:
            a = amount[order]
            self.csum = np.concatenate([[0.0], np.cumsum(a)])
            # 키 내부 누적 최대값
            s = pd.Series(a)
            self.cmax = s.groupby(self.key).cummax().to_numpy()

    def left(self, qkey, qt):
        return np.searchsorted(self.comp, qkey.astype(np.int64) * self.SHIFT + qt, side="left")

    def count_before(self, qkey, qt, window_h=None):
        hi = self.left(qkey, qt)
        if window_h is None:
            lo = np.searchsorted(self.comp, qkey.astype(np.int64) * self.SHIFT, side="left")
        else:
            lo = self.left(qkey, np.maximum(qt - window_h, 0))
        return hi - lo, lo, hi

    def sum_before(self, qkey, qt, window_h=None):
        n, lo, hi = self.count_before(qkey, qt, window_h)
        return self.csum[hi] - self.csum[lo], n

    def last_before(self, qkey, qt):
        hi = self.left(qkey, qt)
        idx = hi - 1
        ok = (idx >= 0) & (self.key[np.clip(idx, 0, None)] == qkey)
        return np.where(ok, self.t[np.clip(idx, 0, None)], -1), np.where(ok, idx, -1)

    def first_time(self, qkey):
        lo = np.searchsorted(self.comp, qkey.astype(np.int64) * self.SHIFT, side="left")
        ok = (lo < len(self.comp)) & (self.key[np.clip(lo, 0, len(self.key) - 1)] == qkey)
        return np.where(ok, self.t[np.clip(lo, 0, len(self.t) - 1)], -1)


def distinct_before(ref_owner, ref_other, ref_t, q_owner, q_t):
    """owner별로, 질의 시각 이전에 처음 등장한 서로 다른 other의 개수."""
    first = pd.DataFrame({"o": ref_owner, "x": ref_other, "t": ref_t}).groupby(["o", "x"], sort=False)["t"].min()
    h = History(first.index.get_level_values(0).to_numpy(), first.to_numpy())
    n, _, _ = h.count_before(q_owner, q_t)
    return n


def build_features(panel: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    ids = {}
    for k in ["payer", "payee", "bankpair", "pair", "입금금융회사일련번호", "출금금융회사일련번호"]:
        codes, uniq = pd.factorize(pd.concat([panel[k], rows[k]], ignore_index=True))
        ids[k] = (codes[: len(panel)], codes[len(panel):])

    ot, qt = panel["t"].to_numpy(), rows["t"].to_numpy()
    amt = panel["거래금액"].to_numpy()
    f = pd.DataFrame(index=rows.index)

    # 송금 계좌 이력 (송금은행도 볼 수 있는 정보)
    hs = History(ids["payer"][0], ot, amt)
    qk = ids["payer"][1]
    f["s_n_prev"], _, _ = hs.count_before(qk, qt)
    f["s_n_7d"], _, _ = hs.count_before(qk, qt, 7 * DAY_H)
    f["s_n_30d"], _, _ = hs.count_before(qk, qt, 30 * DAY_H)
    s30, n30 = hs.sum_before(qk, qt, 30 * DAY_H)
    f["s_amt_sum_30d"] = s30
    sall, nall = hs.sum_before(qk, qt)
    f["s_amt_mean_prev"] = np.where(nall > 0, sall / np.maximum(nall, 1), -1.0)
    _, last_idx = hs.last_before(qk, qt)
    f["s_amt_max_prev"] = np.where(last_idx >= 0, hs.cmax[np.clip(last_idx, 0, None)], -1.0)
    last_t, _ = hs.last_before(qk, qt)
    f["s_hours_since_last"] = np.where(last_t >= 0, qt - last_t, -1)
    first_t = hs.first_time(qk)
    f["s_days_since_first"] = np.where((first_t >= 0) & (first_t < qt), (qt - first_t) // DAY_H, -1)
    f["s_n_payees_prev"] = distinct_before(ids["payer"][0], ids["payee"][0], ot, qk, qt)
    f["s_n_dbanks_prev"] = distinct_before(ids["payer"][0], ids["입금금융회사일련번호"][0], ot, qk, qt)
    hp = History(ids["pair"][0], ot)
    f["pair_n_prev"], _, _ = hp.count_before(ids["pair"][1], qt)

    # 수취 계좌의 타행 이력 (공동망에서만 모이는 정보)
    hr = History(ids["payee"][0], ot)
    rk = ids["payee"][1]
    f["r_n_prev_in"], _, _ = hr.count_before(rk, qt)
    f["r_n_in_30d"], _, _ = hr.count_before(rk, qt, 30 * DAY_H)
    f["r_n_payers_prev"] = distinct_before(ids["payee"][0], ids["payer"][0], ot, rk, qt)
    f["r_n_sendbanks_prev"] = distinct_before(ids["payee"][0], ids["출금금융회사일련번호"][0], ot, rk, qt)
    rfirst = hr.first_time(rk)
    f["r_days_since_first"] = np.where((rfirst >= 0) & (rfirst < qt), (qt - rfirst) // DAY_H, -1)

    # 은행쌍 이력 (공동망 정보)
    hb = History(ids["bankpair"][0], ot)
    f["bankpair_n_30d"], _, _ = hb.count_before(ids["bankpair"][1], qt, 30 * DAY_H)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="_datasets/panel.parquet")
    ap.add_argument("--sample", default="_datasets/sample.parquet")
    ap.add_argument("--out", default="data/finsyn-v2")
    ap.add_argument("--train_end", default="20231231")
    ap.add_argument("--val_end", default="20240630")
    args = ap.parse_args()

    panel, rows = load(args.panel), load(args.sample)
    print(f"panel {len(panel):,} rows | sample (dedup) {len(rows):,} rows, {rows.y.sum():,} suspicious")

    feats = build_features(panel, rows)
    date = rows["거래일자"]
    split = np.where(date <= args.train_end, "train", np.where(date <= args.val_end, "val", "test"))

    cat = pd.DataFrame({
        "hour_band": rows["거래시간대"],
        "dow": pd.to_datetime(rows["거래일자"], format="%Y%m%d").dt.dayofweek.astype(str),
        "withdraw_bank": rows["출금금융회사일련번호"],
        "deposit_bank": rows["입금금융회사일련번호"],
        "media_type": rows["매체구분"],
        "fund_type": rows["자금구분"],
    })
    num = pd.concat([rows[["거래금액"]].rename(columns={"거래금액": "amount"}), feats], axis=1).astype(float)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for s in ["train", "val", "test"]:
        m = split == s
        np.save(out / f"X_num_{s}.npy", num.values[m])
        np.save(out / f"X_cat_{s}.npy", cat.values[m].astype(object), allow_pickle=True)
        np.save(out / f"y_{s}.npy", rows["y"].values[m])
        sizes[s] = int(m.sum())
        print(f"  {s}: {m.sum():,} rows, {rows['y'].values[m].sum():,} suspicious "
              f"({rows['y'].values[m].mean():.2%}), dates {date[m].min()}~{date[m].max()}")

    info = {
        "task_type": "binclass", "name": out.name, "id": f"{out.name}--id",
        "train_size": sizes["train"], "val_size": sizes["val"], "test_size": sizes["test"],
        "n_num_features": num.shape[1], "n_cat_features": cat.shape[1], "n_classes": 2,
        "num_cols": list(num.columns), "cat_cols": list(cat.columns),
        "column_names": list(num.columns) + list(cat.columns) + ["y"],
        "split": {"train": f"~{args.train_end}", "val": f"~{args.val_end}", "test": "rest"},
        "feature_groups": {
            "transaction": ["amount", "hour_band", "dow", "media_type", "fund_type"],
            "bank": ["withdraw_bank", "deposit_bank", "bankpair_n_30d"],
            "sender_history": [c for c in feats.columns if c.startswith("s_") or c == "pair_n_prev"],
            "receiver_crossbank_history": [c for c in feats.columns if c.startswith("r_")],
        },
    }
    info["column_mapping"] = {str(i): c for i, c in enumerate(info["column_names"])}
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))

    # 진단용 전체 테이블 (원천 ID 포함, 공개 금지)
    full = pd.concat([rows[RAW + ["y", "t"]], num.drop(columns=["amount"]), cat], axis=1).assign(split=split)
    full.to_parquet(out / "full_with_ids_DO_NOT_RELEASE.parquet", index=False)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
