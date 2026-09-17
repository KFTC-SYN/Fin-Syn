"""
공개용 릴리스의 프라이버시 평가.

DCR(최근접 실제 레코드까지의 거리) 중앙값만으로는 "이 값이 위험한가"를 말할 수 없다.
그래서 **같은 기간·같은 분포의 비회원(holdout)** 을 기준선으로 잡는다:
추출본에서 학습기간(2021-09~2023-12) 이체 중 벤치마크 train 분할에 포함되지 않은 거래를 뽑아
동일한 피처 규칙으로 만든다. 생성기는 이 거래들을 본 적이 없으므로 "실제이지만 비회원"이다.

지표
  copy_rate    : 비공개 train 행과 완전히 동일한 합성 행의 비율
  dcr_syn      : median DCR(합성 -> train)
  dcr_holdout  : median DCR(비회원 holdout -> train)   ← 기준선. dcr_syn이 이보다 작으면 과도한 근접.
  dcr_ratio    : dcr_syn / dcr_holdout  (1에 가까울수록 "새로 뽑은 실제 표본과 비슷한 거리")
  dcr_share    : 합성 행의 최근접 이웃이 (train ∪ 같은 크기 holdout) 중 train에 있을 확률.
                 0.5 = 회원/비회원 구분 없음, 1에 가까울수록 학습 데이터 기억.
  mia_auc      : 거리 기반 멤버십 추론. 각 실제 레코드에 대해 가장 가까운 합성 행까지의 거리를 점수로
                 회원(train) vs 비회원(holdout)을 구분했을 때의 AUC. 0.5 = 누출 없음.

Usage:
    python scripts/privacy_v2.py --models smote,tabddpm,great,tabpfgen,tabpfgen-prior,ctgan,tvae,ctabgan,ctabgan-plus
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2"
REAL = ROOT / "data/finsyn-v2"
INFO = json.loads((REAL / "info.json").read_text())
NUM, CAT = INFO["num_cols"], INFO["cat_cols"]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset_v2 import build_features, load as load_raw  # noqa: E402

RAW_KEY = ["거래일자", "거래시간대", "출금금융회사일련번호", "출금계좌일련번호",
           "입금금융회사일련번호", "입금계좌일련번호", "거래금액", "매체구분", "자금구분"]


def load_split(d, s):
    Xn = pd.DataFrame(np.load(Path(d) / f"X_num_{s}.npy", allow_pickle=True).astype(float), columns=NUM)
    Xc = pd.DataFrame(np.load(Path(d) / f"X_cat_{s}.npy", allow_pickle=True), columns=CAT).astype(str)
    return Xn, Xc


def build_holdout(orig_path, n, seed):
    """학습기간 중 벤치마크에 포함되지 않은 이체 n건을 같은 피처 규칙으로 만든다."""
    orig = load_raw(orig_path)
    bench = pd.read_parquet(REAL / "full_with_ids_DO_NOT_RELEASE.parquet")
    lo, hi = bench.loc[bench.split == "train", "거래일자"].min(), bench.loc[bench.split == "train", "거래일자"].max()

    key = lambda d: d[RAW_KEY].astype(str).agg("|".join, axis=1)
    used = set(key(bench[bench.split == "train"]))
    period = orig[(orig["거래일자"] >= lo) & (orig["거래일자"] <= hi)].reset_index(drop=True)
    mask = ~key(period).isin(used)
    pool = period[mask].reset_index(drop=True)
    print(f"holdout pool: {len(pool):,} of {len(period):,} train-period transfers "
          f"({mask.mean():.1%} not in the benchmark), flagged {pool.y.mean():.3%}", flush=True)

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(pool), min(n, len(pool)), replace=False))
    rows = pool.iloc[idx].reset_index(drop=True)
    feats = build_features(orig, rows)
    Xc = pd.DataFrame({
        "hour_band": rows["거래시간대"],
        "dow": pd.to_datetime(rows["거래일자"], format="%Y%m%d").dt.dayofweek.astype(str),
        "withdraw_bank": rows["출금금융회사일련번호"],
        "deposit_bank": rows["입금금융회사일련번호"],
        "media_type": rows["매체구분"],
        "fund_type": rows["자금구분"],
    }).astype(str)
    Xn = pd.concat([rows[["거래금액"]].rename(columns={"거래금액": "amount"}), feats], axis=1).astype(float)[NUM]
    return Xn.reset_index(drop=True), Xc.reset_index(drop=True)


class Encoder:
    """standard_metrics_v2.privacy 와 같은 인코딩: 1~99% 분위 min-max + one-hot, 유클리드 거리."""

    def __init__(self, rn, rc):
        lo, hi = rn.quantile(0.01), rn.quantile(0.99)
        self.lo, self.scale = lo, (hi - lo).replace(0, 1)
        self.cols = pd.get_dummies(rc).columns

    def __call__(self, Xn, Xc):
        a = ((Xn[NUM] - self.lo) / self.scale).clip(-1, 2).values
        b = pd.get_dummies(Xc).reindex(columns=self.cols, fill_value=0).values
        return np.hstack([a, b]).astype(np.float32)


def nn_dist(fit, query, k=1):
    d, _ = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(fit).kneighbors(query)
    return d[:, k - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--orig", default=str(ROOT / "_datasets/orig.parquet"))
    ap.add_argument("--n", type=int, default=20000, help="holdout/합성 표본 크기")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(E / "privacy.json"))
    args = ap.parse_args()

    rn, rc = load_split(REAL, "train")
    hn, hc = build_holdout(args.orig, args.n, args.seed)
    enc = Encoder(rn, rc)
    R, H = enc(rn, rc), enc(hn, hc)
    rng = np.random.default_rng(args.seed)

    # 기준선: 비회원 holdout -> train 거리 (생성기가 본 적 없는 '실제' 표본의 근접도)
    dcr_holdout = float(np.median(nn_dist(R, H)))
    # 회원/비회원 균형 세트 (dcr_share, mia 용)
    mi = rng.choice(len(R), min(args.n, len(R)), replace=False)
    members, nonmembers = R[mi], H
    print(f"baseline: median DCR(holdout -> train) = {dcr_holdout:.4f}  "
          f"(members {len(members):,}, non-members {len(nonmembers):,})", flush=True)

    out = Path(args.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    res["_baseline"] = {"dcr_holdout_to_train": dcr_holdout, "n_holdout": int(len(H)), "n_members": int(len(members))}

    key = lambda a, b: pd.concat([a[NUM].round(6).astype(str), b[CAT]], axis=1).agg("|".join, axis=1)
    real_keys = set(key(rn, rc))

    for m in args.models.split(","):
        sn, sc = load_split(E / f"synth/{m}/seed0", "train")
        copy_rate = float(key(sn, sc).isin(real_keys).mean())
        si = rng.choice(len(sn), min(args.n, len(sn)), replace=False)
        S = enc(sn.iloc[si], sc.iloc[si])

        dcr_syn = float(np.median(nn_dist(R, S)))
        # dcr_share: 합성 행의 최근접 이웃이 train 쪽인가, 같은 크기의 비회원 쪽인가
        ri = rng.choice(len(R), len(H), replace=False)
        d_tr, d_ho = nn_dist(R[ri], S), nn_dist(H, S)
        share = float((d_tr < d_ho).mean() + 0.5 * (d_tr == d_ho).mean())
        # MIA: 합성데이터만 보고 회원/비회원을 구분할 수 있는가 (가까울수록 회원이라고 예측)
        s_mem, s_non = -nn_dist(S, members), -nn_dist(S, nonmembers)
        mia = float(roc_auc_score(np.r_[np.ones(len(s_mem)), np.zeros(len(s_non))], np.r_[s_mem, s_non]))

        res[m] = {"copy_rate": copy_rate, "dcr_syn_to_train": dcr_syn, "dcr_holdout_to_train": dcr_holdout,
                  "dcr_ratio": dcr_syn / dcr_holdout, "dcr_share": share, "mia_auc": mia}
        out.write_text(json.dumps(res, indent=1))
        print(f"{m:16s} copy={copy_rate*100:5.2f}%  DCR={dcr_syn:.4f} (ratio {dcr_syn/dcr_holdout:.2f})  "
              f"share={share:.3f}  MIA-AUC={mia:.3f}", flush=True)

    df = pd.DataFrame({k: v for k, v in res.items() if not k.startswith("_")}).T
    print("\n", df.round(3).to_string())


if __name__ == "__main__":
    main()
