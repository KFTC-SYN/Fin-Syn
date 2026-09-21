"""
교환 가능한 비회원으로 잰 SMOTE의 프라이버시(반분 설계).

왜: 비공개 train 밖에서 뽑은 비회원(무작위 또는 같은 표집 규칙)은 벤치마크 멤버와 분포가 완전히 같지 않다.
    그래서 원본에서 먼 릴리스에서도 MIA가 0.43–0.60으로 흔들린다(privacy.json, privacy_matched.json).
    재학습 비용이 거의 없는 SMOTE는 train을 층화 반분해 한쪽(A)으로만 생성하고 다른 쪽(B)을 비회원으로 쓰면
    멤버와 비회원이 교환 가능하다. 공개본과 같게 비공개 레코드와 똑같은 행은 제거한다.
대조: B 자체를 '릴리스'로 두면(실제 데이터의 다른 절반) DCR 비율 약 1, MIA 약 0.5가 나와야 한다.

Usage:
    python scripts/privacy_halfsplit_v2.py --seeds 0,1,2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from privacy_v2 import CAT, E, NUM, REAL, Encoder, load_split, nn_dist  # noqa: E402
from synth_v2 import fit_sample_smote, round_integer_columns  # noqa: E402


def metrics(enc, A, B, S, rng, n=20000):
    ai = rng.choice(len(A), min(n, len(A)), replace=False)
    bi = rng.choice(len(B), min(n, len(B)), replace=False)
    si = rng.choice(len(S), min(n, len(S)), replace=False)
    Am, Bm, Sm = A[ai], B[bi], S[si]
    dcr_syn, dcr_base = float(np.median(nn_dist(Am, Sm))), float(np.median(nn_dist(Am, Bm)))
    d_a, d_b = nn_dist(Am, Sm), nn_dist(Bm, Sm)
    share = float((d_a < d_b).mean() + 0.5 * (d_a == d_b).mean())
    s_mem, s_non = -nn_dist(Sm, Am), -nn_dist(Sm, Bm)
    mia = float(roc_auc_score(np.r_[np.ones(len(s_mem)), np.zeros(len(s_non))], np.r_[s_mem, s_non]))
    return {"dcr_ratio": dcr_syn / dcr_base, "dcr_share": share, "mia_auc": mia}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    rn, rc = load_split(REAL, "train")
    y = np.load(REAL / "y_train.npy").astype(int)
    enc = Encoder(rn, rc)
    key = lambda n, c: pd.concat([n[NUM].round(6).astype(str), c[CAT]], axis=1).agg("|".join, axis=1)
    private = set()
    for s in ("train", "val", "test"):
        n_, c_ = load_split(REAL, s)
        private |= set(key(n_, c_))
    res = {}
    for seed in [int(s) for s in a.seeds.split(",")]:
        ia, ib = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=seed)
        An, Ac, Bn, Bc = rn.iloc[ia], rc.iloc[ia], rn.iloc[ib], rc.iloc[ib]
        sn, sc, sy = fit_sample_smote(An.values, Ac.values, y[ia], seed=seed)
        sn = round_integer_columns(An.values, sn)
        Sn, Sc = pd.DataFrame(sn, columns=NUM), pd.DataFrame(sc, columns=CAT).astype(str)
        keep = ~key(Sn, Sc).isin(private).values
        Sn, Sc = Sn[keep].reset_index(drop=True), Sc[keep].reset_index(drop=True)
        A, B, S = enc(An, Ac), enc(Bn, Bc), enc(Sn, Sc)
        rng = np.random.default_rng(seed)
        res[f"smote|{seed}"] = {**metrics(enc, A, B, S, rng), "copies_removed": float(1 - keep.mean())}
        # 대조: 실제 데이터의 다른 절반을 릴리스로 (B를 둘로 나눠 B1=릴리스, B2=비회원)
        b1, b2 = train_test_split(np.arange(len(ib)), test_size=0.5, stratify=y[ib], random_state=seed)
        res[f"real-half|{seed}"] = metrics(enc, A, B[b2], B[b1], rng)
        print(seed, {k: {m: round(v, 3) for m, v in res[k].items()} for k in res if k.endswith(f"|{seed}")}, flush=True)
    out = E / "privacy_halfsplit.json"
    out.write_text(json.dumps(res, indent=1))
    df = pd.DataFrame(res).T
    df["release"] = [k.split("|")[0] for k in df.index]
    print(df.groupby("release").agg(["mean", "min", "max"]).round(3).to_string())
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
