"""
TabM(ICLR'25) 탐지기: leaderboard.py의 space()/fit_predict()에 붙는 12번째 모델.

공식 레시피(tabm README, example.ipynb, paper/exp/*/0-tuning.toml)를 따른다:
noisy-quantile 수치 변환, TabM.make(k=32, arch 'tabm'), PiecewiseLinearEmbeddings(TabM-dagger),
AdamW, grad-clip 1.0, 서브모델별 독립 배치(share_training_batches=False, 논문 설정), 추론은 k개 확률의 평균,
튜닝 split PR-AUC 기준 early stopping(patience 16) — 부스팅 모델의 eval_metric과 같은 기준.

탐지기 11개로 보고한 본문 결과를 건드리지 않도록 ALL에는 넣지 않는다.
--models tabm --out exp/finsyn-v2/tabm/<리더보드 이름> 으로 따로 돌리고, 12개 기준 분석은
scripts/tabm_extension_v2.py가 기존 결과와 합쳐서 한다(부록용 견고성 점검).
패키지: tabm==0.0.3, rtdl_num_embeddings==0.0.12 (둘 다 순수 파이썬, torch는 기존 것 사용).
"""
import math
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def space_tabm(trial):
    # 논문 탐색공간(TabM + PiecewiseLinearEmbeddings)을 20-trial 예산과 공유 GPU에 맞게 좁혔다(9/18 실측):
    #   n_blocks 1-5 -> 1-3, d_block 64-1024 -> 64-512, lr 1e-4-5e-3 -> 3e-4-3e-3.
    #   첫 TPE 제안(폭 752, 3블록, lr 6e-4)은 한 번 학습에 20분이 넘었고, 원래 공간이면 리더보드 23개에 약 90시간이 든다.
    #   기본 설정(폭 512, 2블록, lr 2e-3)은 좁힌 공간 안에 있다.
    return dict(n_blocks=trial.suggest_int("n_blocks", 1, 3),
                d_block=trial.suggest_int("d_block", 64, 512, step=16),
                dropout=trial.suggest_float("dropout", 0.0, 0.5),
                lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
                weight_decay=0.0 if trial.suggest_categorical("wd_zero", [True, False])
                else trial.suggest_float("wd", 1e-4, 1e-1, log=True),
                n_bins=trial.suggest_int("n_bins", 2, 128),
                d_embedding=trial.suggest_int("d_embedding", 8, 32, step=4))


def fit_predict_tabm(params, prep, Xtr, ytr, Xva, yva, Xte, seed, max_epochs=100, patience=16,
                     batch_size=512, k=32, device=None, amp=True, verbose=False):
    import rtdl_num_embeddings
    import scipy.special
    import tabm
    import torch
    from sklearn.preprocessing import QuantileTransformer

    p = dict(params)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # numeric: drop columns constant in train (compute_bins raises on them), then noisy-quantile
    num = [c for c in prep.num if Xtr[c].nunique() > 1]
    A = Xtr[num].to_numpy(np.float32)
    qt = QuantileTransformer(n_quantiles=max(min(len(A) // 30, 1000), 10), output_distribution="normal",
                             subsample=10**9, random_state=seed)
    qt.fit(A + np.random.default_rng(seed).normal(0.0, 1e-5, A.shape).astype(np.float32))
    xn = [torch.as_tensor(qt.transform(X[num].to_numpy(np.float32)), dtype=torch.float32, device=dev)
          for X in (Xtr, Xva, Xte)]

    # categorical: ordinal codes over the training-split categories (prep.cats); unseen -> extra bucket
    card = [len(prep.cats[c]) + 1 for c in prep.cat]

    def codes(X):
        cols = []
        for c in prep.cat:
            v = pd.Categorical(X[c], categories=prep.cats[c]).codes.astype(np.int64)
            cols.append(np.where(v < 0, len(prep.cats[c]), v))
        return torch.as_tensor(np.stack(cols, 1), device=dev)

    xc = [codes(X) for X in (Xtr, Xva, Xte)]
    y = torch.as_tensor(ytr, dtype=torch.long, device=dev)

    bins = rtdl_num_embeddings.compute_bins(xn[0], n_bins=min(p.pop("n_bins"), len(A) - 1))
    emb = rtdl_num_embeddings.PiecewiseLinearEmbeddings(bins, d_embedding=p.pop("d_embedding"),
                                                        activation=False, version="B")
    lr, wd = p.pop("lr"), p.pop("weight_decay")
    model = tabm.TabM.make(n_num_features=len(num), cat_cardinalities=card, d_out=2, num_embeddings=emb, k=k,
                           **p).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    use_amp = amp and dev.type == "cuda" and torch.cuda.is_bf16_supported()

    def forward(i, idx):
        with torch.autocast(dev.type, enabled=use_amp, dtype=torch.bfloat16):
            return model(xn[i][idx], xc[i][idx]).float()

    @torch.inference_mode()
    def proba(i):
        model.eval()
        out = torch.cat([forward(i, idx) for idx in torch.arange(len(xn[i]), device=dev).split(8192)])
        return scipy.special.softmax(out.cpu().numpy(), axis=-1).mean(1)[:, 1]  # mean of k probabilities

    n, best, best_state, bad, t0 = len(y), -math.inf, None, 0, time.time()
    for epoch in range(max_epochs):
        model.train()
        # independent batch order per submodel (share_training_batches=False)
        for idx in torch.rand((n, model.k), device=dev).argsort(dim=0).split(batch_size, dim=0):
            opt.zero_grad()
            out = forward(0, idx)                                   # (B, k, 2)
            loss = torch.nn.functional.cross_entropy(out.flatten(0, 1), y[idx].flatten(0, 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        score = average_precision_score(yva, proba(1))
        if verbose:
            print(f"epoch {epoch} val PR-AUC {score:.4f} ({time.time() - t0:.1f}s)", flush=True)
        if score > best:
            best, bad = score, 0
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1
            if bad > patience:
                break
    model.load_state_dict(best_state)
    return proba(1), proba(2)
