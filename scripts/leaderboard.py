"""
사기 탐지 모델 리더보드 (leaderboard fidelity 실험의 공통 모듈).

학습/튜닝/평가 데이터를 각각 지정할 수 있어 세 조건에 같은 코드를 쓴다.
  real/real   : --train data/finsyn-v2 --tune data/finsyn-v2:val --test data/finsyn-v2:test
  synth/real  : --train <synthetic dir> --tune <synthetic dir>:val --test data/finsyn-v2:test
  synth/synth : --train <synthetic dir> --tune <synthetic dir>:val --test <synthetic dir>:test

모든 모델은 동일 예산(Optuna TPE n_trials, 튜닝 지표 = 튜닝 split PR-AUC)으로 튜닝한 뒤
최적 설정으로 seed 반복 학습, 평가 split에서 PR-AUC / Recall@FPR / ROC-AUC를 기록한다.
노이즈 기준선: 평가 행 bootstrap으로 모델 순위의 불확실성(τ ceiling)과 쌍별 유의 차이를 계산.

Usage:
    python scripts/leaderboard.py --train data/finsyn-v2 --out exp/finsyn-v2/leaderboard_real --models all
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
# 동시 실행 시 과점유를 막기 위해 환경변수로 조절한다(단독 실행이면 LB_N_JOBS=8 권장).
N_JOBS = int(os.environ.get("LB_N_JOBS", "4"))


# ---------------------------------------------------------------- data
def load_split(spec: str, default_split: str):
    d, s = (spec.split(":") + [default_split])[:2]
    if (Path(d) / f"{s}.parquet").exists() and not (Path(d) / "info.json").exists():
        return load_parquet(Path(d) / f"{s}.parquet")
    info = json.loads((Path(d) / "info.json").read_text())
    X = pd.concat([
        pd.DataFrame(np.load(Path(d) / f"X_num_{s}.npy", allow_pickle=True).astype(float), columns=info["num_cols"]),
        pd.DataFrame(np.load(Path(d) / f"X_cat_{s}.npy", allow_pickle=True), columns=info["cat_cols"]).astype(str),
    ], axis=1)
    y = np.load(Path(d) / f"y_{s}.npy", allow_pickle=True).astype(int)
    return X, y, info


def load_parquet(f: Path):
    """공개본 형식(release/data/<generator>/seed<k>/<split>.parquet)을 읽는다. 열 순서와 형식은
    package_release.py가 쓴 그대로이고, 범주 열은 category, 정답은 label이다(9/23: README 예시가 이 경로를 쓴다)."""
    df = pd.read_parquet(f)
    y = df.pop("label").to_numpy().astype(int)
    cat = [c for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == object]
    num = [c for c in df.columns if c not in cat]
    X = pd.concat([df[num].astype(float), df[cat].astype(str)], axis=1)
    return X, y, {"num_cols": num, "cat_cols": cat}


class Prep:
    """모델 계열별 입력 변환. 범주 목록·스케일은 학습 split에서만 fit."""

    def __init__(self, info, Xtr):
        self.num, self.cat = info["num_cols"], info["cat_cols"]
        self.cats = {c: sorted(Xtr[c].unique()) for c in self.cat}
        lx = self._log(Xtr[self.num])
        self.mu, self.sd = lx.mean(), lx.std().replace(0, 1)

    @staticmethod
    def _log(X):
        return np.sign(X) * np.log1p(np.abs(X))

    def native(self, X):  # LightGBM / XGBoost / CatBoost
        X = X.copy()
        for c in self.cat:
            X[c] = pd.Categorical(X[c], categories=self.cats[c])
        return X

    def dense(self, X):  # LR / kNN / MLP / RF / ET / DT / NB
        num = (self._log(X[self.num]) - self.mu) / self.sd
        oh = [pd.get_dummies(pd.Categorical(X[c], categories=self.cats[c]), prefix=c, dtype=float) for c in self.cat]
        return np.hstack([num.values] + [o.values for o in oh]).astype(np.float32)


# ---------------------------------------------------------------- models
def space(name, trial):
    if name == "tabm":
        from tabm_detector import space_tabm
        return space_tabm(trial)
    if name == "lr":
        return dict(C=trial.suggest_float("C", 1e-3, 1e2, log=True), class_weight=trial.suggest_categorical("cw", [None, "balanced"]))
    if name == "dt":
        return dict(max_depth=trial.suggest_int("max_depth", 3, 20), min_samples_leaf=trial.suggest_int("msl", 1, 200, log=True))
    if name in ("rf", "et"):
        return dict(n_estimators=300, max_depth=trial.suggest_int("max_depth", 6, 40),
                    min_samples_leaf=trial.suggest_int("msl", 1, 50, log=True),
                    max_features=trial.suggest_float("max_features", 0.1, 1.0),
                    class_weight=trial.suggest_categorical("cw", [None, "balanced_subsample"]))
    if name == "knn":
        return dict(n_neighbors=trial.suggest_int("k", 5, 200, log=True), weights=trial.suggest_categorical("w", ["uniform", "distance"]))
    if name == "nb":
        return dict(var_smoothing=trial.suggest_float("vs", 1e-12, 1e-2, log=True))
    if name == "mlp":
        return dict(hidden_layer_sizes=(trial.suggest_categorical("width", [64, 128, 256]),) * trial.suggest_int("depth", 1, 3),
                    alpha=trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
                    learning_rate_init=trial.suggest_float("lr", 1e-4, 1e-2, log=True))
    if name == "hgb":
        return dict(learning_rate=trial.suggest_float("lr", 0.01, 0.3, log=True), max_leaf_nodes=trial.suggest_int("leaves", 8, 128, log=True),
                    min_samples_leaf=trial.suggest_int("msl", 5, 200, log=True), l2_regularization=trial.suggest_float("l2", 1e-8, 10, log=True))
    if name == "lgbm":
        return dict(learning_rate=trial.suggest_float("lr", 0.01, 0.2, log=True), num_leaves=trial.suggest_int("leaves", 8, 256, log=True),
                    min_child_samples=trial.suggest_int("mcs", 5, 200, log=True), subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample", 0.3, 1.0), reg_lambda=trial.suggest_float("l2", 1e-8, 10, log=True))
    if name == "xgb":
        return dict(learning_rate=trial.suggest_float("lr", 0.01, 0.2, log=True), max_depth=trial.suggest_int("max_depth", 3, 10),
                    min_child_weight=trial.suggest_float("mcw", 1e-2, 20, log=True), subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample", 0.3, 1.0), reg_lambda=trial.suggest_float("l2", 1e-8, 10, log=True))
    if name == "catboost":
        return dict(learning_rate=trial.suggest_float("lr", 0.01, 0.2, log=True), depth=trial.suggest_int("depth", 4, 10),
                    l2_leaf_reg=trial.suggest_float("l2", 1e-2, 30, log=True), bagging_temperature=trial.suggest_float("bt", 0, 1))
    raise ValueError(name)


def fit_predict(name, params, prep, Xtr, ytr, Xva, yva, Xte, seed):
    """학습 후 (튜닝 split 점수용 예측, 평가 split 예측) 반환. 부스팅은 튜닝 split으로 early stopping."""
    if name == "tabm":  # GPU. 튜닝 split PR-AUC로 early stopping (tabm_detector.py)
        from tabm_detector import fit_predict_tabm
        return fit_predict_tabm(params, prep, Xtr, ytr, Xva, yva, Xte, seed)
    if name in ("lgbm", "xgb", "catboost"):
        A, B, C = prep.native(Xtr), prep.native(Xva), prep.native(Xte)
        if name == "lgbm":
            import lightgbm as lgb
            m = lgb.LGBMClassifier(n_estimators=3000, subsample_freq=1, random_state=seed, n_jobs=N_JOBS, verbose=-1, **params)
            m.fit(A, ytr, eval_set=[(B, yva)], eval_metric="average_precision", callbacks=[lgb.early_stopping(100, verbose=False)])
        elif name == "xgb":
            import xgboost as xgb
            m = xgb.XGBClassifier(n_estimators=3000, tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
                                  eval_metric="aucpr", early_stopping_rounds=100, random_state=seed, n_jobs=N_JOBS, **params)
            m.fit(A, ytr, eval_set=[(B, yva)], verbose=False)
        else:
            from catboost import CatBoostClassifier
            cat_idx = [A.columns.get_loc(c) for c in prep.cat]
            A, B, C = (X.astype({c: str for c in prep.cat}) for X in (A, B, C))
            m = CatBoostClassifier(iterations=3000, eval_metric="PRAUC", od_type="Iter", od_wait=100, random_seed=seed,
                                   thread_count=N_JOBS, verbose=False, cat_features=cat_idx, **params)
            m.fit(A, ytr, eval_set=(B, yva))
        return m.predict_proba(B)[:, 1], m.predict_proba(C)[:, 1]

    A, B, C = prep.dense(Xtr), prep.dense(Xva), prep.dense(Xte)
    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=2000, random_state=seed, **params)
    elif name == "dt":
        from sklearn.tree import DecisionTreeClassifier
        m = DecisionTreeClassifier(random_state=seed, **params)
    elif name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(random_state=seed, n_jobs=N_JOBS, **params)
    elif name == "et":
        from sklearn.ensemble import ExtraTreesClassifier
        m = ExtraTreesClassifier(random_state=seed, n_jobs=N_JOBS, **params)
    elif name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        m = KNeighborsClassifier(n_jobs=N_JOBS, **params)
    elif name == "nb":
        from sklearn.naive_bayes import GaussianNB
        m = GaussianNB(**params)
    elif name == "mlp":
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier(early_stopping=True, max_iter=200, random_state=seed, **params)
    elif name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=1000, early_stopping=True, random_state=seed, **params)
    m.fit(A, ytr)
    return m.predict_proba(B)[:, 1], m.predict_proba(C)[:, 1]


# ---------------------------------------------------------------- metrics
def recall_at_fpr(y, p, fpr_target):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(fpr_target, fpr, tpr))


def metrics(y, p):
    if len(np.unique(y)) < 2:  # 합성 평가셋에 양성이 없으면 순위 지표 정의 불가
        return {"pr_auc": float("nan"), "roc_auc": float("nan"), "recall@0.1%fpr": float("nan"), "recall@1%fpr": float("nan")}
    return {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            "recall@0.1%fpr": recall_at_fpr(y, p, 0.001), "recall@1%fpr": recall_at_fpr(y, p, 0.01)}


def board_score(y, p):
    """리더보드 점수 = 시드별 PR-AUC의 평균. p는 (시드, 행) 배열. 요약·부트스트랩·감사가 모두 이 정의를 쓴다(9/23).
    1차원이면(시드별 예측이 없는 옛 결과) 그 예측 하나의 PR-AUC."""
    p = np.atleast_2d(p)
    return float(np.mean([average_precision_score(y, q) for q in p]))


def seed_preds(out, name):
    """저장된 시드별 평가 예측을 (시드, 행)으로 쌓는다. 없으면 평균 예측 하나를 돌려준다."""
    fs = sorted(Path(out).glob(f"pred_test_{name}_seed*.npy"), key=lambda f: int(f.stem.rsplit("seed", 1)[1]))
    return np.stack([np.load(f) for f in fs]) if fs else np.load(Path(out) / f"pred_test_{name}.npy")


def bootstrap_noise(y, preds: dict, n_boot=1000, seed=0):
    """평가 행 bootstrap으로 순위 불확실성: 전체 대비 bootstrap 순위의 Kendall τ 분포와 쌍별 유의 차이.
    preds의 값이 (시드, 행)이면 각 bootstrap 표본에서도 시드별 PR-AUC의 평균으로 점수를 낸다."""
    rng = np.random.default_rng(seed)
    names = list(preds)
    P = {n: np.atleast_2d(preds[n]) for n in names}
    full = np.array([board_score(y, P[n]) for n in names])
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = np.empty((n_boot, len(names)))
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])  # 층화 bootstrap
        boots[b] = [board_score(y[idx], P[n][:, idx]) for n in names]
    taus = np.array([kendalltau(full, boots[b]).statistic for b in range(n_boot)])
    sig = {}
    for i, a in enumerate(names):
        for j, c in enumerate(names):
            if i < j:
                d = boots[:, i] - boots[:, j]
                # 동점은 반씩 나눈다. 두 탐지기가 매 표본 같은 점수(예: 둘 다 1.000)면 0.5가 되어
                # 분리 가능한 쌍으로 세지 않는다. 예전에는 동점을 뒤쪽의 승리로 세었다(9/23 발견).
                sig[f"{a}>{c}"] = float((d > 0).mean() + 0.5 * (d == 0).mean())
    ci = {n: [float(np.quantile(boots[:, k], 0.025)), float(np.quantile(boots[:, k], 0.975))] for k, n in enumerate(names)}
    return {"tau_vs_full_mean": float(np.nanmean(taus)), "tau_vs_full_q05": float(np.nanquantile(taus, 0.05)),
            "pr_auc_ci95": ci, "pairwise_win_prob": sig}


# ---------------------------------------------------------------- main
ALL = ["nb", "dt", "lr", "knn", "mlp", "rf", "et", "hgb", "lgbm", "xgb", "catboost"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--tune", default=None, help="dir[:split], default <train>:val")
    ap.add_argument("--test", default="data/finsyn-v2:test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="all")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    Xtr, ytr, info = load_split(args.train, "train")
    Xva, yva, _ = load_split(args.tune or args.train, "val")
    Xte, yte, _ = load_split(args.test, "test")
    prep = Prep(info, Xtr)
    models = ALL if args.models == "all" else args.models.split(",")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    print(f"train {len(ytr):,} ({ytr.mean():.2%}) | tune {len(yva):,} ({yva.mean():.2%}) | test {len(yte):,} ({yte.mean():.2%})", flush=True)

    for name in models:
        if name in res:
            print(f"skip {name}", flush=True)
            continue
        t0 = time.time()
        # kNN / NB / DT 는 탐색공간이 작아 trial을 줄여도 동일 예산 원칙 위반이 아님(상한 동일)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))

        def objective(trial):
            pv, _ = fit_predict(name, space(name, trial), prep, Xtr, ytr, Xva, yva, Xva.iloc[:1], seed=0)
            return average_precision_score(yva, pv)

        study.optimize(objective, n_trials=args.trials, catch=(Exception,))
        # 합성데이터가 소수 클래스를 거의 잃으면(예: 양성 1건) 일부 모델은 학습 자체가 불가능하다.
        # 이 경우 학습 데이터의 양성 비율을 상수로 예측한 것으로 기록한다(무작위 순위 수준) — 공개 데이터의 실패로 그대로 반영.
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        best = space(name, optuna.trial.FixedTrial(study.best_params)) if completed else None
        degenerate = best is None
        runs, test_preds = [], []
        for s in range(args.seeds):
            try:
                if degenerate:
                    raise ValueError("no successful tuning trial")
                pv, pt = fit_predict(name, best, prep, Xtr, ytr, Xva, yva, Xte, seed=s)
            except Exception:  # noqa: BLE001
                degenerate = True
                pv, pt = np.full(len(yva), ytr.mean()), np.full(len(yte), ytr.mean())
            runs.append({"seed": s, "tune_pr_auc": float(average_precision_score(yva, pv)), **metrics(yte, pt)})
            test_preds.append(pt)
            np.save(out / f"pred_test_{name}_seed{s}.npy", pt)
        np.save(out / f"pred_test_{name}.npy", np.mean(test_preds, axis=0))
        summ = {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))]
                for k in ["tune_pr_auc", "pr_auc", "roc_auc", "recall@0.1%fpr", "recall@1%fpr"]}
        res[name] = {"best_params": study.best_params if completed else None,
                     "best_tune_pr_auc": study.best_value if completed else None, "degenerate": degenerate,
                     "runs": runs, "summary": summ, "minutes": round((time.time() - t0) / 60, 1)}
        res_path.write_text(json.dumps(res, indent=1))
        print(f"{name:9s} test PR-AUC {summ['pr_auc'][0]:.3f}±{summ['pr_auc'][1]:.3f}  R@0.1%FPR {summ['recall@0.1%fpr'][0]:.3f}  "
              f"R@1%FPR {summ['recall@1%fpr'][0]:.3f}  ROC {summ['roc_auc'][0]:.3f}  ({res[name]['minutes']} min)", flush=True)

    done = [m for m in models if m in res]
    if len(done) < 2:  # 단일 모델(예: --models tabm 부가 실행)은 순위 노이즈를 정의할 수 없다
        return
    preds = {m: seed_preds(out, m) for m in done}
    noise = bootstrap_noise(yte, preds)
    # seed 노이즈: seed별 리더보드끼리의 τ
    seed_boards = np.array([[res[m]["runs"][s]["pr_auc"] for m in done] for s in range(args.seeds)])
    seed_taus = [kendalltau(seed_boards[a], seed_boards[b]).statistic for a in range(args.seeds) for b in range(a + 1, args.seeds)]
    noise["seed_tau_mean"] = float(np.nanmean(seed_taus))
    (out / "noise_floor.json").write_text(json.dumps(noise, indent=1))
    board = sorted(done, key=lambda m: -res[m]["summary"]["pr_auc"][0])
    print("\nleaderboard (PR-AUC averaged over seeds, bootstrap CI95):")
    for m in board:
        print(f"  {m:9s} {res[m]['summary']['pr_auc'][0]:.3f}  CI95 {noise['pr_auc_ci95'][m][0]:.3f}-{noise['pr_auc_ci95'][m][1]:.3f}")
    print(f"bootstrap τ(full, boot) mean {noise['tau_vs_full_mean']:.3f} (5% {noise['tau_vs_full_q05']:.3f}) | seed τ mean {noise['seed_tau_mean']:.3f}")
    sep = sum(1 for v in noise["pairwise_win_prob"].values() if v >= 0.975 or v <= 0.025)
    print(f"pairs separable at 95%: {sep}/{len(noise['pairwise_win_prob'])}")


if __name__ == "__main__":
    main()
