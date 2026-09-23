"""
감사 반분 B를 합성 과정에서도 제외한 재생성 실험의 생성 단계(9/23 외부 리뷰 W2 대응).

비공개 테스트 기간을 층화 반분(release_selection_v2.py의 첫 분할과 같은 StratifiedShuffleSplit, random_state=0)해
A 절반만으로 각 생성기의 테스트 split 생성기를 다시 학습·표집한다. 설정은 원래 테스트 split 설정 그대로이고
표본 크기만 |A|이다(exp/finsyn-v2-halfA/gen/<model>/test[_seed<k>]/config.toml). 공개본의 train/val split은 그대로 쓴다.
SMOTE는 CPU 작업이라 로컬에서 따로 만든다.

GPU마다 작업 슬롯을 두고, 오래 걸리는 생성기부터 채운다.

Usage (서버, 저장소 루트에서):
    python scripts/halfA_driver.py --gpus 0,1,2 --per-gpu 2
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "exp/finsyn-v2-halfA"
# 원래 테스트 split 생성 시간(분, 로컬 5시드 평균) 순서. 긴 것부터 배치한다.
ORDER = ["tabdiff", "tabsyn", "great", "tabddpm", "tabpfgen-prior", "tvae", "ctabgan", "ctgan", "findiff",
         "ctabgan-plus", "tabpfgen"]
SEEDS = range(5)


def done(m, k):
    d = E / "gen" / m / ("test" if k == 0 else f"test_seed{k}")
    return (d / "y_train.npy").exists()


def patch_columns():
    """CTAB-GAN(+)는 데이터 폴더 이름으로 열 설정을 찾는다. 원래 테스트 기간 항목을 그대로 복사한다."""
    for f in ("CTAB-GAN/columns.json", "CTAB-GAN-Plus/columns.json"):
        p = ROOT / f
        c = json.loads(p.read_text())
        c["finsyn-v2-halfA-part-test"] = c["finsyn-v2-part-test"]
        p.write_text(json.dumps(c, indent=4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--per-gpu", type=int, default=2)
    ap.add_argument("--models", default=",".join(ORDER))
    a = ap.parse_args()
    patch_columns()
    todo = [(m, k) for m in a.models.split(",") for k in SEEDS if not done(m, k)]
    slots = [g for g in a.gpus.split(",") for _ in range(a.per_gpu)]
    (E / "logs").mkdir(parents=True, exist_ok=True)
    tag = "" if a.models == ",".join(ORDER) else "_" + a.models.replace(",", "_")
    running, t0 = {}, time.time()
    print(f"작업 {len(todo)}개, 슬롯 {len(slots)}개", flush=True)
    while todo or running:
        for i, g in enumerate(slots):
            if i not in running and todo:
                m, k = todo.pop(0)
                # 원래 공개본은 GPU 1장에서 만들었다. GReaT(HuggingFace Trainer)는 보이는 GPU를 모두 써서 실질 배치가
                # 커지므로, 외부 래퍼(tabsyn/tabdiff/findiff, 자체적으로 CUDA_VISIBLE_DEVICES를 건다)가 아닌 생성기는
                # 이 GPU 한 장만 보이게 하고 cuda:0으로 실행한다(9/23 발견).
                external = m in ("tabsyn", "tabdiff", "findiff")
                env = dict(os.environ) if external else {**os.environ, "CUDA_VISIBLE_DEVICES": g}
                dev = f"cuda:{g}" if external else "cuda:0"
                cmd = [sys.executable, "scripts/run_generators_v2.py", "run", "--models", m, "--device", dev,
                       "--seed", str(k), "--parts", "test", "--real", "data/finsyn-v2-halfA", "--exp", "exp/finsyn-v2-halfA"]
                fh = open(E / "logs" / f"{m}_seed{k}{tag}.txt", "w")
                running[i] = ((m, k), fh, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT),
                              time.time())
                print(f"[{(time.time() - t0) / 60:6.1f}분] 시작 {m} seed{k} (GPU {g})", flush=True)
        time.sleep(30)
        for i in list(running):
            (m, k), fh, p, ts = running[i]
            if p.poll() is not None:
                fh.close()
                del running[i]
                print(f"[{(time.time() - t0) / 60:6.1f}분] 끝   {m} seed{k} rc={p.returncode} ok={done(m, k)} "
                      f"({(time.time() - ts) / 60:.1f}분, 남은 {len(todo)})", flush=True)
    print("모두 끝", flush=True)


if __name__ == "__main__":
    main()
