"""
공개본 Parquet이 평가에 쓴 내부 배열과 값 하나까지 같은지 확인한다(9/23).

leaderboard.py가 두 형식을 읽은 결과(수치 열 float, 범주 열 문자열, 정답 int)를 split마다 비교한다.
하나라도 다르면 0이 아닌 코드로 끝난다.

Usage:
    python scripts/verify_release.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from leaderboard import load_split  # noqa: E402

bad, n = [], 0
for d in sorted((ROOT / "release/data").glob("*/seed*")):
    src = ROOT / "exp/finsyn-v2/synth" / d.parent.name / d.name
    for sp in ("train", "val", "test"):
        Xa, ya, _ = load_split(str(d), sp)
        Xb, yb, _ = load_split(str(src), sp)
        n += 1
        ok = (list(Xa.columns) == list(Xb.columns) and np.array_equal(ya, yb)
              and all(np.array_equal(Xa[c].to_numpy(), Xb[c].to_numpy()) for c in Xa.columns))
        if not ok:
            bad.append(f"{d.parent.name}/{d.name}/{sp}")
print(f"{n} splits checked, {len(bad)} differ")
for b in bad[:20]:
    print("  ", b)
sys.exit(1 if bad else 0)
