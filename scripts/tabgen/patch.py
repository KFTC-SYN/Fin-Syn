"""공식 구현을 수정하지 않고, 실행 시점에 모듈 소스의 특정 줄만 바꿔 불러온다.

각 치환 문자열은 원본에 정확히 한 번 있어야 한다(없으면 즉시 실패). 저장소가 바뀌어 치환이 조용히
무시되는 일을 막기 위해서다. 바꾸는 것은 학습 길이(공통 학습시간 상한)와 호환성 문제뿐이다.
"""
import sys
import types
from pathlib import Path

import torch.optim.lr_scheduler as L


def load_patched(modname, path, replacements):
    src = Path(path).read_text()
    for a, b in replacements:
        n = src.count(a)
        assert n == 1, f"{path}: expected exactly one '{a}', found {n}"
        src = src.replace(a, b)
    mod = types.ModuleType(modname)
    mod.__file__ = str(path)
    mod.__package__ = modname.rpartition(".")[0]
    sys.modules[modname] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


class _RLROP(L.ReduceLROnPlateau):
    """torch 2.x에서 없어진 verbose 인자를 무시한다(동작은 동일)."""

    def __init__(self, *a, verbose=None, **k):
        super().__init__(*a, **k)


def torch_compat():
    L.ReduceLROnPlateau = _RLROP
