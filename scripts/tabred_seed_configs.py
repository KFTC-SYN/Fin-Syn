"""
TabReD 복제를 여러 시드로 확장한다: 시드 0 설정을 그대로 복제하고 seed 값만 바꾼다.

왜 초기 템플릿 폴더가 아니라 시드 0 설정을 복제하는가: 템플릿이 남아 있지 않고,
시드 0 설정을 복제하면 "같은 설정, 다른 시드"라는 전제가 파일 수준에서 보장된다.

Usage:
    python scripts/tabred_seed_configs.py --seeds 1,2,3,4
"""
import argparse
import json
import shutil
from pathlib import Path

import tomli
import tomli_w

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "exp/tabred-hc/gen"
PARTS = ["train", "val", "test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4")
    a = ap.parse_args()
    n = 0
    for model_dir in sorted(GEN.iterdir()):
        if not model_dir.is_dir():
            continue
        for part in PARTS:
            src = model_dir / part
            cfg_src = src / "config.toml"
            if not cfg_src.exists():
                continue
            for s in [int(x) for x in a.seeds.split(",")]:
                dst = model_dir / f"{part}_seed{s}"
                dst.mkdir(parents=True, exist_ok=True)
                cfg = tomli.loads(cfg_src.read_text())
                cfg["seed"] = s
                cfg["parent_dir"] = str(dst.relative_to(ROOT))
                for key in ("train", "sample", "diffusion_params", "model_params"):
                    if isinstance(cfg.get(key), dict) and "seed" in cfg[key]:
                        cfg[key]["seed"] = s
                (dst / "config.toml").write_bytes(tomli_w.dumps(cfg).encode())
                if (src / "info.json").exists():
                    shutil.copyfile(src / "info.json", dst / "info.json")
                n += 1
    print(f"설정 {n}개 작성 ({len(PARTS)} part x 시드 x 생성기)")


if __name__ == "__main__":
    main()
