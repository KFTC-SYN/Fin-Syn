#!/bin/bash
# TabReD 복제를 시드 1~4로 확장한다. GPU 경합을 피해 시드를 순차 실행한다.
set -u
V=/home/recordame/workspace/seonkyu/fin-syn/.venv/bin/python
cd /home/recordame/workspace/seonkyu/fin-syn/Fin-Syn
M=ctgan,tvae,ctabgan,ctabgan-plus,tabddpm,tabpfgen
LOG=exp/tabred-hc/logs/seeds_1_4.txt
mkdir -p exp/tabred-hc/logs
for S in 1 2 3 4; do
  echo "=== seed $S 시작 $(date +%H:%M:%S)" >> "$LOG"
  $V scripts/run_generators_v2.py run --models "$M" --real data/tabred-hc --exp exp/tabred-hc --seed "$S" >> "$LOG" 2>&1
  echo "=== seed $S run rc=$? $(date +%H:%M:%S)" >> "$LOG"
  $V scripts/run_generators_v2.py assemble --models "$M" --real data/tabred-hc --exp exp/tabred-hc --seed "$S" >> "$LOG" 2>&1
  echo "=== seed $S assemble rc=$? $(date +%H:%M:%S)" >> "$LOG"
done
echo "=== 전체 완료 $(date +%H:%M:%S)" >> "$LOG"
