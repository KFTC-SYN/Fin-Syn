# README
#########################################################################################
# TabPFGen 학습/생성/평가 작업은 conda finsyn 환경에서 수행 필요
#########################################################################################

# TabPFGen 튜닝(파라미터서치)
# python TabPFGen/tune_tabpfgen.py data/orig-micro/ 66528 synthetic cuda:0
# python TabPFGen/tune_tabpfgen.py data/orig-micro-retry/ 63703 synthetic cuda:0

# TabPFGen 학습/생성/평가
# python TabPFGen/pipeline_tabpfgen.py --config exp/orig-micro/tabpfgen/config.toml --train --sample --eval
# python TabPFGen/pipeline_tabpfgen.py --config exp/orig-micro-retry/tabpfgen/config.toml --train --sample --eval

# TabPFGen 평가 (여러 시드로 평가)
# python scripts/eval_seeds.py --config exp/orig-micro/tabpfgen/config.toml 10 tabpfgen synthetic catboost 5
# python scripts/eval_seeds.py --config exp/orig-micro-retry/tabpfgen/config.toml 10 tabpfgen synthetic catboost 5

# TabPFGen 평가 (합성데이터 크기 별 MLE 평가)
# python scripts/sample_and_eval_scale.py --config exp/orig-micro/tabpfgen/config.toml --model_type tabpfgen --sizes 1.0 1.5 2.0 --seed 0 --change_val --n_seeds 5
# python scripts/sample_and_eval_scale.py --config exp/orig-micro-retry/tabpfgen/config.toml --model_type tabpfgen --sizes 1.0 1.5 2.0 --seed 0 --change_val --n_seeds 5

# TabPFGen 평가 (합성데이터 크기 별 MLE 평가 결과 plot)
# python scripts/plot_scale_mle.py --base_dir exp/orig-micro/tabpfgen
# python scripts/plot_scale_mle.py --base_dir exp/orig-micro-retry/tabpfgen

# TabPFGen 평가 (SynthEval 공식 라이브러리 - 전체 메트릭)
# python scripts/eval_syntheval_total.py --config exp/orig-micro/tabpfgen/config.toml --preset full_eval --change_val
python scripts/eval_syntheval_total.py --config exp/orig-micro-retry/tabpfgen/config.toml --preset full_eval --change_val --exclude nnaa --sizes 1.0 1.5 2.0
