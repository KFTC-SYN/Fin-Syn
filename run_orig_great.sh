# README
#########################################################################################
# GReaT 학습/생성/평가 작업은 conda tddpm2에서 수행 필요
# syntheval 라이브러리를 활용한 평가(eval_syntheval_total.py)는 conda finsyn에서 수행 필요
# syntheval 라이브러리를 활용한 평가 시 ks_test는 cpu/memory overflow 이슈로 제외할 필요
# ks_test 메트릭은 syntheval 스타일(eval_syntheval.py)로 별도 평가 필요
#########################################################################################

# GReaT 튜닝(파라미터서치)
# python be_great/tune_great.py data/orig-micro/ 66528 synthetic cuda:0
# python be_great/tune_great.py data/orig-micro-retry/ 63703 synthetic cuda:0

# GReaT 학습/생성/평가
# python be_great/pipeline_great.py --config exp/orig-micro/great/config.toml --train --sample --eval
# python be_great/pipeline_great.py --config exp/orig-micro-retry/great/config.toml --sample --eval

# GReaT 평가 (여러 시드로 평가)
# python scripts/eval_seeds.py --config exp/orig-micro/great/config.toml 10 great synthetic catboost 5
# python scripts/eval_seeds.py --config exp/orig-micro-retry/great/config.toml 10 great synthetic catboost 5

# GReaT 평가 (합성데이터 크기 별 MLE 평가)
# python scripts/sample_and_eval_scale.py --config exp/orig-micro/great/config.toml --model_type great --sizes 1.0 1.5 2.0 --seed 0 --change_val --n_seeds 5
# python scripts/sample_and_eval_scale.py --config exp/orig-micro-retry/great/config.toml --model_type great --sizes 1.0 1.5 2.0 --seed 0 --change_val --n_seeds 5

# GReaT 평가 (합성데이터 크기 별 MLE 평가 결과 plot)
# python scripts/plot_scale_mle.py --base_dir exp/orig-micro/great
# python scripts/plot_scale_mle.py --base_dir exp/orig-micro-retry/great

# GReaT 평가 (syntheval)
# python scripts/eval_syntheval_total.py --config exp/orig-micro/great/config.toml --preset full_eval --change_val --exclude nnaa
# python scripts/eval_syntheval_total.py --config exp/orig-micro-retry/great/config.toml --preset full_eval --change_val --exclude nnaa nndr dcr hit_rate eps_risk auroc_diff cls_acc ks_test --sizes 1.5 2.0
python scripts/eval_syntheval_style.py --config exp/orig-micro-retry/great/config.toml --sizes 1.0 1.5 2.0 --metrics ks_test nndr dcr hit_rate eps_risk --change_val
