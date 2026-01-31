# README
#########################################################################################
# NVIDIA L40 GPU에서는 JIT 컴파일 오류로 인해 학습 실패 (nvrtc: error: invalid value for --gpu-architecture)
# syntheval 라이브러리를 활용한 평가(eval_syntheval_total.py)는 conda finsyn에서 수행 필요
# syntheval 라이브러리를 활용한 평가 시 ks_test는 cpu/memory overflow 이슈로 제외할 필요
# ks_test 메트릭은 syntheval 스타일(eval_syntheval.py)로 별도 평가 필요
#########################################################################################

# 평가 모델(CatBoost) 학습 및 튜닝
# python scripts/tune_evaluation_model.py orig-micro catboost cv cuda:0
# python scripts/tune_evaluation_model.py orig-micro-retry catboost cv cuda:0

# 평가 모델(CatBoost)용 TabDDPM 튜닝 (파라미터 서치)
# python scripts/tune_ddpm.py orig-micro 66528 synthetic catboost ddpm_cb
# python scripts/tune_ddpm.py orig-micro-retry 63703 synthetic catboost ddpm_cb

# 평가 모델(CatBoost)용 TabDDPM 학습/생성/평가
# python scripts/pipeline.py --config exp/orig-micro/ddpm_cb_best/config.toml --train --sample --eval
# python scripts/pipeline.py --config exp/orig-micro-retry/ddpm_cb_best/config.toml --train --change_val
# python scripts/pipeline.py --config exp/orig-micro-retry/ddpm_cb_best/config.toml --change_val --sample --eval
# python scripts/pipeline.py --config exp/orig-micro-retry/ddpm_cb_best/config.toml --train --sample --eval

# 평가 모델(CatBoost)용 Real 데이터 평가 (eval_catboost.json에 "real" 섹션 추가)
# python scripts/eval_seeds.py --config exp/orig-micro/ddpm_cb_best/config.toml 10 ddpm real catboost 1
# python scripts/eval_seeds.py --config exp/orig-micro-retry/ddpm_cb_best/config.toml 10 ddpm real catboost 1

# 평가 모델(CatBoost)용 TabDDPM 생성/평가
# python scripts/eval_seeds.py --config exp/orig-micro/ddpm_cb_best/config.toml 10 ddpm synthetic catboost 5
# python scripts/eval_seeds.py --config exp/orig-micro-retry/ddpm_cb_best/config.toml 10 ddpm synthetic catboost 5

# 평가 모델(CatBoost)용 TabDDPM 평가 (합성데이터 크기 별 MLE 평가)
# python scripts/sample_and_eval_scale.py --config exp/orig-micro/ddpm_cb_best/config.toml --model_type ddpm --sizes 1.0 1.5 2.0 --seed 0 --change_val --n_seeds 5
# python scripts/sample_and_eval_scale.py --config exp/orig-micro-retry/ddpm_cb_best/config.toml --model_type ddpm --sizes 1.0 1.5 2.0 --seed 0 --change_val --n_seeds 5

# 평가 모델(CatBoost)용 TabDDPM 평가 (합성데이터 크기 별 MLE 평가 결과 plot)
# python scripts/plot_scale_mle.py --base_dir exp/orig-micro/ddpm_cb_best
# python scripts/plot_scale_mle.py --base_dir exp/orig-micro-retry/ddpm_cb_best

# 평가 모델(CatBoost)용 TabDDPM 평가 (SynthEval 공식 라이브러리 - 전체 메트릭, conda finsyn에서 수행 필요)
# 단일크기 평가:
# python scripts/eval_syntheval_total.py --config exp/orig-micro/ddpm_cb_best/config.toml --preset full_eval --change_val --exclude nnaa
# 크기별 평가 (size_1.0x, size_1.5x, size_2.0x):
python scripts/eval_syntheval_total.py --config exp/orig-micro-retry/ddpm_cb_best/config.toml --preset full_eval --change_val --exclude nnaa --sizes 1.0 1.5 2.0
