# SMOTE 튜닝(파라미터서치)
# python smote/tune_smote.py data/orig-micro/ synthetic
# python smote/tune_smote.py data/orig-micro-retry/ synthetic

# SMOTE 학습/생성/평가
# python smote/pipeline_smote.py --config exp/orig-micro/smote/config.toml --sample --eval
# python smote/pipeline_smote.py --config exp/orig-micro-retry/smote/config.toml --sample --eval

# SMOTE 평가 (합성데이터 크기 별 MLE 평가)
# python scripts/sample_and_eval_scale.py --config exp/orig-micro/smote/config.toml --model_type smote --sizes 1.0 1.5 2.0 --seed 0 --change_val --n_seeds 5
# python scripts/sample_and_eval_scale.py --config exp/orig-micro-retry/smote/config.toml --model_type smote --sizes 1.0 1.5 2.0 --seed 0 --change_val --n_seeds 5

# SMOTE 평가 (합성데이터 크기 별 MLE 평가 결과 plot)
# python scripts/plot_scale_mle.py --base_dir exp/orig-micro/smote
# python scripts/plot_scale_mle.py --base_dir exp/orig-micro-retry/smote

# SMOTE 평가 (SynthEval 공식 라이브러리 - 전체 메트릭, conda finsyn에서 수행 필요)
# python scripts/eval_syntheval_total.py --config exp/orig-micro/smote/config.toml --preset full_eval --change_val --exclude nnaa
python scripts/eval_syntheval_total.py --config exp/orig-micro-retry/smote/config.toml --preset full_eval --change_val --exclude nnaa --sizes 1.5 2.0
