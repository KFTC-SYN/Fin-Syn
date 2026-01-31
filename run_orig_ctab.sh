# CTAB-GAN 튜닝(파라미터서치)
# python CTAB-GAN/tune_ctabgan.py data/orig-micro/ 66528 synthetic cuda:0
# python CTAB-GAN/tune_ctabgan.py data/orig-micro-retry/ 63703 synthetic cuda:0

# CTAB-GAN 학습/생성/평가
# python CTAB-GAN/pipeline_ctabgan.py --config exp/orig-micro/ctabgan/config.toml --train --sample --eval
# python CTAB-GAN/pipeline_ctabgan.py --config exp/orig-micro-retry/ctabgan/config.toml --train --sample --eval

# CTAB-GAN 평가 (여러 시드로 평가)
# python scripts/eval_seeds.py --config exp/orig-micro/ctabgan/config.toml 10 ctabgan synthetic catboost 5
# python scripts/eval_seeds.py --config exp/orig-micro-retry/ctabgan/config.toml 10 ctabgan synthetic catboost 5

# CTAB-GAN 평가 (합성데이터 크기 별 MLE 평가)
# python scripts/sample_and_eval_scale.py --config exp/orig-micro/ctabgan/config.toml --model_type ctabgan --sizes 1.0 1.5 2.0 --seed 0 --change_val --n_seeds 5
# python scripts/sample_and_eval_scale.py --config exp/orig-micro-retry/ctabgan/config.toml --model_type ctabgan --sizes 1.0 1.5 2.0 --seed 0 --change_val --n_seeds 5

# CTAB-GAN 평가 (합성데이터 크기 별 MLE 평가 결과 plot)
# python scripts/plot_scale_mle.py --base_dir exp/orig-micro/ctabgan
# python scripts/plot_scale_mle.py --base_dir exp/orig-micro-retry/ctabgan

# CTAB-GAN 평가 (SynthEval 공식 라이브러리 - 전체 메트릭, conda finsyn에서 수행 필요)
# python scripts/eval_syntheval_total.py --config exp/orig-micro/ctabgan/config.toml --preset full_eval --change_val --exclude nnaa
python scripts/eval_syntheval_total.py --config exp/orig-micro-retry/ctabgan/config.toml --preset full_eval --change_val --exclude nnaa --sizes 2.0
