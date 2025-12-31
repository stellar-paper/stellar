DEPLOYMENT_NAME="gpt-4o-mini" python run_tests_navi.py \
        --sut "IPA_LOS" \
        --population_size 5 \
        --n_generations 5 \
        --algorithm "nsga2" \
        --max_time "00:00:30" \
        --features_config "configs/navi_features.json"\
        --no_wandb \
        --use_rag \
        --seed 1