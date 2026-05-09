#!/bin/bash

source .venv/bin/activate

DATASETS=("NIPS17" "LLaVA-Instruct-150K" "Medical-Multimodal-Eval")


TRAIN_DATA="NIPS17"
TEST_DATA="NIPS17"
ATTACK_METHOD="FOA-Attack"
SAE_LOCATIONS=("projection-mlp2" "vision-block0" "vision-block10")

for TRAIN_DATA in "${DATASETS[@]}"; do
    for TEST_DATA in "${DATASETS[@]}"; do
        echo "============================================================"
        echo "Running dense_cos_similarity_ensemble.py with train_data=$TRAIN_DATA and test_data=$TEST_DATA"
        echo "SAE locations: ${SAE_LOCATIONS[*]}"

        python baseline/dense_cos_similarity_ensemble.py \
            --base_path "./sae_activations" \
            --train_data "$TRAIN_DATA" \
            --test_data "$TEST_DATA" \
            --attack_method "$ATTACK_METHOD" \
            --sae_location "${SAE_LOCATIONS[@]}"

    done
done
