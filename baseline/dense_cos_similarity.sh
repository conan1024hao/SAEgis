#!/bin/bash

source .venv/bin/activate

DATASETS=("NIPS17" "LLaVA-Instruct-150K" "Medical-Multimodal-Eval")
ATTACK_METHOD="FOA-Attack"
SAE_LOCATION="projection-mlp2"

for train_data in "${DATASETS[@]}"; do
    for test_data in "${DATASETS[@]}"; do
        echo "============================================================"
        echo "Running dense_cos_similarity.py with train_data=$train_data and test_data=$test_data"

        python baseline/dense_cos_similarity.py \
            --base_path "./sae_activations" \
            --train_data "$train_data" \
            --test_data "$test_data" \
            --attack_method "$ATTACK_METHOD" \
            --sae_location "$SAE_LOCATION"
    done
done