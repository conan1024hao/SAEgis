#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

DATASET_NAME="NIPS17"
ATTACK_METHOD="SSA-CWA"
SAE_LOCATION="projection-mlp2"
NUM_IMAGES_TO_USE=100

python analyze_features.py \
  --attacked_activation_path "./sae_activations/$DATASET_NAME/$ATTACK_METHOD/$SAE_LOCATION/attacked-dev" \
  --original_activation_path "./sae_activations/$DATASET_NAME/$SAE_LOCATION/original-train" \
  --num_images_to_use "$NUM_IMAGES_TO_USE" \
  --output_path "./top_feature_idxs/$DATASET_NAME/$ATTACK_METHOD/$SAE_LOCATION/top_features.json"