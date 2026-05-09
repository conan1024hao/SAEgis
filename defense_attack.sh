#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

DATASET_NAME="NIPS17"
ATTACK_METHOD="SSA-CWA"
SAE_LOCATION="projection-mlp2"

CLEAN_DEV_PATH="./sae_activations/$DATASET_NAME/$SAE_LOCATION/original-dev"
CLEAN_TEST_PATH="./sae_activations/$DATASET_NAME/$SAE_LOCATION/original-test"
ATTACKED_TEST_PATH="./sae_activations/$DATASET_NAME/$ATTACK_METHOD/$SAE_LOCATION/attacked-test"
FIG_DIR="./figs/$DATASET_NAME/$SAE_LOCATION"
TOP_FEATURE_FILE="./top_feature_idxs/$DATASET_NAME/$ATTACK_METHOD/$SAE_LOCATION/top_features.json"
NUM_FEATURES=64

python defense_attack.py \
  --clean_dev_path "$CLEAN_DEV_PATH" \
  --clean_test_path "$CLEAN_TEST_PATH" \
  --attacked_test_path "$ATTACKED_TEST_PATH" \
  --fig_dir "$FIG_DIR" \
  --top_feature_file "$TOP_FEATURE_FILE" \
  --num_features_to_select "$NUM_FEATURES"