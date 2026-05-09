#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

TRAIN_DATA="Medical-Multimodal-Eval"
TEST_DATA="LLaVA-Instruct-150K"
ATTACK_METHOD="FOA-Attack"
NUM_FEATURES=256

SAE_LOCATIONS=("projection-mlp2" "vision-block0" "vision-block10")

CLEAN_DEV_PATHS=()
CLEAN_TEST_PATHS=()
ATTACKED_TEST_PATHS=()
TOP_FEATURE_FILES=()

for SAE_LOCATION in "${SAE_LOCATIONS[@]}"; do
    CLEAN_DEV_PATHS+=("./sae_activations/$TRAIN_DATA/$SAE_LOCATION/original-dev")
    CLEAN_TEST_PATHS+=("./sae_activations/$TEST_DATA/$SAE_LOCATION/original-test")
    ATTACKED_TEST_PATHS+=("./sae_activations/$TEST_DATA/$ATTACK_METHOD/$SAE_LOCATION/attacked-test")
    TOP_FEATURE_FILES+=("./top_feature_idxs/$TRAIN_DATA/$ATTACK_METHOD/$SAE_LOCATION/top_features.json")
done

FIG_DIR="./figs/$TRAIN_DATA-$TEST_DATA/ensemble_projection-mlp2_vision-block0_vision-block10"

echo "============================================================"
echo "Running defense_attack_ensemble.py with 3 SAE locations: ${SAE_LOCATIONS[*]}"

python defense_attack_ensemble.py \
    --clean_dev_path "${CLEAN_DEV_PATHS[@]}" \
    --clean_test_path "${CLEAN_TEST_PATHS[@]}" \
    --attacked_test_path "${ATTACKED_TEST_PATHS[@]}" \
    --fig_dir "$FIG_DIR" \
    --top_feature_file "${TOP_FEATURE_FILES[@]}" \
    --num_features_to_select "$NUM_FEATURES"
