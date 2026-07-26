#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Default values
DATASET="NIPS17"
SUBSET="attacked" # or "original"
SPLIT="dev"
ATTACK_METHOD="SSA-CWA"
if [ "$SUBSET" == "original" ]; then
    IMAGES_PATH="images/$DATASET/$SUBSET/$SPLIT"
else
    IMAGES_PATH="images/$DATASET/$SUBSET/$ATTACK_METHOD/$SPLIT"
fi

# Which base model / SAE to cache activations for. Both SAEs sit at the vision->LM
# projection; the location names differ so their activation directories never collide.
MODEL="qwen2.5-vl-3b" # or "gemma-4-e2b"
if [ "$MODEL" == "gemma-4-e2b" ]; then
    MODEL_PATH="google/gemma-4-E2B-it"
    SAE_LOCATION="vision-projection"
    SAE_MODEL_PATH="mtri-admin/gemma-4-e2b-sae-$SAE_LOCATION-finevisionmax-500k"
else
    MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
    SAE_LOCATION="projection-mlp2"
    SAE_MODEL_PATH="mtri-admin/qwen2.5-vl-3b-sae-$SAE_LOCATION-finevisionmax-500k"
fi

if [ "$SUBSET" == "original" ]; then
    ACTIVATION_PATH="./sae_activations/$DATASET/$SAE_LOCATION/$SUBSET-$SPLIT"
else
    ACTIVATION_PATH="./sae_activations/$DATASET/$ATTACK_METHOD/$SAE_LOCATION/$SUBSET-$SPLIT"
fi

PROMPT="Describe this image."

# Run the Python script with the parsed arguments
python3 cache_activations.py \
    --model_path "$MODEL_PATH" \
    --sae_model_path "$SAE_MODEL_PATH" \
    --images_path "$IMAGES_PATH" \
    --prompt "$PROMPT" \
    --activation_path "$ACTIVATION_PATH"