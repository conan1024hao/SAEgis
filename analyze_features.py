import torch
import argparse
import json
import os

SAE_DIM = 32768

def compute_feature_scores_fast(data, sae_dim=SAE_DIM, device=None, dtype=torch.float32):
    """
    data: dict with keys:
      - "activations": list/iter of tensors, each shape [seq, k] (or [B, seq, k] also supported below)
      - "indices":     list/iter of tensors, same shape as activations, int64/int32

    returns: tensor shape [sae_dim], average score over samples
    score(feature) = max_over_tokens(feature) * log1p(count_over_tokens(feature))
    """
    acts_list = data["activations"]
    idxs_list = data["indices"]
    assert len(acts_list) == len(idxs_list)

    # Pick device
    if device is None:
        device = acts_list[0].device

    total = torch.zeros(sae_dim, device=device, dtype=dtype)
    n = 0

    for activations, indices in zip(acts_list, idxs_list):
        activations = activations.to(device=device, dtype=dtype)
        indices = indices.to(device=device)

        # Support [seq, k] or [B, seq, k] by flattening batch into "samples"
        if activations.dim() == 3:
            B = activations.shape[0]
            for b in range(B):
                total += _score_one_sample(activations[b], indices[b], sae_dim)
                n += 1
            continue

        total += _score_one_sample(activations, indices, sae_dim)
        n += 1

    return total / max(n, 1)


def _score_one_sample(activations_2d, indices_2d, sae_dim):
    """
    activations_2d: [seq, k]
    indices_2d:     [seq, k]
    """
    seq, k = indices_2d.shape

    flat_idx = indices_2d.reshape(-1).to(torch.int64)
    flat_act = activations_2d.reshape(-1)

    # 1) max_over_tokens:
    # Your original does: per token -> max over k for that feature -> then max over tokens.
    # That equals "max over all positions (token,k)" for each feature.
    # Use scatter_reduce amax to get max activation per feature in one shot.
    max_per_feature = torch.full((sae_dim,), float("-inf"), device=flat_act.device, dtype=flat_act.dtype)
    max_per_feature.scatter_reduce_(0, flat_idx, flat_act, reduce="amax", include_self=True)

    # If a feature never appears, set max to 0 (same as your init max_over_tokens=0)
    max_per_feature = torch.where(torch.isfinite(max_per_feature), max_per_feature, torch.zeros_like(max_per_feature))

    # 2) count_over_tokens:
    # Need number of tokens where feature appears at least once (not occurrences).
    # Create (token_id, feature_id) pairs then unique them, then bincount by feature.
    token_ids = torch.arange(seq, device=flat_idx.device, dtype=torch.int64).unsqueeze(1).expand(seq, k).reshape(-1)
    pair = token_ids * sae_dim + flat_idx
    unique_pair = torch.unique(pair)
    unique_feat = unique_pair.remainder(sae_dim)

    count_tokens = torch.bincount(unique_feat, minlength=sae_dim).to(flat_act.dtype)

    # score vector
    return max_per_feature * torch.log1p(count_tokens)


# ----------------- usage -----------------
parser = argparse.ArgumentParser(description="Analyze SAE feature activations.")
parser.add_argument("--attacked_activation_path", type=str, default="./sae_activations/NIPS17/projection-mlp2/attacked-dev-100.pt", help="Path to attacked activations file")
parser.add_argument("--original_activation_path", type=str, default="./sae_activations/NIPS17/projection-mlp2/original-train-800.pt", help="Path to original activations file")
parser.add_argument("--num_images_to_use", type=int, default=100, help="Number of images to use")
parser.add_argument("--num_features_to_select", type=int, default=20, help="Number of features to select")
parser.add_argument("--output_path", type=str, default="top_features.json", help="Path to output JSON file")
parser.add_argument("--dev_mode", action="store_true", help="Whether to use dev mode with fewer images")

args = parser.parse_args()

attacked_activation_path = args.attacked_activation_path
original_activation_path = args.original_activation_path
num_images_to_use = args.num_images_to_use
num_features_to_select = args.num_features_to_select

if ".pt" in attacked_activation_path:
    attacked_activations = torch.load(attacked_activation_path, map_location="cpu")
else:
    step_files = sorted([f for f in os.listdir(attacked_activation_path) if f.endswith(".pt")])
    if args.dev_mode:
        assert len(step_files) >= 50, "Not enough step files for dev mode"
        step_files = step_files[:50]
    attacked_activations = {"activations": [], "indices": []}
    for step_file in step_files:
        step_data = torch.load(os.path.join(attacked_activation_path, step_file), map_location="cpu")
        attacked_activations["activations"].append(step_data["activations"])
        attacked_activations["indices"].append(step_data["indices"])
if ".pt" in original_activation_path:
    original_activations = torch.load(original_activation_path, map_location="cpu")
else:
    step_files = sorted([f for f in os.listdir(original_activation_path) if f.endswith(".pt")])
    if args.dev_mode:
        assert len(step_files) >= 100, "Not enough step files for dev mode"
        step_files = step_files[:100]
    original_activations = {"activations": [], "indices": []}
    for step_file in step_files:
        step_data = torch.load(os.path.join(original_activation_path, step_file), map_location="cpu")
        original_activations["activations"].append(step_data["activations"])
        original_activations["indices"].append(step_data["indices"])

assert len(attacked_activations["activations"]) >= num_images_to_use
attacked_activations = {
    "activations": attacked_activations["activations"][:num_images_to_use],
    "indices": attacked_activations["indices"][:num_images_to_use],
} 

# Put compute on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

attacked_scores = compute_feature_scores_fast(attacked_activations, sae_dim=SAE_DIM, device=device)
original_scores = compute_feature_scores_fast(original_activations, sae_dim=SAE_DIM, device=device)
diffs = attacked_scores - original_scores

top_values, top_indices = torch.topk(diffs, 10)
for rank, idx in enumerate(top_indices.tolist(), start=1):
    signed = diffs[idx].item()
    print(f"{rank:02d}: feature {idx} | abs diff {top_values[rank-1].item():.6f} | signed diff {signed:.6f}")

top_1000_values, top_1000_indices = torch.topk(diffs, min(1000, len(diffs)))
top_1000_features = top_1000_indices.tolist()
output_data = {
    "top_1000_features": top_1000_features,
    "diffs": [diffs[idx].item() for idx in top_1000_indices]
}
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
with open(args.output_path, "w") as f:
    json.dump(output_data, f, indent=2)
