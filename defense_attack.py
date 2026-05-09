import math
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import argparse
from sklearn.metrics import roc_auc_score, precision_recall_curve
from matplotlib.ticker import MaxNLocator

SAE_DIM = 32768
TOPK = 64

parser = argparse.ArgumentParser(description="Analyze SAE activations for defense/attack.")
parser.add_argument("--clean_dev_path", type=str, default="./sae_activations/NIPS17/projection-mlp2/original-dev-100.pt", help="Path to clean activations file")
parser.add_argument("--clean_test_path", type=str, default="./sae_activations/NIPS17/projection-mlp2/original-test-100.pt", help="Path to clean activations file")
parser.add_argument("--attacked_test_path", type=str, default="./sae_activations/NIPS17/projection-mlp2/attacked-test-100.pt", help="Path to attacked activations file")
parser.add_argument("--fig_dir", type=str, default="figs/", help="Directory to save figures")
parser.add_argument("--top_feature_file", type=str, default="./top_feature_idxs/NIPS17/projection-mlp2/top_features.json", help="Path to top feature indices JSON file")
parser.add_argument("--num_features_to_select", type=int, default=20, help="Number of features to select")
parser.add_argument("--dev_mode", action="store_true", help="Whether to use dev mode with fewer images")

args = parser.parse_args()

if ".pt" in args.clean_dev_path:
    clean_activations_dev = torch.load(args.clean_dev_path)
else:
    step_files = sorted([f for f in os.listdir(args.clean_dev_path) if f.endswith(".pt")])
    if args.dev_mode:
        step_files = step_files[:50]
    clean_activations_dev = {"latents": []}
    for step_file in step_files:
        step_data = torch.load(os.path.join(args.clean_dev_path, step_file))
        clean_activations_dev["latents"].append(step_data["latents"])

if ".pt" in args.clean_test_path:
    clean_activations_test = torch.load(args.clean_test_path)
else:
    step_files = sorted([f for f in os.listdir(args.clean_test_path) if f.endswith(".pt")])
    if args.dev_mode:
        step_files = step_files[50:]
    clean_activations_test = {"latents": []}
    for step_file in step_files:
        step_data = torch.load(os.path.join(args.clean_test_path, step_file))
        clean_activations_test["latents"].append(step_data["latents"])

if ".pt" in args.attacked_test_path:
    attacked_activations_test = torch.load(args.attacked_test_path)
else:
    step_files = sorted([f for f in os.listdir(args.attacked_test_path) if f.endswith(".pt")])
    if args.dev_mode:
        step_files = step_files[50:]
    attacked_activations_test = {"latents": []}
    for step_file in step_files:
        step_data = torch.load(os.path.join(args.attacked_test_path, step_file))
        attacked_activations_test["latents"].append(step_data["latents"])

with open(args.top_feature_file, "r") as f:
    top_feature_data = json.load(f)
attack_feature_idxs = top_feature_data["top_1000_features"][:args.num_features_to_select]

fig_dir = args.fig_dir

os.makedirs(fig_dir, exist_ok=True)


# Analyze activated feature numbers
def compute_mean_activated(activations_list, feature_idxs):
    return [
        ((activations.squeeze(0) if activations.dim() == 3 else activations)[:, feature_idxs] > 0).sum(dim=-1).float().mean().item()
        for activations in activations_list
    ]

N_clean_dev_list = compute_mean_activated(clean_activations_dev["latents"], attack_feature_idxs)
N_clean_test_list = compute_mean_activated(clean_activations_test["latents"], attack_feature_idxs)
N_attacked_test_list = compute_mean_activated(attacked_activations_test["latents"], attack_feature_idxs)

# Plot histograms
all_vals = np.concatenate([np.array(N_clean_dev_list + N_clean_test_list), np.array(N_attacked_test_list)])
bins = np.linspace(all_vals.min(), all_vals.max(), 26)
plt.hist(N_clean_dev_list, bins=bins, alpha=0.5, label="clean_dev")
plt.hist(N_clean_test_list, bins=bins, alpha=0.5, label="clean_test")
plt.hist(N_attacked_test_list, bins=bins, alpha=0.5, label="attacked_test")
plt.legend()
plt.xlabel("Mean Number of Activated Features")
plt.ylabel("Number of Samples")
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig(f"{fig_dir}/activated_feature_number_distribution.png", dpi=300)
plt.close()

# Prepare labels and scores
scores = np.array(N_clean_test_list + N_attacked_test_list)
labels = np.array([0] * len(N_clean_test_list) + [1] * len(N_attacked_test_list))  # 0: clean, 1: attacked

# Calculate metrics
auc = roc_auc_score(labels, scores)
precision, recall, thresholds = precision_recall_curve(labels, scores)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
max_f1 = np.max(f1_scores)

clean_dev_tau_0_90 = np.percentile(N_clean_dev_list, 90)
clean_dev_tau_0_95 = np.percentile(N_clean_dev_list, 95)
clean_dev_tau_0_98 = np.percentile(N_clean_dev_list, 98)
clean_dev_tau_0_99 = np.percentile(N_clean_dev_list, 99)
precision_fpr_0_10 = np.sum(np.array(N_attacked_test_list) > clean_dev_tau_0_90) / (
    np.sum(np.array(N_clean_test_list) > clean_dev_tau_0_90) + np.sum(np.array(N_attacked_test_list) > clean_dev_tau_0_90) + 1e-8
)
precision_fpr_0_05 = np.sum(np.array(N_attacked_test_list) > clean_dev_tau_0_95) / (
    np.sum(np.array(N_clean_test_list) > clean_dev_tau_0_95) + np.sum(np.array(N_attacked_test_list) > clean_dev_tau_0_95) + 1e-8
)
precision_fpr_0_02 = np.sum(np.array(N_attacked_test_list) > clean_dev_tau_0_98) / (
    np.sum(np.array(N_clean_test_list) > clean_dev_tau_0_98) + np.sum(np.array(N_attacked_test_list) > clean_dev_tau_0_98) + 1e-8
)
precision_fpr_0_01 = np.sum(np.array(N_attacked_test_list) > clean_dev_tau_0_99) / (
    np.sum(np.array(N_clean_test_list) > clean_dev_tau_0_99) + np.sum(np.array(N_attacked_test_list) > clean_dev_tau_0_99) + 1e-8
)
recall_fpr_0_10 = np.sum(np.array(N_attacked_test_list) > clean_dev_tau_0_90) / len(N_attacked_test_list)
recall_fpr_0_05 = np.sum(np.array(N_attacked_test_list) > clean_dev_tau_0_95) / len(N_attacked_test_list)
recall_fpr_0_02 = np.sum(np.array(N_attacked_test_list) > clean_dev_tau_0_98) / len(N_attacked_test_list)
recall_fpr_0_01 = np.sum(np.array(N_attacked_test_list) > clean_dev_tau_0_99) / len(N_attacked_test_list)

print(f"AUC: {auc:.4f}")
print(f"Max-F1: {max_f1:.4f}")
print(f"Precision at FPR=0.10: {precision_fpr_0_10:.4f}, Recall at FPR=0.10: {recall_fpr_0_10:.4f}, F1 at FPR=0.10: {2 * precision_fpr_0_10 * recall_fpr_0_10 / (precision_fpr_0_10 + recall_fpr_0_10 + 1e-8):.4f}") 
print(f"Precision at FPR=0.05: {precision_fpr_0_05:.4f}, Recall at FPR=0.05: {recall_fpr_0_05:.4f}, F1 at FPR=0.05: {2 * precision_fpr_0_05 * recall_fpr_0_05 / (precision_fpr_0_05 + recall_fpr_0_05 + 1e-8):.4f}")
print(f"Precision at FPR=0.02: {precision_fpr_0_02:.4f}, Recall at FPR=0.02: {recall_fpr_0_02:.4f}, F1 at FPR=0.02: {2 * precision_fpr_0_02 * recall_fpr_0_02 / (precision_fpr_0_02 + recall_fpr_0_02 + 1e-8):.4f}")
print(f"Precision at FPR=0.01: {precision_fpr_0_01:.4f}, Recall at FPR=0.01: {recall_fpr_0_01:.4f}, F1 at FPR=0.01: {2 * precision_fpr_0_01 * recall_fpr_0_01 / (precision_fpr_0_01 + recall_fpr_0_01 + 1e-8):.4f}")

# Plot Precision-Recall curve
plt.figure()
plt.plot(recall, precision, label="Precision-Recall curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig(f"{fig_dir}/precision_recall_curve.png", dpi=300)
plt.close()

# Plot ROC curve
plt.figure()
fpr = 1 - precision
tpr = recall
plt.plot(fpr, tpr, label="ROC curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(f"{fig_dir}/roc_curve.png", dpi=300)
plt.close()
