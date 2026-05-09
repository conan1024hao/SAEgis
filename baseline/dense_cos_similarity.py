import numpy as np
import torch
import os
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import argparse

# -----------------------
# Configuration
# -----------------------
parser = argparse.ArgumentParser(description="Dense Cosine Similarity Detection")
parser.add_argument("--base_path", type=str, default="./sae_activations", help="Base path for SAE activations")
parser.add_argument("--train_data", type=str, default="NIPS17", help="Training dataset name")
parser.add_argument("--test_data", type=str, default="Medical-Multimodal-Eval", help="Test dataset name")
parser.add_argument("--train_attack_method", type=str, default="SSA-CWA", help="Train attack method name")
parser.add_argument("--test_attack_method", type=str, default="M-Attack", help="Test attack method name")
parser.add_argument("--sae_location", type=str, default="projection-mlp2", help="SAE location")

args = parser.parse_args()
base_path = args.base_path
train_data = args.train_data
test_data = args.test_data
train_attack_method = args.train_attack_method
test_attack_method = args.test_attack_method
sae_location = args.sae_location

# Paths
path_clean_train = f"{base_path}/{train_data}/{sae_location}/original-train"
path_clean_dev   = f"{base_path}/{train_data}/{sae_location}/original-dev"
path_clean_test  = f"{base_path}/{test_data}/{sae_location}/original-test"
path_atk_dev     = f"{base_path}/{train_data}/{train_attack_method}/{sae_location}/attacked-dev"
path_atk_test    = f"{base_path}/{test_data}/{test_attack_method}/{sae_location}/attacked-test"

# -----------------------
# Helpers
# -----------------------
def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = pick_device()
print(f"Using device: {device}")

def load_origin_out(path):
    """
    Robustly loads 'origin_out' from a file or directory of .pt files.
    """
    if not os.path.exists(path):
        print(f"Warning: Path not found -> {path}")
        return []

    # Case 1: Single file
    if path.endswith(".pt") and os.path.isfile(path):
        try:
            d = torch.load(path, map_location="cpu")
            return d["origin_out"]
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return []

    # Case 2: Directory
    step_files = sorted([f for f in os.listdir(path) if f.endswith(".pt")])
    outs = []
    print(f"Loading {len(step_files)} files from {path}...")
    
    for step_file in step_files:
        try:
            full_path = os.path.join(path, step_file)
            step_data = torch.load(full_path, map_location="cpu")
            
            if "origin_out" not in step_data:
                continue
                
            oo = step_data["origin_out"]
            if isinstance(oo, list):
                outs.extend(oo)
            else:
                outs.append(oo)
        except Exception as e:
            print(f"Skipping {step_file}: {e}")
            
    return outs

def pool_seq(x):
    """
    Robustly pools sequence data to a single vector [dim].
    Handles: [seq, dim], [1, seq, dim], or [dim].
    """
    # Ensure tensor
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    
    # Move to device for computation
    x = x.to(device)
    
    # Handle shapes
    if x.dim() == 1:
        return x  # Already pooled [dim]
    elif x.dim() == 2:
        return x.mean(0)  # [seq, dim] -> [dim]
    elif x.dim() == 3:
        # [batch, seq, dim] -> flatten -> mean
        return x.view(-1, x.shape[-1]).mean(0)
    else:
        # Fallback for higher dims
        return x.view(-1, x.shape[-1]).mean(0)

def build_proto(data_list, n=None):
    """
    Builds a prototype vector by averaging pooled vectors from the list.
    """
    if not data_list:
        raise ValueError("Empty data list provided to build_proto")
        
    use = data_list if n is None else data_list[:n]
    
    # Accumulate sum and count to save memory (avoid stacking huge list)
    proto_sum = None
    count = 0
    
    with torch.no_grad():
        for item in use:
            vec = pool_seq(item) # [dim] on device
            if proto_sum is None:
                proto_sum = torch.zeros_like(vec)
            proto_sum += vec
            count += 1
            
    return proto_sum / count

def detection_score(act, clean_proto, atk_proto):
    """
    Score > 0 implies 'Attacked' (closer to attacked prototype).
    """
    with torch.no_grad():
        v = pool_seq(act)
        # Cosine Similarity
        sim_clean = F.cosine_similarity(v, clean_proto, dim=0)
        sim_atk   = F.cosine_similarity(v, atk_proto, dim=0)
        return (sim_atk - sim_clean).item()

# -----------------------
# Metrics Logic
# -----------------------
def get_threshold_at_fpr(neg_scores, target_fpr):
    """
    Finds threshold where FPR on neg_scores is <= target_fpr.
    Uses np.percentile for O(1) lookup after sort.
    """
    neg_scores = np.sort(np.asarray(neg_scores, dtype=np.float64))
    if len(neg_scores) == 0:
        return np.inf
    
    # The index corresponding to the (1 - target_fpr) quantile
    # e.g., for FPR 0.05, we want the score at the 95th percentile.
    # Scores > this threshold are False Positives.
    cutoff_index = int((1.0 - target_fpr) * len(neg_scores))
    cutoff_index = min(cutoff_index, len(neg_scores) - 1)
    
    return neg_scores[cutoff_index]

def calculate_precision_recall(pos_scores, neg_scores, thr):
    """
    Calculates Recall and Precision on Test Set given a threshold.
    """
    pos_scores = np.asarray(pos_scores)
    neg_scores = np.asarray(neg_scores)
    
    # True Positives (Attacked detected as Attacked)
    tp = (pos_scores >= thr).sum()
    # False Positives (Clean detected as Attacked)
    fp = (neg_scores >= thr).sum()
    # False Negatives (Attacked detected as Clean)
    fn = (pos_scores < thr).sum()
    
    recall    = tp / (tp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    
    return recall, precision, fp, tp

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    # 1. Load Data
    print("--- Loading Data ---")
    clean_train = load_origin_out(path_clean_train)
    clean_dev   = load_origin_out(path_clean_dev)
    atk_dev     = load_origin_out(path_atk_dev)
    
    clean_te    = load_origin_out(path_clean_test)
    atk_te      = load_origin_out(path_atk_test)

    print(f"Loaded: CleanTrain={len(clean_train)}, CleanDev={len(clean_dev)}, AtkDev={len(atk_dev)}")
    print(f"Loaded: CleanTest={len(clean_te)}, AtkTest={len(atk_te)}")

    # 2. Build Prototypes
    print("\n--- Building Prototypes ---")
    clean_proto = build_proto(clean_train)
    atk_proto   = build_proto(atk_dev)

    # 3. Compute Scores
    print("\n--- Computing Scores ---")
    # Scores on Dev (to find threshold)
    dev_clean_scores = [detection_score(x, clean_proto, atk_proto) for x in clean_dev]
    
    # Scores on Test (to evaluate)
    test_clean_scores = [detection_score(x, clean_proto, atk_proto) for x in clean_te]
    test_atk_scores   = [detection_score(x, clean_proto, atk_proto) for x in atk_te]

    # 4. Global AUC on Test
    all_test_scores = np.concatenate([test_clean_scores, test_atk_scores])
    all_test_labels = np.concatenate([np.zeros(len(test_clean_scores)), np.ones(len(test_atk_scores))])
    
    auc = roc_auc_score(all_test_labels, all_test_scores)
    
    # 5. Evaluate at Target FPRs
    print("\n" + "="*60)
    print(f"RESULTS: {train_data} (Source) -> {test_data} (Target)")
    print(f"Test AUC: {auc:.4f}")
    print("="*60)

    for fpr_target in [0.05, 0.02, 0.01]:
        # Step A: Find threshold using Clean Dev scores
        thr = get_threshold_at_fpr(dev_clean_scores, fpr_target)
        
        # Step B: Apply threshold to Test sets
        rec, prec, fp_count, tp_count = calculate_precision_recall(test_atk_scores, test_clean_scores, thr)
        
        print(f"Target FPR (Dev) <= {fpr_target*100}%")
        print(f"  Threshold Used: {thr:.6f}")
        print(f"  Test Precision: {prec:.4f}  (TP={tp_count} / (TP={tp_count} + FP={fp_count}))")
        print(f"  Test Recall:    {rec:.4f}  ({tp_count}/{len(test_atk_scores)})")
        print(f"  Test F1 Score:  {2*prec*rec/(prec+rec+1e-12):.4f}")
        print("-" * 40)
