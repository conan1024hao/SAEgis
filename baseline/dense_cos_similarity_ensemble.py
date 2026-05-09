import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


parser = argparse.ArgumentParser(description="Dense Cosine Similarity Detection (Ensemble)")
parser.add_argument("--base_path", type=str, default="./sae_activations", help="Base path for SAE activations")
parser.add_argument("--train_data", type=str, default="NIPS17", help="Training dataset name")
parser.add_argument("--test_data", type=str, default="Medical-Multimodal-Eval", help="Test dataset name")
parser.add_argument("--train_attack_method", type=str, default="SSA-CWA", help="Train attack method name")
parser.add_argument("--test_attack_method", type=str, default="M-Attack", help="Test attack method name")
parser.add_argument(
    "--sae_location",
    nargs="+",
    type=str,
    default=["projection-mlp2"],
    help="One or more SAE locations to ensemble",
)

args = parser.parse_args()
base_path = args.base_path
train_data = args.train_data
test_data = args.test_data
train_attack_method = args.train_attack_method
test_attack_method = args.test_attack_method
sae_locations = args.sae_location


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = pick_device()
print(f"Using device: {device}")
print(f"Ensembling {len(sae_locations)} SAE location(s): {sae_locations}")


def load_origin_out(path):
    if not os.path.exists(path):
        print(f"Warning: Path not found -> {path}")
        return []

    if path.endswith(".pt") and os.path.isfile(path):
        try:
            d = torch.load(path, map_location="cpu")
            return d["origin_out"]
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return []

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
    if not torch.is_tensor(x):
        x = torch.tensor(x)

    x = x.to(device)

    if x.dim() == 1:
        return x
    if x.dim() == 2:
        return x.mean(0)
    if x.dim() == 3:
        return x.view(-1, x.shape[-1]).mean(0)
    return x.view(-1, x.shape[-1]).mean(0)


def build_proto(data_list):
    if not data_list:
        raise ValueError("Empty data list provided to build_proto")

    proto_sum = None
    count = 0

    with torch.no_grad():
        for item in data_list:
            vec = pool_seq(item)
            if proto_sum is None:
                proto_sum = torch.zeros_like(vec)
            proto_sum += vec
            count += 1

    return proto_sum / count


def detection_score(act, clean_proto, atk_proto):
    with torch.no_grad():
        v = pool_seq(act)
        sim_clean = F.cosine_similarity(v, clean_proto, dim=0)
        sim_atk = F.cosine_similarity(v, atk_proto, dim=0)
        return (sim_atk - sim_clean).item()


def get_threshold_at_fpr(neg_scores, target_fpr):
    neg_scores = np.sort(np.asarray(neg_scores, dtype=np.float64))
    if len(neg_scores) == 0:
        return np.inf
    cutoff_index = int((1.0 - target_fpr) * len(neg_scores))
    cutoff_index = min(cutoff_index, len(neg_scores) - 1)
    return neg_scores[cutoff_index]


def calculate_precision_recall(pos_scores, neg_scores, thr):
    pos_scores = np.asarray(pos_scores)
    neg_scores = np.asarray(neg_scores)

    tp = (pos_scores >= thr).sum()
    fp = (neg_scores >= thr).sum()
    fn = (pos_scores < thr).sum()

    recall = tp / (tp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    return recall, precision, fp, tp


def ensemble_score_lists(score_lists, name):
    lengths = {len(x) for x in score_lists}
    if len(lengths) != 1:
        raise ValueError(f"{name} score lists must have the same length for ensembling, got lengths={sorted(lengths)}")
    return np.asarray(score_lists, dtype=np.float64).mean(axis=0).tolist()


if __name__ == "__main__":
    model_dev_clean_scores = []
    model_test_clean_scores = []
    model_test_atk_scores = []

    for sae_location in sae_locations:
        print("\n" + "-" * 60)
        print(f"Processing SAE location: {sae_location}")

        path_clean_train = f"{base_path}/{train_data}/{sae_location}/original-train"
        path_clean_dev = f"{base_path}/{train_data}/{sae_location}/original-dev"
        path_clean_test = f"{base_path}/{test_data}/{sae_location}/original-test"
        path_atk_dev = f"{base_path}/{train_data}/{train_attack_method}/{sae_location}/attacked-dev"
        path_atk_test = f"{base_path}/{test_data}/{test_attack_method}/{sae_location}/attacked-test"

        clean_train = load_origin_out(path_clean_train)
        clean_dev = load_origin_out(path_clean_dev)
        atk_dev = load_origin_out(path_atk_dev)
        clean_te = load_origin_out(path_clean_test)
        atk_te = load_origin_out(path_atk_test)

        print(f"Loaded: CleanTrain={len(clean_train)}, CleanDev={len(clean_dev)}, AtkDev={len(atk_dev)}")
        print(f"Loaded: CleanTest={len(clean_te)}, AtkTest={len(atk_te)}")

        clean_proto = build_proto(clean_train)
        atk_proto = build_proto(atk_dev)

        dev_clean_scores = [detection_score(x, clean_proto, atk_proto) for x in clean_dev]
        test_clean_scores = [detection_score(x, clean_proto, atk_proto) for x in clean_te]
        test_atk_scores = [detection_score(x, clean_proto, atk_proto) for x in atk_te]

        model_dev_clean_scores.append(dev_clean_scores)
        model_test_clean_scores.append(test_clean_scores)
        model_test_atk_scores.append(test_atk_scores)

    dev_clean_scores = ensemble_score_lists(model_dev_clean_scores, "Dev clean")
    test_clean_scores = ensemble_score_lists(model_test_clean_scores, "Test clean")
    test_atk_scores = ensemble_score_lists(model_test_atk_scores, "Test attacked")

    all_test_scores = np.concatenate([test_clean_scores, test_atk_scores])
    all_test_labels = np.concatenate([np.zeros(len(test_clean_scores)), np.ones(len(test_atk_scores))])
    auc = roc_auc_score(all_test_labels, all_test_scores)

    print("\n" + "=" * 60)
    print(f"RESULTS (ENSEMBLE): {train_data} (Source) -> {test_data} (Target)")
    print(f"Test AUC: {auc:.4f}")
    print("=" * 60)

    for fpr_target in [0.05, 0.02, 0.01]:
        thr = get_threshold_at_fpr(dev_clean_scores, fpr_target)
        rec, prec, fp_count, tp_count = calculate_precision_recall(test_atk_scores, test_clean_scores, thr)

        print(f"Target FPR (Dev) <= {fpr_target*100}%")
        print(f"  Threshold Used: {thr:.6f}")
        print(f"  Test Precision: {prec:.4f}  (TP={tp_count} / (TP={tp_count} + FP={fp_count}))")
        print(f"  Test Recall:    {rec:.4f}  ({tp_count}/{len(test_atk_scores)})")
        print(f"  Test F1 Score:  {2*prec*rec/(prec+rec+1e-12):.4f}")
        print("-" * 40)
