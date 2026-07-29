"""Reproduce the SAEgis tables (in-domain / cross-domain / cross-attack / overall).

The paper's protocol, per Sec. 4.2-4.3:

  feature selection  score(f) = mean over images of  max_act(f) * log1p(#tokens firing f)
                     ranked by  score_attacked(dev) - score_clean(train),  top K = 256
  detection score    per image, mean over tokens of #selected features with latent > 0
  threshold          98th percentile of clean *dev* scores  (FPR = 0.02)
  metrics            P / R / F1 on clean test + attacked test at that threshold

Loading raw latents for every table cell would be prohibitive -- one image is a
(tokens, 32768) float32 array, ~33 MB. Instead each activation directory is reduced once
to three per-image vectors over the 32768 latents:

  pos_count[f]  tokens where latent_f > 0        -> detection score for ANY feature subset
  max_act[f]    max top-k activation of f        -> feature-selection score
  topk_count[f] tokens where f appears in top-k  -> feature-selection score

because  mean_over_tokens(#selected active) == pos_count[selected].sum() / n_tokens.
That makes every table cell a cheap dot product over cached summaries.
"""

import argparse
import json
import os

import numpy as np
import torch

SAE_DIM = 32768
DATASETS = ["NIPS17", "LLaVA-Instruct-150K", "Medical-Multimodal-Eval"]
SHORT = {"NIPS17": "NIPS17", "LLaVA-Instruct-150K": "LLaVA", "Medical-Multimodal-Eval": "Medical"}
ATTACKS = ["SSA-CWA", "M-Attack", "FOA-Attack"]


# --------------------------------------------------------------------------- summaries
def summarize_dir(act_dir: str) -> dict:
    """Reduce one activation directory to per-image vectors over the latent dimension."""
    files = sorted(f for f in os.listdir(act_dir) if f.endswith(".pt"))
    pos_count, max_act, topk_count, n_tokens = [], [], [], []

    for name in files:
        d = torch.load(os.path.join(act_dir, name), map_location="cpu")
        lat = d["latents"]
        acts, idxs = d["activations"], d["indices"]
        if lat.dim() == 3:
            lat, acts, idxs = lat.squeeze(0), acts.squeeze(0), idxs.squeeze(0)
        T = lat.shape[0]

        pos_count.append((lat > 0).sum(0).to(torch.int32).numpy())

        flat_i = idxs.reshape(-1).to(torch.int64)
        flat_a = acts.reshape(-1).float()
        m = torch.zeros(SAE_DIM, dtype=torch.float32)
        m.scatter_reduce_(0, flat_i, flat_a, reduce="amax", include_self=True)
        max_act.append(m.numpy())

        tok = torch.arange(T).unsqueeze(1).expand_as(idxs).reshape(-1)
        uniq = torch.unique(tok * SAE_DIM + flat_i).remainder(SAE_DIM)
        topk_count.append(torch.bincount(uniq, minlength=SAE_DIM).to(torch.int32).numpy())

        n_tokens.append(T)

    return {
        "pos_count": np.stack(pos_count),
        "max_act": np.stack(max_act),
        "topk_count": np.stack(topk_count),
        "n_tokens": np.array(n_tokens),
    }


def load_or_build(act_dir: str, cache_dir: str) -> dict:
    os.makedirs(cache_dir, exist_ok=True)
    key = act_dir.strip("./").replace("/", "__") + ".npz"
    path = os.path.join(cache_dir, key)
    if os.path.exists(path):
        return dict(np.load(path))
    s = summarize_dir(act_dir)
    np.savez_compressed(path, **s)
    return s


# --------------------------------------------------------------------- paper protocol
def feature_scores(s: dict) -> np.ndarray:
    """score(f) = mean over images of max_act(f) * log1p(#tokens firing f)."""
    return (s["max_act"] * np.log1p(s["topk_count"].astype(np.float32))).mean(0)


def select_features(attacked: dict, clean: dict, k: int = 256) -> np.ndarray:
    diff = feature_scores(attacked) - feature_scores(clean)
    return np.argsort(-diff)[:k]


def detection_scores(s: dict, idxs: np.ndarray) -> np.ndarray:
    """Per image: mean over tokens of the number of selected features with latent > 0."""
    return s["pos_count"][:, idxs].sum(1) / s["n_tokens"]


def prf(clean_dev, clean_test, attacked_test, fpr: float = 0.02):
    tau = np.percentile(clean_dev, 100 * (1 - fpr))
    tp = int((attacked_test > tau).sum())
    fp = int((clean_test > tau).sum())
    p = 100.0 * tp / (tp + fp) if tp + fp else 0.0
    r = 100.0 * tp / len(attacked_test)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"P": round(p, 1), "R": round(r, 1), "F1": round(f1, 1), "TP": tp, "FP": fp}


# ------------------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--location", required=True, help="SAE location dir name, e.g. vision-projection")
    ap.add_argument("--act_root", default="./sae_activations")
    ap.add_argument("--cache_dir", default="./table_cache")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--fpr", type=float, default=0.02)
    ap.add_argument("--out", required=True, help="write results JSON here")
    args = ap.parse_args()

    loc, root = args.location, args.act_root
    clean = lambda d, sp: f"{root}/{d}/{loc}/original-{sp}"
    attacked = lambda d, a, sp: f"{root}/{d}/{a}/{loc}/attacked-{sp}"

    cache = {}

    def S(path):
        if path not in cache:
            cache[path] = load_or_build(path, os.path.join(args.cache_dir, loc))
        return cache[path]

    # Features selected per (dataset, attack) from that dataset's attacked dev vs clean train.
    feats = {
        (d, a): select_features(S(attacked(d, a, "dev")), S(clean(d, "train")), args.k)
        for d in DATASETS
        for a in ATTACKS
    }

    def cell(src_d, src_a, tgt_d, tgt_a):
        idxs = feats[(src_d, src_a)]
        return prf(
            detection_scores(S(clean(tgt_d, "dev")), idxs),
            detection_scores(S(clean(tgt_d, "test")), idxs),
            detection_scores(S(attacked(tgt_d, tgt_a, "test")), idxs),
            args.fpr,
        )

    results = {"location": loc, "k": args.k, "fpr": args.fpr}

    # Table 1: in-domain -- select and evaluate on the same dataset and attack.
    results["table1"] = {SHORT[d]: {a: cell(d, a, d, a) for a in ATTACKS} for d in DATASETS}

    # Table 2: cross-domain -- same attack, features from a different dataset.
    transfers = [
        ("NIPS17", "Medical-Multimodal-Eval"),
        ("LLaVA-Instruct-150K", "Medical-Multimodal-Eval"),
        ("Medical-Multimodal-Eval", "NIPS17"),
        ("Medical-Multimodal-Eval", "LLaVA-Instruct-150K"),
    ]
    results["table2"] = {
        f"{SHORT[s]}->{SHORT[t]}": {a: cell(s, a, t, a) for a in ATTACKS} for s, t in transfers
    }

    # Table 3: cross-attack -- same dataset, features selected on SSA-CWA.
    results["table3"] = {
        f"SSA-CWA->{tgt}": {SHORT[d]: cell(d, "SSA-CWA", d, tgt) for d in DATASETS}
        for tgt in ["M-Attack", "FOA-Attack"]
    }

    # Table 4: averages. Cross-attack is reported as a delta against the in-domain score
    # for the same target attack, following Sec. 4.3.
    def avg(cells, key):
        return round(float(np.mean([c[key] for c in cells])), 1)

    t1 = [results["table1"][SHORT[d]][a] for d in DATASETS for a in ATTACKS]
    t2 = [v[a] for v in results["table2"].values() for a in ATTACKS]
    t3 = [results["table3"][f"SSA-CWA->{t}"][SHORT[d]] for t in ["M-Attack", "FOA-Attack"] for d in DATASETS]
    t1_tgt = [results["table1"][SHORT[d]][t] for t in ["M-Attack", "FOA-Attack"] for d in DATASETS]

    results["table4"] = {
        "in_domain": {k: avg(t1, k) for k in ("P", "R", "F1")},
        "cross_domain": {k: avg(t2, k) for k in ("P", "R", "F1")},
        "cross_attack_delta": {
            k: round(avg(t3, k) - avg(t1_tgt, k), 1) for k in ("P", "R", "F1")
        },
    }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["table4"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
