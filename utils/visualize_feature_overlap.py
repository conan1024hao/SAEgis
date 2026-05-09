import json
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize top feature overlap using a 3-set Venn diagram.")
    parser.add_argument(
        "--nips17-feature-path",
        default=None,
        help="Path to NIPS17 top_features.json",
    )
    parser.add_argument(
        "--llava-feature-path",
        default=None,
        help="Path to LLaVA-Instruct-150K top_features.json",
    )
    parser.add_argument(
        "--medical-feature-path",
        default=None,
        help="Path to Medical-Multimodal-Eval top_features.json",
    )
    parser.add_argument("--topk", type=int, default=256, help="Number of top features to compare.")
    parser.add_argument(
        "--attack-method",
        default=None,
        help="Optional attack method label used in output filename (auto-inferred if omitted).",
    )
    parser.add_argument(
        "--sae-location",
        default=None,
        help="Optional SAE location label used in output filename (auto-inferred if omitted).",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output directory or .png file path. Defaults to top_<topk>_feature_overlap_venn_<attack>_<sae>.png in current directory.",
    )
    return parser.parse_args()


def infer_attack_and_sae(feature_path):
    # Expected shape: .../<attack_method>/<sae_location>/top_features.json
    p = Path(feature_path)
    if len(p.parts) >= 3:
        return p.parts[-3], p.parts[-2]
    return "unknown_attack", "unknown_sae"


def main():
    args = parse_args()
    dataset_A = json.load(open(args.nips17_feature_path, "r"))["top_1000_features"]
    dataset_B = json.load(open(args.llava_feature_path, "r"))["top_1000_features"]
    dataset_C = json.load(open(args.medical_feature_path, "r"))["top_1000_features"]

    topk = args.topk
    inferred_attack, inferred_sae = infer_attack_and_sae(args.nips17_feature_path)
    attack_method = args.attack_method or inferred_attack
    sae_location = args.sae_location or inferred_sae

    default_filename = f"top_{topk}_feature_overlap_venn_{attack_method}_{sae_location}.png"
    if args.output_path is None:
        output_path = Path.cwd() / default_filename
    else:
        candidate = Path(args.output_path)
        output_path = candidate if candidate.suffix.lower() == ".png" else candidate / default_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = str(output_path)

    A = set(dataset_A[:topk])
    B = set(dataset_B[:topk])
    C = set(dataset_C[:topk])

    # Compute regions
    only_A = len(A - B - C)
    only_B = len(B - A - C)
    only_C = len(C - A - B)

    A_B = len((A & B) - C)
    A_C = len((A & C) - B)
    B_C = len((B & C) - A)

    A_B_C = len(A & B & C)

    print("Region sizes:")
    print("Only A:", only_A)
    print("Only B:", only_B)
    print("Only C:", only_C)
    print("A∩B only:", A_B)
    print("A∩C only:", A_C)
    print("B∩C only:", B_C)
    print("A∩B∩C:", A_B_C)

    # Plot Venn diagram
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn3

    plt.figure()
    venn3(
        subsets=(only_A, only_B, A_B, only_C, A_C, B_C, A_B_C),
        set_labels=("NIPS17", "LLaVA-Instruct-150K", "Medical-Multimodal-Eval"),
    )
    plt.savefig(output_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()