#!/usr/bin/env python3
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


@dataclass
class FeatureImageScore:
    image_index: int
    image_path: str
    score: float
    step_file_path: str
    clean_image_path: Optional[str] = None
    clean_step_file_path: Optional[str] = None
    clean_score: Optional[float] = None
    gap_score: Optional[float] = None


@dataclass
class SAEAggregateRow:
    sae_location: str
    top_features_json: str
    activations_dir: str
    selected_features: List[int]
    aggregate_images: List[FeatureImageScore]
    # First K keys from top-features list -> same image columns as aggregate (when multi-sae-per-feature-k > 0).
    per_feature_images: Optional[Dict[int, List[FeatureImageScore]]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize top SAE feature activations on images. Supports single-SAE mode "
            "and multi-SAE aggregation mode."
        )
    )
    parser.add_argument("--top-features-json", type=str, default=None)
    parser.add_argument("--activations-dir", type=str, default=None)
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Image directory aligned with --activations-dir (often attacked images). Pair with --clean-* for clean vs attack comparison.",
    )
    parser.add_argument("--num-features", type=int, default=5)
    parser.add_argument("--num-images-per-feature", type=int, default=5)
    parser.add_argument("--output-pdf", type=str, default="./cursor/top_feature_image_grid.pdf")
    parser.add_argument(
        "--output-pdf-heatmap-only",
        type=str,
        default=None,
        help=(
            "Optional extra PDF path for heatmap-only matrix (no image background). "
            "In multi-SAE mode, defaults to '<output-pdf stem>_heatmap_only.pdf'."
        ),
    )
    parser.add_argument("--output-json", type=str, default="./cursor/top_feature_image_grid.json")
    parser.add_argument(
        "--visualize-activated-patches",
        action="store_true",
        help="Overlay continuous patch heatmap on images.",
    )
    parser.add_argument(
        "--sae-locations",
        nargs="+",
        default=None,
        help="Multi-SAE mode. Example: projection-mlp2 vision-block10 vision-block0",
    )
    parser.add_argument(
        "--top-features-json-pattern",
        type=str,
        default=None,
        help="Must include {sae_location}.",
    )
    parser.add_argument(
        "--activations-dir-pattern",
        type=str,
        default=None,
        help="Must include {sae_location}.",
    )
    parser.add_argument(
        "--image-indices",
        type=str,
        default=None,
        help="Comma-separated image indices, e.g. 0,1,2,3,4",
    )
    parser.add_argument(
        "--clean-images-dir",
        type=str,
        default=None,
        help="Directory with clean (original) images, same indices as --images-dir.",
    )
    parser.add_argument(
        "--clean-activations-dir",
        type=str,
        default=None,
        help="Single-SAE mode: directory of cached .pt activations on clean images.",
    )
    parser.add_argument(
        "--clean-activations-dir-pattern",
        type=str,
        default=None,
        help="Multi-SAE mode: pattern with {sae_location}, e.g. ./sae_activations/DSET/{sae_location}/original-dev",
    )
    parser.add_argument(
        "--multi-sae-per-feature-k",
        type=int,
        default=0,
        help=(
            "Multi-SAE mode only: include rows for top-K features by rank (1..K) before each SAE's aggregate row. "
            "0 disables (aggregate only)."
        ),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.sae_locations:
        if not args.top_features_json_pattern or "{sae_location}" not in args.top_features_json_pattern:
            raise ValueError("Multi-SAE mode requires --top-features-json-pattern with {sae_location}.")
        if not args.activations_dir_pattern or "{sae_location}" not in args.activations_dir_pattern:
            raise ValueError("Multi-SAE mode requires --activations-dir-pattern with {sae_location}.")
        if args.clean_images_dir or args.clean_activations_dir_pattern:
            if not args.clean_images_dir:
                raise ValueError("Clean comparison requires --clean-images-dir.")
            if not args.clean_activations_dir_pattern or "{sae_location}" not in args.clean_activations_dir_pattern:
                raise ValueError("Multi-SAE clean comparison requires --clean-activations-dir-pattern with {sae_location}.")
    else:
        if not args.top_features_json:
            raise ValueError("Single-SAE mode requires --top-features-json.")
        if not args.activations_dir:
            raise ValueError("Single-SAE mode requires --activations-dir.")
        if args.clean_images_dir or args.clean_activations_dir:
            if not args.clean_images_dir:
                raise ValueError("Clean comparison requires --clean-images-dir.")
            if not args.clean_activations_dir:
                raise ValueError("Single-SAE clean comparison requires --clean-activations-dir.")


def _parse_image_indices(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or raw.strip() == "":
        return None
    out: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return out


def _extract_int_from_name(name: str) -> int:
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else -1


def _list_step_files(activations_dir: Path) -> List[Path]:
    files = [p for p in activations_dir.iterdir() if p.is_file() and p.suffix == ".pt"]
    files.sort(key=lambda p: (_extract_int_from_name(p.stem), p.name))
    return files


def _load_top_features(path: Path, n: int) -> List[int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "top_1000_features" not in data:
        raise KeyError(f"Missing 'top_1000_features' in {path}")
    return [int(x) for x in data["top_1000_features"][:n]]


def _resolve_image_path(images_dir: Path, image_idx: int) -> Path:
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = images_dir / f"{image_idx}{ext}"
        if p.exists():
            return p
    return images_dir / f"{image_idx}.png"


def _squeeze_to_seq_k(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 3 and t.size(0) == 1:
        return t.squeeze(0)
    return t


def _feature_scores_for_step(step_data: Dict[str, torch.Tensor], features: List[int]) -> Dict[int, float]:
    activations = _squeeze_to_seq_k(step_data["activations"])
    indices = _squeeze_to_seq_k(step_data["indices"])
    scores: Dict[int, float] = {}
    for feature in features:
        mask = indices == feature
        if torch.any(mask):
            scores[feature] = float(activations[mask].max().item())
        else:
            scores[feature] = 0.0
    return scores


def collect_top_images(
    step_files: List[Path],
    images_dir: Path,
    features: List[int],
    num_images_per_feature: int,
    image_indices: Optional[List[int]] = None,
) -> Tuple[Dict[int, List[FeatureImageScore]], List[FeatureImageScore]]:
    per_feature: Dict[int, List[FeatureImageScore]] = {f: [] for f in features}
    aggregate: List[FeatureImageScore] = []
    selected = set(image_indices) if image_indices is not None else None

    for fallback_idx, step_file in enumerate(step_files):
        step_data = torch.load(step_file, map_location="cpu")
        step_idx = _extract_int_from_name(step_file.stem)
        image_idx = step_idx if step_idx >= 0 else fallback_idx
        if selected is not None and image_idx not in selected:
            continue

        image_path = _resolve_image_path(images_dir, image_idx)
        f_scores = _feature_scores_for_step(step_data, features)
        agg_score = float(sum(f_scores.values()))

        aggregate.append(
            FeatureImageScore(
                image_index=image_idx,
                image_path=str(image_path),
                score=agg_score,
                step_file_path=str(step_file),
            )
        )
        for f in features:
            per_feature[f].append(
                FeatureImageScore(
                    image_index=image_idx,
                    image_path=str(image_path),
                    score=f_scores[f],
                    step_file_path=str(step_file),
                )
            )

    # If user pinned indices, keep that exact order.
    if image_indices is not None:
        agg_by_idx = {x.image_index: x for x in aggregate}
        ordered_agg: List[FeatureImageScore] = []
        for idx in image_indices:
            if idx in agg_by_idx:
                ordered_agg.append(agg_by_idx[idx])
            else:
                ordered_agg.append(
                    FeatureImageScore(
                        image_index=idx,
                        image_path=str(_resolve_image_path(images_dir, idx)),
                        score=0.0,
                        step_file_path="",
                    )
                )

        topk: Dict[int, List[FeatureImageScore]] = {}
        for f in features:
            by_idx = {x.image_index: x for x in per_feature[f]}
            ordered: List[FeatureImageScore] = []
            for idx in image_indices:
                if idx in by_idx:
                    ordered.append(by_idx[idx])
                else:
                    ordered.append(
                        FeatureImageScore(
                            image_index=idx,
                            image_path=str(_resolve_image_path(images_dir, idx)),
                            score=0.0,
                            step_file_path="",
                        )
                    )
            topk[f] = ordered
        return topk, ordered_agg

    for f, items in per_feature.items():
        items.sort(key=lambda x: x.score, reverse=True)
        per_feature[f] = items[:num_images_per_feature]
    aggregate.sort(key=lambda x: x.score, reverse=True)
    return per_feature, aggregate[:num_images_per_feature]


def _feature_patch_scores(indices: torch.Tensor, activations: torch.Tensor, feature: int) -> np.ndarray:
    idx_2d = _squeeze_to_seq_k(indices)
    act_2d = _squeeze_to_seq_k(activations)
    if idx_2d.shape != act_2d.shape or idx_2d.dim() != 2:
        raise ValueError(f"indices/activations shape mismatch: {tuple(idx_2d.shape)} vs {tuple(act_2d.shape)}")
    match = idx_2d == feature
    vals = torch.where(match, act_2d, torch.zeros_like(act_2d))
    token_scores = vals.max(dim=1).values
    token_scores = torch.clamp(token_scores, min=0.0)
    return token_scores.cpu().numpy().astype(np.float32)


def _aggregate_feature_patch_scores(indices: torch.Tensor, activations: torch.Tensor, features: List[int]) -> np.ndarray:
    if not features:
        return np.array([], dtype=np.float32)
    total: Optional[np.ndarray] = None
    for f in features:
        s = _feature_patch_scores(indices, activations, f)
        total = s if total is None else (total + s)
    return total if total is not None else np.array([], dtype=np.float32)


def _discrete_gap_patch_scores(
    attack_patch: np.ndarray,
    clean_patch: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Per-patch gap: attack value only where attacked activates and clean does not; 0 if both activate or else."""
    a = attack_patch > eps
    c = clean_patch > eps
    out = np.zeros_like(attack_patch, dtype=np.float32)
    new_attack = (~c) & a
    out[new_attack] = attack_patch[new_attack].astype(np.float32)
    return out


def _gap_patch_scores_from_steps(
    attack_step: Dict[str, torch.Tensor],
    clean_step: Dict[str, torch.Tensor],
    overlay_single_feature: Optional[int],
    features_for_agg: Optional[List[int]],
) -> np.ndarray:
    a_idx, a_act = attack_step["indices"], attack_step["activations"]
    c_idx, c_act = clean_step["indices"], clean_step["activations"]
    if overlay_single_feature is not None:
        pa = _feature_patch_scores(a_idx, a_act, overlay_single_feature)
        pc = _feature_patch_scores(c_idx, c_act, overlay_single_feature)
    else:
        pa = _aggregate_feature_patch_scores(a_idx, a_act, features_for_agg or [])
        pc = _aggregate_feature_patch_scores(c_idx, c_act, features_for_agg or [])
    return _discrete_gap_patch_scores(pa, pc)


def _scalar_discrete_gap_from_steps(
    attack_step: Dict[str, torch.Tensor],
    clean_step: Dict[str, torch.Tensor],
    overlay_single_feature: Optional[int],
    features_for_agg: Optional[List[int]],
) -> float:
    g = _gap_patch_scores_from_steps(attack_step, clean_step, overlay_single_feature, features_for_agg)
    return float(np.sum(g))


def _build_image_index_to_step(activations_dir: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for p in _list_step_files(activations_dir):
        idx = _extract_int_from_name(p.stem)
        if idx >= 0:
            out[idx] = p
    return out


def enrich_with_clean_activations(
    topk: Dict[int, List[FeatureImageScore]],
    aggregate_topk: List[FeatureImageScore],
    features: List[int],
    clean_activations_dir: Path,
    clean_images_dir: Path,
) -> Tuple[Dict[int, List[FeatureImageScore]], List[FeatureImageScore]]:
    idx_map = _build_image_index_to_step(clean_activations_dir)

    def one(item: FeatureImageScore, single_f: Optional[int]) -> FeatureImageScore:
        cstep = idx_map.get(item.image_index)
        clean_img = str(_resolve_image_path(clean_images_dir, item.image_index))
        if cstep is None or not cstep.exists():
            return FeatureImageScore(
                image_index=item.image_index,
                image_path=item.image_path,
                score=item.score,
                step_file_path=item.step_file_path,
                clean_image_path=clean_img,
                clean_step_file_path=str(cstep) if cstep else "",
                clean_score=None,
                gap_score=None,
            )
        step_data = torch.load(cstep, map_location="cpu")
        if single_f is not None:
            f_scores = _feature_scores_for_step(step_data, [single_f])
            clean_s = float(f_scores.get(single_f, 0.0))
        else:
            f_scores = _feature_scores_for_step(step_data, features)
            clean_s = float(sum(f_scores.values()))
        if not item.step_file_path or not Path(item.step_file_path).exists():
            gap: Optional[float] = None
        else:
            attack_data = torch.load(item.step_file_path, map_location="cpu")
            gap = _scalar_discrete_gap_from_steps(attack_data, step_data, single_f, features)
        return FeatureImageScore(
            image_index=item.image_index,
            image_path=item.image_path,
            score=item.score,
            step_file_path=item.step_file_path,
            clean_image_path=clean_img,
            clean_step_file_path=str(cstep),
            clean_score=clean_s,
            gap_score=gap,
        )

    new_topk = {f: [one(it, f) for it in items] for f, items in topk.items()}
    new_agg = [one(it, None) for it in aggregate_topk]
    return new_topk, new_agg


def _normalize_patch_scores(patch_scores: np.ndarray) -> np.ndarray:
    seq_len = int(patch_scores.shape[0])
    scale = float(np.percentile(patch_scores, 98)) if seq_len > 0 else 0.0
    if scale > 0:
        norm_scores = patch_scores / scale
    else:
        norm_scores = patch_scores
    norm_scores = np.clip(norm_scores, 0.0, 1.0)
    return np.power(norm_scores, 0.85)


def _overlay_patch_heatmap(image: Image.Image, patch_scores: np.ndarray) -> Image.Image:
    seq_len = int(patch_scores.shape[0])
    side = int(round(seq_len ** 0.5))
    if side * side != seq_len:
        return image

    norm_scores = _normalize_patch_scores(patch_scores)

    img_arr = np.array(image, dtype=np.float32)
    h, w = img_arr.shape[:2]
    y_edges = np.linspace(0, h, side + 1, dtype=int)
    x_edges = np.linspace(0, w, side + 1, dtype=int)
    heat_arr = np.zeros_like(img_arr, dtype=np.float32)
    alpha_arr = np.zeros((h, w, 1), dtype=np.float32)
    cmap = plt.get_cmap("autumn")
    max_alpha = 0.9
    min_alpha_nonzero = 0.2

    for token_idx in range(seq_len):
        score = float(norm_scores[token_idx])
        score = max(0.0, min(1.0, score))
        row = token_idx // side
        col = token_idx % side
        y0, y1 = y_edges[row], y_edges[row + 1]
        x0, x1 = x_edges[col], x_edges[col + 1]
        heat_rgb = np.array(cmap(score)[:3], dtype=np.float32) * 255.0
        heat_arr[y0:y1, x0:x1, :] = heat_rgb
        if score >= 0.1:
            alpha = min_alpha_nonzero + (max_alpha - min_alpha_nonzero) * score
        else:
            alpha = 0.0
        alpha_arr[y0:y1, x0:x1, 0] = alpha

    blended = img_arr * (1.0 - alpha_arr) + heat_arr * alpha_arr
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def _heatmap_only_image(
    patch_scores: np.ndarray,
    width: int,
    height: int,
) -> Image.Image:
    seq_len = int(patch_scores.shape[0])
    side = int(round(seq_len ** 0.5))
    if side * side != seq_len or width <= 0 or height <= 0:
        return Image.new("RGB", (max(width, 1), max(height, 1)), color=(0, 0, 0))

    norm_scores = _normalize_patch_scores(patch_scores)
    y_edges = np.linspace(0, height, side + 1, dtype=int)
    x_edges = np.linspace(0, width, side + 1, dtype=int)
    heat_arr = np.zeros((height, width, 3), dtype=np.float32)
    alpha_arr = np.zeros((height, width, 1), dtype=np.float32)
    cmap = plt.get_cmap("autumn")
    max_alpha = 0.9
    min_alpha_nonzero = 0.2

    for token_idx in range(seq_len):
        score = float(norm_scores[token_idx])
        score = max(0.0, min(1.0, score))
        row = token_idx // side
        col = token_idx % side
        y0, y1 = y_edges[row], y_edges[row + 1]
        x0, x1 = x_edges[col], x_edges[col + 1]
        heat_rgb = np.array(cmap(score)[:3], dtype=np.float32) * 255.0
        heat_arr[y0:y1, x0:x1, :] = heat_rgb
        if score >= 0.1:
            alpha = min_alpha_nonzero + (max_alpha - min_alpha_nonzero) * score
        else:
            alpha = 0.0
        alpha_arr[y0:y1, x0:x1, 0] = alpha

    # Keep identical threshold/alpha logic with overlay heatmap:
    # low-score patches are suppressed, high-score patches stay vivid.
    heat_only = heat_arr * alpha_arr
    return Image.fromarray(np.clip(heat_only, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _safe_load_image(path: Path) -> Optional[Image.Image]:
    if not path.exists():
        return None
    with Image.open(path) as img:
        return img.convert("RGB")


def _draw_cell(
    ax,
    item: FeatureImageScore,
    title: str,
    features_for_overlay: Optional[List[int]],
    overlay_single_feature: Optional[int],
    use_overlay: bool,
    *,
    image_path: Optional[str] = None,
    step_file_path: Optional[str] = None,
) -> None:
    path = image_path if image_path is not None else item.image_path
    step_path = step_file_path if step_file_path is not None else item.step_file_path
    img = _safe_load_image(Path(path))
    if img is None:
        ax.text(0.5, 0.5, "Image not found", ha="center", va="center")
    else:
        if use_overlay and step_path:
            step_data = torch.load(step_path, map_location="cpu")
            if overlay_single_feature is not None:
                patch_scores = _feature_patch_scores(step_data["indices"], step_data["activations"], overlay_single_feature)
            else:
                patch_scores = _aggregate_feature_patch_scores(
                    step_data["indices"], step_data["activations"], features_for_overlay or []
                )
            img = _overlay_patch_heatmap(img, patch_scores)
        ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_gap_cell(
    ax,
    item: FeatureImageScore,
    title: str,
    features_for_overlay: Optional[List[int]],
    overlay_single_feature: Optional[int],
    use_overlay: bool,
    *,
    background_image_path: Optional[str] = None,
) -> None:
    bg = background_image_path if background_image_path is not None else item.image_path
    img = _safe_load_image(Path(bg))
    if img is None:
        ax.text(0.5, 0.5, "Image not found", ha="center", va="center")
    elif not use_overlay or not item.step_file_path:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center")
    elif not item.clean_step_file_path or not Path(item.clean_step_file_path).exists():
        ax.text(0.5, 0.5, "No clean .pt", ha="center", va="center")
    else:
        attack_data = torch.load(item.step_file_path, map_location="cpu")
        clean_data = torch.load(item.clean_step_file_path, map_location="cpu")
        gap = _gap_patch_scores_from_steps(attack_data, clean_data, overlay_single_feature, features_for_overlay)
        img = _overlay_patch_heatmap(img, gap)
        ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def render_single_sae_pdf(
    topk: Dict[int, List[FeatureImageScore]],
    aggregate_topk: List[FeatureImageScore],
    features: List[int],
    num_columns: int,
    output_pdf: Path,
    use_overlay: bool,
    compare_clean: bool = False,
) -> None:
    # Without clean: one row per feature + one aggregate row.
    # With clean: three rows per block (clean, attack, gap).
    n_blocks = len(features) + 1
    rows = n_blocks * 3 if compare_clean else n_blocks
    fig, axes = plt.subplots(rows, num_columns, figsize=(3.2 * num_columns, 3.2 * rows))
    if rows == 1 and num_columns == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif num_columns == 1:
        axes = [[ax] for ax in axes]

    def row_index(block: int, sub: int) -> int:
        if compare_clean:
            return block * 3 + sub
        return block

    for fi, feature in enumerate(features):
        items = topk.get(feature, [])
        br = row_index(fi, 0)
        if compare_clean:
            for c in range(num_columns):
                ax = axes[br][c]
                if c < len(items):
                    item = items[c]
                    ct = (
                        f"img {item.image_index}\ncln={item.clean_score:.4f}"
                        if item.clean_score is not None
                        else f"img {item.image_index}\ncln=N/A"
                    )
                    _draw_cell(
                        ax=ax,
                        item=item,
                        title=ct,
                        features_for_overlay=None,
                        overlay_single_feature=feature,
                        use_overlay=use_overlay,
                        image_path=item.clean_image_path,
                        step_file_path=item.clean_step_file_path,
                    )
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                    ax.set_xticks([])
                    ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(f"feature {feature}\nclean", fontsize=9)
            for c in range(num_columns):
                ax = axes[br + 1][c]
                if c < len(items):
                    item = items[c]
                    _draw_cell(
                        ax=ax,
                        item=item,
                        title=f"img {item.image_index}\natk={item.score:.4f}",
                        features_for_overlay=None,
                        overlay_single_feature=feature,
                        use_overlay=use_overlay,
                    )
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                    ax.set_xticks([])
                    ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel("attacked", fontsize=9)
            for c in range(num_columns):
                ax = axes[br + 2][c]
                if c < len(items):
                    item = items[c]
                    gt = (
                        f"img {item.image_index}\nΔ={item.gap_score:.4f}"
                        if item.gap_score is not None
                        else f"img {item.image_index}\nΔ=N/A"
                    )
                    _draw_gap_cell(
                        ax=ax,
                        item=item,
                        title=gt,
                        features_for_overlay=None,
                        overlay_single_feature=feature,
                        use_overlay=use_overlay,
                    )
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                    ax.set_xticks([])
                    ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel("Δ (new atk)", fontsize=9)
        else:
            for c in range(num_columns):
                ax = axes[br][c]
                if c < len(items):
                    item = items[c]
                    _draw_cell(
                        ax=ax,
                        item=item,
                        title=f"img {item.image_index}\nact={item.score:.4f}",
                        features_for_overlay=None,
                        overlay_single_feature=feature,
                        use_overlay=use_overlay,
                    )
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                    ax.set_xticks([])
                    ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(f"feature {feature}", fontsize=10)

    agg_block = len(features)
    ar = row_index(agg_block, 0)
    if compare_clean:
        for c in range(num_columns):
            ax = axes[ar][c]
            if c < len(aggregate_topk):
                item = aggregate_topk[c]
                ct = (
                    f"img {item.image_index}\ncln={item.clean_score:.4f}"
                    if item.clean_score is not None
                    else f"img {item.image_index}\ncln=N/A"
                )
                _draw_cell(
                    ax=ax,
                    item=item,
                    title=ct,
                    features_for_overlay=features,
                    overlay_single_feature=None,
                    use_overlay=use_overlay,
                    image_path=item.clean_image_path,
                    step_file_path=item.clean_step_file_path,
                )
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"top-{len(features)} agg\nclean", fontsize=9)
        for c in range(num_columns):
            ax = axes[ar + 1][c]
            if c < len(aggregate_topk):
                item = aggregate_topk[c]
                _draw_cell(
                    ax=ax,
                    item=item,
                    title=f"img {item.image_index}\natk={item.score:.4f}",
                    features_for_overlay=features,
                    overlay_single_feature=None,
                    use_overlay=use_overlay,
                )
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            if c == 0:
                ax.set_ylabel("attacked", fontsize=9)
        for c in range(num_columns):
            ax = axes[ar + 2][c]
            if c < len(aggregate_topk):
                item = aggregate_topk[c]
                gt = (
                    f"img {item.image_index}\nΔ={item.gap_score:.4f}"
                    if item.gap_score is not None
                    else f"img {item.image_index}\nΔ=N/A"
                )
                _draw_gap_cell(
                    ax=ax,
                    item=item,
                    title=gt,
                    features_for_overlay=features,
                    overlay_single_feature=None,
                    use_overlay=use_overlay,
                )
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            if c == 0:
                ax.set_ylabel("Δ (new atk)", fontsize=9)
    else:
        for c in range(num_columns):
            ax = axes[ar][c]
            if c < len(aggregate_topk):
                item = aggregate_topk[c]
                _draw_cell(
                    ax=ax,
                    item=item,
                    title=f"img {item.image_index}\nagg={item.score:.4f}",
                    features_for_overlay=features,
                    overlay_single_feature=None,
                    use_overlay=use_overlay,
                )
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"top-{len(features)} agg", fontsize=10)

    title = (
        "Clean vs attacked SAE activations and new-attack-only gap (per feature + aggregate)"
        if compare_clean
        else "Top Activated Dev Images per Feature + Aggregation"
    )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_multi_sae_aggregation_pdf(
    rows: List[SAEAggregateRow],
    num_columns: int,
    output_pdf: Path,
    use_overlay: bool,
    compare_clean: bool = False,
    per_feature_k: int = 0,
) -> None:
    # Rows per SAE: optional top-K single-feature blocks (#1..#K), then aggregate. Each block is 1 row
    # (no clean comparison) or 3 rows (clean / attacked / gap).
    if not rows:
        return
    K = min(per_feature_k, len(rows[0].selected_features)) if per_feature_k > 0 else 0
    blocks_per_sae = K + 1
    row_h = 3 if compare_clean else 1
    ref_rows = 0 if compare_clean else 1
    total_rows = ref_rows + len(rows) * blocks_per_sae * row_h
    fig, axes = plt.subplots(total_rows, num_columns, figsize=(3.2 * num_columns, 3.2 * total_rows))
    if total_rows == 1 and num_columns == 1:
        axes = [[axes]]
    elif total_rows == 1:
        axes = [axes]
    elif num_columns == 1:
        axes = [[ax] for ax in axes]

    rr = 0
    if ref_rows:
        reference = rows[0].aggregate_images
        for c in range(num_columns):
            ax = axes[rr][c]
            if c < len(reference):
                item = reference[c]
                _draw_cell(
                    ax=ax,
                    item=item,
                    title=f"img {item.image_index}",
                    features_for_overlay=None,
                    overlay_single_feature=None,
                    use_overlay=False,
                )
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            if c == 0:
                ax.set_ylabel("original", fontsize=10)
        rr += 1

    for row in rows:
        pf = row.per_feature_images or {}
        for b in range(blocks_per_sae):
            is_agg = K == 0 or b == K
            if is_agg:
                items = row.aggregate_images
                fid: Optional[int] = None
                rank_suffix = "agg (top-N sum)"
            else:
                fid = row.selected_features[b]
                items = pf.get(fid, [])
                rank_suffix = f"#{b + 1} (f{fid})"

            if compare_clean:
                for c in range(num_columns):
                    ax = axes[rr][c]
                    if c < len(items):
                        item = items[c]
                        ct = (
                            f"img {item.image_index}\ncln={item.clean_score:.4f}"
                            if item.clean_score is not None
                            else f"img {item.image_index}\ncln=N/A"
                        )
                        _draw_cell(
                            ax=ax,
                            item=item,
                            title=ct,
                            features_for_overlay=row.selected_features if is_agg else None,
                            overlay_single_feature=None if is_agg else fid,
                            use_overlay=use_overlay,
                            image_path=item.clean_image_path,
                            step_file_path=item.clean_step_file_path,
                        )
                    else:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                        ax.set_xticks([])
                        ax.set_yticks([])
                    if c == 0:
                        ax.set_ylabel(f"{row.sae_location}\n{rank_suffix}\nclean", fontsize=8)
                rr += 1
                for c in range(num_columns):
                    ax = axes[rr][c]
                    if c < len(items):
                        item = items[c]
                        _draw_cell(
                            ax=ax,
                            item=item,
                            title=f"img {item.image_index}\natk={item.score:.4f}",
                            features_for_overlay=row.selected_features if is_agg else None,
                            overlay_single_feature=None if is_agg else fid,
                            use_overlay=use_overlay,
                        )
                    else:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                        ax.set_xticks([])
                        ax.set_yticks([])
                    if c == 0:
                        ax.set_ylabel("attacked", fontsize=9)
                rr += 1
                for c in range(num_columns):
                    ax = axes[rr][c]
                    if c < len(items):
                        item = items[c]
                        gt = (
                            f"img {item.image_index}\nΔ={item.gap_score:.4f}"
                            if item.gap_score is not None
                            else f"img {item.image_index}\nΔ=N/A"
                        )
                        _draw_gap_cell(
                            ax=ax,
                            item=item,
                            title=gt,
                            features_for_overlay=row.selected_features if is_agg else None,
                            overlay_single_feature=None if is_agg else fid,
                            use_overlay=use_overlay,
                        )
                    else:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                        ax.set_xticks([])
                        ax.set_yticks([])
                    if c == 0:
                        ax.set_ylabel("Δ (new atk)", fontsize=9)
                rr += 1
            else:
                for c in range(num_columns):
                    ax = axes[rr][c]
                    if c < len(items):
                        item = items[c]
                        title = (
                            f"img {item.image_index}\nagg={item.score:.4f}"
                            if is_agg
                            else f"img {item.image_index}\n#{b + 1} act={item.score:.4f}"
                        )
                        _draw_cell(
                            ax=ax,
                            item=item,
                            title=title,
                            features_for_overlay=row.selected_features if is_agg else None,
                            overlay_single_feature=None if is_agg else fid,
                            use_overlay=use_overlay,
                        )
                    else:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                        ax.set_xticks([])
                        ax.set_yticks([])
                    if c == 0:
                        yl = f"{row.sae_location}\n{rank_suffix}" if K > 0 else row.sae_location
                        ax.set_ylabel(yl, fontsize=9 if K > 0 else 10)
                rr += 1

    if K > 0:
        st = (
            "Multi-SAE: top-K features (by rank) + aggregate per SAE"
            + (" (clean / attacked / gap)" if compare_clean else "")
        )
    else:
        st = (
            "Multi-SAE: clean vs attacked activations and new-attack-only gap"
            if compare_clean
            else "Multi-SAE aggregation (reference row + per-SAE overlays)"
        )
    fig.suptitle(st, fontsize=13)
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_multi_sae_heatmap_only_pdf(
    rows: List[SAEAggregateRow],
    num_columns: int,
    output_pdf: Path,
    compare_clean: bool = False,
    per_feature_k: int = 0,
) -> None:
    if not rows:
        return
    K = min(per_feature_k, len(rows[0].selected_features)) if per_feature_k > 0 else 0
    blocks_per_sae = K + 1
    row_h = 3 if compare_clean else 1
    n_row = len(rows) * blocks_per_sae * row_h
    fig, axes = plt.subplots(n_row, num_columns, figsize=(3.2 * num_columns, 3.2 * n_row))
    if n_row == 1 and num_columns == 1:
        axes = [[axes]]
    elif n_row == 1:
        axes = [axes]
    elif num_columns == 1:
        axes = [[ax] for ax in axes]

    rr = 0
    for row in rows:
        pf = row.per_feature_images or {}
        for b in range(blocks_per_sae):
            is_agg = K == 0 or b == K
            if is_agg:
                items = row.aggregate_images
                fid: Optional[int] = None
                y_clean = f"{row.sae_location}\nagg\nclean"
                y_rank = row.sae_location
            else:
                fid = row.selected_features[b]
                items = pf.get(fid, [])
                y_clean = f"{row.sae_location}\n#{b + 1}\nclean"
                y_rank = f"{row.sae_location}\n#{b + 1}"

            if compare_clean:
                for c in range(num_columns):
                    ax = axes[rr][c]
                    if c < len(items):
                        item = items[c]
                        img = _safe_load_image(Path(item.clean_image_path or item.image_path))
                        width, height = (img.size if img is not None else (336, 336))
                        if item.clean_step_file_path:
                            step_data = torch.load(item.clean_step_file_path, map_location="cpu")
                            if is_agg:
                                patch_scores = _aggregate_feature_patch_scores(
                                    step_data["indices"],
                                    step_data["activations"],
                                    row.selected_features,
                                )
                            else:
                                patch_scores = _feature_patch_scores(
                                    step_data["indices"],
                                    step_data["activations"],
                                    fid,
                                )
                            hm = _heatmap_only_image(patch_scores, width=width, height=height)
                            ax.imshow(hm)
                            cs = item.clean_score
                            ax.set_title(
                                f"img {item.image_index}\ncln={cs:.4f}" if cs is not None else f"img {item.image_index}\ncln=N/A",
                                fontsize=9,
                            )
                        else:
                            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                    else:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                    if c == 0:
                        ax.set_ylabel(y_clean, fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                rr += 1
                for c in range(num_columns):
                    ax = axes[rr][c]
                    if c < len(items):
                        item = items[c]
                        img = _safe_load_image(Path(item.image_path))
                        width, height = (img.size if img is not None else (336, 336))
                        if item.step_file_path:
                            step_data = torch.load(item.step_file_path, map_location="cpu")
                            if is_agg:
                                patch_scores = _aggregate_feature_patch_scores(
                                    step_data["indices"],
                                    step_data["activations"],
                                    row.selected_features,
                                )
                            else:
                                patch_scores = _feature_patch_scores(
                                    step_data["indices"],
                                    step_data["activations"],
                                    fid,
                                )
                            hm = _heatmap_only_image(patch_scores, width=width, height=height)
                            ax.imshow(hm)
                            ax.set_title(f"img {item.image_index}\natk={item.score:.4f}", fontsize=9)
                        else:
                            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                    else:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                    if c == 0:
                        ax.set_ylabel("attacked", fontsize=9)
                    ax.set_xticks([])
                    ax.set_yticks([])
                rr += 1
                for c in range(num_columns):
                    ax = axes[rr][c]
                    if c < len(items):
                        item = items[c]
                        img = _safe_load_image(Path(item.image_path))
                        width, height = (img.size if img is not None else (336, 336))
                        if item.step_file_path and item.clean_step_file_path:
                            attack_data = torch.load(item.step_file_path, map_location="cpu")
                            clean_data = torch.load(item.clean_step_file_path, map_location="cpu")
                            gap = _gap_patch_scores_from_steps(
                                attack_data,
                                clean_data,
                                None if is_agg else fid,
                                row.selected_features if is_agg else None,
                            )
                            hm = _heatmap_only_image(gap, width=width, height=height)
                            ax.imshow(hm)
                            gs = item.gap_score
                            ax.set_title(
                                f"img {item.image_index}\nΔ={gs:.4f}" if gs is not None else f"img {item.image_index}\nΔ=N/A",
                                fontsize=9,
                            )
                        else:
                            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                    else:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                    if c == 0:
                        ax.set_ylabel("Δ (new atk)", fontsize=9)
                    ax.set_xticks([])
                    ax.set_yticks([])
                rr += 1
            else:
                for c in range(num_columns):
                    ax = axes[rr][c]
                    if c < len(items):
                        item = items[c]
                        img = _safe_load_image(Path(item.image_path))
                        width, height = (img.size if img is not None else (336, 336))
                        if item.step_file_path:
                            step_data = torch.load(item.step_file_path, map_location="cpu")
                            if is_agg:
                                patch_scores = _aggregate_feature_patch_scores(
                                    step_data["indices"],
                                    step_data["activations"],
                                    row.selected_features,
                                )
                                tit = f"img {item.image_index}\nagg={item.score:.4f}"
                            else:
                                patch_scores = _feature_patch_scores(
                                    step_data["indices"],
                                    step_data["activations"],
                                    fid,
                                )
                                tit = f"img {item.image_index}\n#{b + 1} act={item.score:.4f}"
                            hm = _heatmap_only_image(patch_scores, width=width, height=height)
                            ax.imshow(hm)
                            ax.set_title(tit, fontsize=9)
                        else:
                            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                    else:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                    if c == 0:
                        ax.set_ylabel(y_rank if K > 0 and not is_agg else row.sae_location, fontsize=9 if K > 0 else 10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                rr += 1

    fig.suptitle(
        "Multi-SAE heatmaps: top-K + aggregate (clean / attacked / gap)"
        if compare_clean and K > 0
        else (
            "Multi-SAE heatmaps: clean, attacked, new-attack-only gap"
            if compare_clean
            else ("Multi-SAE heatmaps: top-K + aggregate" if K > 0 else "Multi-SAE Aggregation Heatmaps Only")
        ),
        fontsize=14,
    )
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_single_json(
    topk: Dict[int, List[FeatureImageScore]],
    aggregate_topk: List[FeatureImageScore],
    selected_features: List[int],
    output_json: Path,
    top_features_json: Path,
    activations_dir: Path,
    images_dir: Path,
    image_indices: Optional[List[int]],
    clean_images_dir: Optional[Path] = None,
    clean_activations_dir: Optional[Path] = None,
) -> None:
    payload = {
        "mode": "single_sae",
        "top_features_json": str(top_features_json),
        "activations_dir": str(activations_dir),
        "images_dir": str(images_dir),
        "clean_images_dir": str(clean_images_dir) if clean_images_dir else None,
        "clean_activations_dir": str(clean_activations_dir) if clean_activations_dir else None,
        "image_indices": image_indices,
        "selected_features_for_aggregation": selected_features,
        "features": {
            str(feature): [
                {
                    "image_index": item.image_index,
                    "image_path": item.image_path,
                    "score": item.score,
                    "step_file_path": item.step_file_path,
                    "clean_image_path": item.clean_image_path,
                    "clean_step_file_path": item.clean_step_file_path,
                    "clean_score": item.clean_score,
                    "gap_score_discrete_new_attack": item.gap_score,
                }
                for item in items
            ]
            for feature, items in topk.items()
        },
        "aggregation_top_images": [
            {
                "image_index": item.image_index,
                "image_path": item.image_path,
                "score": item.score,
                "step_file_path": item.step_file_path,
                "clean_image_path": item.clean_image_path,
                "clean_step_file_path": item.clean_step_file_path,
                "clean_score": item.clean_score,
                "gap_score_discrete_new_attack": item.gap_score,
            }
            for item in aggregate_topk
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_multi_json(
    rows: List[SAEAggregateRow],
    output_json: Path,
    images_dir: Path,
    image_indices: Optional[List[int]],
    clean_images_dir: Optional[Path] = None,
    clean_activations_dir_pattern: Optional[str] = None,
    multi_sae_per_feature_k: int = 0,
) -> None:
    payload = {
        "mode": "multi_sae_aggregation",
        "images_dir": str(images_dir),
        "clean_images_dir": str(clean_images_dir) if clean_images_dir else None,
        "clean_activations_dir_pattern": clean_activations_dir_pattern,
        "multi_sae_per_feature_k": multi_sae_per_feature_k,
        "image_indices": image_indices,
        "rows": [
            {
                "sae_location": row.sae_location,
                "top_features_json": row.top_features_json,
                "activations_dir": row.activations_dir,
                "selected_features_for_aggregation": row.selected_features,
                "per_feature_images_by_id": (
                    {
                        str(fid): [
                            {
                                "image_index": item.image_index,
                                "image_path": item.image_path,
                                "score": item.score,
                                "step_file_path": item.step_file_path,
                                "clean_image_path": item.clean_image_path,
                                "clean_step_file_path": item.clean_step_file_path,
                                "clean_score": item.clean_score,
                                "gap_score_discrete_new_attack": item.gap_score,
                            }
                            for item in items
                        ]
                        for fid, items in (row.per_feature_images or {}).items()
                    }
                    if row.per_feature_images
                    else None
                ),
                "aggregation_images": [
                    {
                        "image_index": item.image_index,
                        "image_path": item.image_path,
                        "score": item.score,
                        "step_file_path": item.step_file_path,
                        "clean_image_path": item.clean_image_path,
                        "clean_step_file_path": item.clean_step_file_path,
                        "clean_score": item.clean_score,
                        "gap_score_discrete_new_attack": item.gap_score,
                    }
                    for item in row.aggregate_images
                ],
            }
            for row in rows
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    _validate_args(args)

    images_dir = Path(args.images_dir)
    output_pdf = Path(args.output_pdf)
    output_pdf_heatmap_only = (
        Path(args.output_pdf_heatmap_only)
        if args.output_pdf_heatmap_only
        else output_pdf.with_name(f"{output_pdf.stem}_heatmap_only{output_pdf.suffix}")
    )
    output_json = Path(args.output_json)
    image_indices = _parse_image_indices(args.image_indices)
    num_columns = len(image_indices) if image_indices is not None else args.num_images_per_feature
    compare_clean = args.clean_images_dir is not None
    clean_images_path = Path(args.clean_images_dir) if args.clean_images_dir else None

    if args.sae_locations:
        rows: List[SAEAggregateRow] = []
        for sae_location in args.sae_locations:
            top_json = Path(args.top_features_json_pattern.format(sae_location=sae_location))
            act_dir = Path(args.activations_dir_pattern.format(sae_location=sae_location))
            features = _load_top_features(top_json, args.num_features)
            step_files = _list_step_files(act_dir)
            if not step_files:
                raise FileNotFoundError(f"No .pt files in {act_dir}")
            topk, agg = collect_top_images(
                step_files=step_files,
                images_dir=images_dir,
                features=features,
                num_images_per_feature=num_columns,
                image_indices=image_indices,
            )
            if compare_clean and clean_images_path is not None and args.clean_activations_dir_pattern:
                clean_act_dir = Path(args.clean_activations_dir_pattern.format(sae_location=sae_location))
                topk, agg = enrich_with_clean_activations(topk, agg, features, clean_act_dir, clean_images_path)
            mk = args.multi_sae_per_feature_k
            k_vis = min(mk, len(features)) if mk > 0 else 0
            per_feat = {features[i]: topk[features[i]] for i in range(k_vis)} if k_vis else None
            rows.append(
                SAEAggregateRow(
                    sae_location=sae_location,
                    top_features_json=str(top_json),
                    activations_dir=str(act_dir),
                    selected_features=features,
                    aggregate_images=agg,
                    per_feature_images=per_feat,
                )
            )

        render_multi_sae_aggregation_pdf(
            rows=rows,
            num_columns=num_columns,
            output_pdf=output_pdf,
            use_overlay=args.visualize_activated_patches,
            compare_clean=compare_clean,
            per_feature_k=args.multi_sae_per_feature_k,
        )
        render_multi_sae_heatmap_only_pdf(
            rows=rows,
            num_columns=num_columns,
            output_pdf=output_pdf_heatmap_only,
            compare_clean=compare_clean,
            per_feature_k=args.multi_sae_per_feature_k,
        )
        write_multi_json(
            rows=rows,
            output_json=output_json,
            images_dir=images_dir,
            image_indices=image_indices,
            clean_images_dir=clean_images_path,
            clean_activations_dir_pattern=args.clean_activations_dir_pattern,
            multi_sae_per_feature_k=args.multi_sae_per_feature_k,
        )
        print(f"Saved PDF: {output_pdf}")
        print(f"Saved Heatmap-only PDF: {output_pdf_heatmap_only}")
        print(f"Saved JSON: {output_json}")
        return

    top_json = Path(args.top_features_json)
    act_dir = Path(args.activations_dir)
    features = _load_top_features(top_json, args.num_features)
    step_files = _list_step_files(act_dir)
    if not step_files:
        raise FileNotFoundError(f"No .pt files in {act_dir}")

    topk, agg = collect_top_images(
        step_files=step_files,
        images_dir=images_dir,
        features=features,
        num_images_per_feature=num_columns,
        image_indices=image_indices,
    )
    clean_act_path = Path(args.clean_activations_dir) if args.clean_activations_dir else None
    if compare_clean and clean_images_path is not None and clean_act_path is not None:
        topk, agg = enrich_with_clean_activations(topk, agg, features, clean_act_path, clean_images_path)
    render_single_sae_pdf(
        topk=topk,
        aggregate_topk=agg,
        features=features,
        num_columns=num_columns,
        output_pdf=output_pdf,
        use_overlay=args.visualize_activated_patches,
        compare_clean=compare_clean,
    )
    write_single_json(
        topk=topk,
        aggregate_topk=agg,
        selected_features=features,
        output_json=output_json,
        top_features_json=top_json,
        activations_dir=act_dir,
        images_dir=images_dir,
        image_indices=image_indices,
        clean_images_dir=clean_images_path,
        clean_activations_dir=clean_act_path,
    )
    print(f"Saved PDF: {output_pdf}")
    print(f"Saved JSON: {output_json}")


if __name__ == "__main__":
    main()
