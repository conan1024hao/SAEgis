"""Cache SAE activations for many image directories in one model load.

Reproducing the paper's tables needs activations for every
(dataset, split) and (dataset, attack, split) combination -- 27 directories and
~4800 images. Invoking cache_activations.py once per directory would reload the
5.1B-parameter base model 27 times, so this driver takes a job list instead and
walks it with a single model in memory.

    python cache_activations_batch.py --model_path ... --sae_model_path ... \
        --jobs jobs.tsv          # tab-separated: <images_path>\t<activation_path>

Directories whose activation_path already holds the expected number of .pt files
are skipped, so an interrupted run can simply be restarted.
"""

import argparse
import os

from tqdm import tqdm

from cache_activations import (
    VISION_SPAN_TOKEN_IDS,
    get_architecture,
    load_model_and_processor,
    prepare_inputs,
)
from sae import PeftSaeModel
from sae.models.topk_sae import Linear as SaeLinear


def parse_jobs(path):
    jobs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            images_path, activation_path = line.split("\t")
            jobs.append((images_path, activation_path))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--sae_model_path", required=True)
    ap.add_argument("--jobs", required=True, help="TSV of <images_path>\\t<activation_path>")
    ap.add_argument("--prompt", default="Describe this image.")
    args = ap.parse_args()

    jobs = parse_jobs(args.jobs)
    architecture = get_architecture(args.model_path)
    model, processor = load_model_and_processor(args.model_path, architecture)
    model_with_sae = PeftSaeModel.from_pretrained(
        model, args.sae_model_path, adapter_name="default", low_cpu_mem_usage=True
    )

    hooked_module = next(m for m in model.model.modules() if isinstance(m, SaeLinear))
    hooked_module.cache_activations = True

    for images_path, activation_path in jobs:
        images = sorted(os.listdir(images_path))
        done = len(os.listdir(activation_path)) if os.path.isdir(activation_path) else 0
        if done >= len(images):
            print(f"[skip] {activation_path} ({done} files already cached)", flush=True)
            continue

        os.makedirs(activation_path, exist_ok=True)
        # cache_step names the output files, so reset it per directory.
        hooked_module.activation_path = activation_path
        hooked_module.cache_step = 0

        for name in tqdm(images, desc=os.path.basename(activation_path), leave=False):
            inputs = prepare_inputs(processor, architecture, os.path.join(images_path, name), args.prompt)
            if "lm-layer" in args.sae_model_path:
                boi, eoi = VISION_SPAN_TOKEN_IDS[architecture]
                ids = inputs["input_ids"][0]
                hooked_module.vision_start = (ids == boi).nonzero(as_tuple=True)[0][0].item() + 1
                hooked_module.vision_end = (ids == eoi).nonzero(as_tuple=True)[0][0].item() + 1
            model_with_sae.generate(**inputs, max_new_tokens=1, do_sample=False)

        print(f"[done] {activation_path} ({len(images)} images)", flush=True)


if __name__ == "__main__":
    main()
