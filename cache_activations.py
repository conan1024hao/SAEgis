import os
import argparse

import transformers
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoConfig, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from sae import PeftSaeModel
from sae.models.topk_sae import Linear as SaeLinear
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# (begin-of-image, end-of-image) token ids per architecture. Only needed to slice the
# vision span out of language-model-layer activations ("lm-layer" SAEs); the vision-side
# SAEs already see nothing but vision tokens.
VISION_SPAN_TOKEN_IDS = {
    # <|vision_start|>, <|vision_end|>
    "Qwen2VLForConditionalGeneration": (151652, 151653),
    "Qwen2_5_VLForConditionalGeneration": (151652, 151653),
    # <|image>, <image|>
    "Gemma4ForConditionalGeneration": (255999, 258882),
}


def get_architecture(model_path):
    """Read the model class name out of the checkpoint's config."""
    config = AutoConfig.from_pretrained(model_path)
    architectures = getattr(config, "architectures", None)
    if not architectures:
        raise ValueError(f"Model {model_path} has no architecture defined in its config.")
    return architectures[0]


def load_model_and_processor(model_path, architecture):
    """Load the base model and processor."""
    model_class = getattr(transformers, architecture, None)
    if model_class is None:
        raise ValueError(
            f"Architecture {architecture} is not available in transformers "
            f"{transformers.__version__}. Gemma 4 needs transformers>=5.5."
        )
    model = model_class.from_pretrained(model_path, dtype="auto", device_map="cuda")

    processor_kwargs = {}
    if architecture.startswith("Qwen"):
        # Qwen2.5-VL's vision token count is driven by the pixel budget. Other models
        # (Gemma 4) use a fixed soft-token count taken from their processor config.
        processor_kwargs = {"min_pixels": 256 * 28 * 28, "max_pixels": 1280 * 28 * 28}
    processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
    return model, processor


def prepare_inputs(processor, architecture, image_path, prompt):
    """Prepare model inputs from image and prompt."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if architecture.startswith("Qwen"):
        image_inputs, video_inputs = process_vision_info(messages)
    else:
        image_inputs, video_inputs = [Image.open(image_path).convert("RGB")], None
    # `text` already carries every special token written by the chat template; Gemma's
    # tokenizer would otherwise prepend a second <bos>. This matches the SAE training path.
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return inputs.to("cuda")


def generate_output(model, processor, inputs, label="Model Output", max_new_tokens=1):
    """Generate and print output from the model."""
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("=" * 44)
    print(f"{label}:")
    print(output_text[0])
    return output_text[0]


def main():
    parser = argparse.ArgumentParser(description="Cache SAE activations for a VLM on a directory of images")
    parser.add_argument(
        "--model_path",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Path to the base model"
    )
    parser.add_argument(
        "--sae_model_path",
        default="mtri-admin/qwen2.5-vl-3b-sae-projection-mlp2-finevisionmax-500k",
        help="Path to the SAE model"
    )
    parser.add_argument(
        "--images_path",
        default="./images/original",
        help="Path to the images"
    )
    parser.add_argument(
        "--prompt",
        default="Describe this image.",
        help="Prompt to use for generation"
    )
    parser.add_argument(
        "--activation_path",
        default="./sae_activations.pt",
        help="Path to save SAE activations"
    )
    parser.add_argument("--dev_mode", action="store_true", help="Whether to use dev mode with fewer images")
    args = parser.parse_args()

    # Load model and processor
    architecture = get_architecture(args.model_path)
    model, processor = load_model_and_processor(args.model_path, architecture)
    model_with_sae = PeftSaeModel.from_pretrained(
        model,
        args.sae_model_path,
        adapter_name="default",
        low_cpu_mem_usage=True,
        token=HUGGINGFACE_TOKEN,
    )

    # Enable SAE activation caching
    for module in model.model.modules():
        if isinstance(module, SaeLinear):
            hooked_module = module
            break
    hooked_module.cache_activations = True
    hooked_module.activation_path = args.activation_path

    # Generate outputs and save SAE activations for each image
    image_list = sorted(os.listdir(args.images_path))
    if args.dev_mode:
        image_list = image_list[:100]
    for image_name in tqdm(image_list):
        image_path = os.path.join(args.images_path, image_name)
        inputs = prepare_inputs(processor, architecture, image_path, args.prompt)

        # For language-model-layer SAEs, restrict the cached activations to the vision span.
        if "lm-layer" in args.sae_model_path:
            if architecture not in VISION_SPAN_TOKEN_IDS:
                raise ValueError(
                    f"No begin/end-of-image token ids registered for {architecture}; "
                    "add them to VISION_SPAN_TOKEN_IDS to cache lm-layer activations."
                )
            boi_token_id, eoi_token_id = VISION_SPAN_TOKEN_IDS[architecture]
            hooked_module.vision_start = (inputs["input_ids"][0] == boi_token_id).nonzero(as_tuple=True)[0][0].item() + 1
            hooked_module.vision_end = (inputs["input_ids"][0] == eoi_token_id).nonzero(as_tuple=True)[0][0].item() + 1

        generate_output(model_with_sae, processor, inputs, "Model with SAE Output", max_new_tokens=1)

if __name__ == "__main__":
    main()
