import os
import argparse

from dotenv import load_dotenv
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from sae import PeftSaeModel
from sae.models.topk_sae import Linear as SaeLinear
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


def load_model_and_processor(model_path):
    """Load the base model and processor."""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="cuda"
    )
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=min_pixels, max_pixels=max_pixels
    )
    return model, processor


def prepare_inputs(processor, image_path, prompt):
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
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
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
    parser = argparse.ArgumentParser(description="Cache SAE activations for Qwen2.5-VL on a directory of images")
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
    model, processor = load_model_and_processor(args.model_path)
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
        inputs = prepare_inputs(processor, image_path, args.prompt)

        # <vision_start> and <vision_end> tokens are 151652 and 151653 respectively for Qwen2.5-VL.
        if "lm-layer" in args.sae_model_path:
            hooked_module.vision_start = (inputs["input_ids"][0] == 151652).nonzero(as_tuple=True)[0][0].item() + 1
            hooked_module.vision_end = (inputs["input_ids"][0] == 151653).nonzero(as_tuple=True)[0][0].item() + 1

        generate_output(model_with_sae, processor, inputs, "Model with SAE Output", max_new_tokens=1)

if __name__ == "__main__":
    main()
