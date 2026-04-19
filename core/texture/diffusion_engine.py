import os
from pathlib import Path


def try_generate_with_diffusion(
    image_paths: list[str],
    output_dir: Path,
    n_candidates: int,
) -> list[str] | None:
    """Best-effort diffusion generation.

    Returns None when diffusion is disabled or unavailable so callers can fallback.
    """
    if os.getenv("TEXTURE_ENGINE", "baseline") != "diffusion":
        return None

    model_id = os.getenv("DIFFUSION_MODEL_ID", "runwayml/stable-diffusion-v1-5")
    lora_path = os.getenv("LORA_WEIGHTS_PATH", "").strip()
    lora_scale = float(os.getenv("LORA_SCALE", "0.8"))
    prompt = os.getenv("DIFFUSION_PROMPT", "fabric seamless textile pattern")
    negative_prompt = os.getenv(
        "DIFFUSION_NEGATIVE_PROMPT",
        "blurry, distorted, watermark, text",
    )

    try:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline
        from PIL import Image

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

        if lora_path and Path(lora_path).exists():
            try:
                pipe.load_lora_weights(lora_path)
                pipe.fuse_lora(lora_scale=lora_scale)
            except Exception:
                # If LoRA cannot be loaded, continue with base diffusion model.
                pass

        if torch.cuda.is_available():
            pipe = pipe.to("cuda")

        if not image_paths:
            return []

        with Image.open(image_paths[0]) as first:
            init_image = first.convert("RGB").resize((768, 768))

        outputs: list[str] = []
        for idx in range(1, n_candidates + 1):
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=0.65,
                guidance_scale=7.5,
            )
            image = result.images[0]
            file_path = output_dir / f"candidate_{idx:02d}.png"
            image.save(file_path, format="PNG")
            outputs.append(str(file_path))

        return outputs
    except Exception:
        return None
