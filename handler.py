import runpod
import os
import base64
import io
import torch
from PIL import Image
import boto3
import uuid
from pathlib import Path
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    FluxPipeline,
    WanPipeline,
    AutoencoderKLWan
)
from diffusers.utils import (
    export_to_video, load_image
)

# Global model cache
loaded_models = {}

def get_valid_kwargs(pipe, input_data):
    """Filter input_data to only the kwargs accepted by the pipeline."""
    valid_keys = pipe.__call__.__code__.co_varnames
    return {k: v for k, v in input_data.items() if k in valid_keys}

def upload_to_s3(file_path, file_type):
    # endpoint = os.environ["AWS_ENDPOINT"]
    bucket = os.environ["AWS_BUCKET"]
    url = os.environ["AWS_URL"]
    key = f"outputs/{uuid.uuid4()}{Path(file_path).suffix}"

    s3 = boto3.client("s3")
    s3.upload_file(file_path, bucket, key, ExtraArgs={
        "ContentType": f"{file_type}",
        "ACL": "public-read"
    })

    return f"{url}/{bucket}/{key}"

def handler(event):
    #kwargs = get_valid_kwargs(pipe, event["input"])
    task = event['input'].get('task')                           # text-to-image, image-to-image, image-to-video
    model_id = event['input'].get('model')                      # e.g., runwayml/stable-diffusion-v1-5
    prompt = event['input'].get('prompt')                       # Prompt text
    input_image = event['input'].get('image')                   # base64 image input (optional)
    image_url = event['input'].get('image_url')                 # Image URL input (optional)
    seed = event['input'].get('seed', -1)                       # Random seed (optional)
    height = event['input'].get('height', 480)                  # Image height (optional)
    width = event['input'].get('width', 832)                    # Image height (optional)
    num_frames = event['input'].get('num_frames', 33)           # Number of frames for video generation (optional)
    guidance_scale = event['input'].get('guidance_scale', 5.0)  # Guidance scale for generation (optional)
    negative_prompt = event['input'].get('negative_prompt', "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")

    if not task or not model_id or not prompt:
        return {"error": "Missing required fields: 'task', 'model', or 'prompt'"}

    # Setup device
    device = torch.device("cuda")
    torch_dtype = torch.float16

    if task == "text-to-image":
        if model_id not in loaded_models:
            if model_id.startswith("black-forest-labs/FLUX"):
                model = FluxPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
            else:
                model = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch_dtype)
            model.to(device)
            loaded_models[model_id] = model

        model = loaded_models[model_id]
        output = model(prompt).images[0]

        out_path = "output.png"
        output.save(out_path)
        s3_url = upload_to_s3(out_path, "image/png")

        return { "url": s3_url }

    elif task == "image-to-image":
        if not image_url:
            return {"error": "Missing 'image_url' input for image-to-video task."}

        input_image = load_image(image_url)

        if model_id not in loaded_models:
            model = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=torch_dtype)
            model.to(device)
            loaded_models[model_id] = model

        model = loaded_models[model_id]
        output = model(prompt=prompt, image=input_image, strength=0.75, guidance_scale=7.5).images[0]

        out_path = "output.png"
        output.save(out_path)
        s3_url = upload_to_s3(out_path, "image/png")

        return { "url": s3_url }

    elif task == "text-to-video":
        if model_id not in loaded_models:
            vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
            if model_id.startswith("Wan-AI/"):
                model = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
            else:
                model = AutoPipelineForText2Image.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
            model.to(device)
            loaded_models[model_id] = model

        model = loaded_models[model_id]

        out_path = "output.mp4"
        frames = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
        ).frames[0]
        export_to_video(frames, out_path, fps=16)
        s3_url = upload_to_s3(out_path, "video/mp4")

        return { "url": s3_url }

    elif task == "image-to-video":

        if not image_url:
            return {"error": "Missing 'image_url' input for image-to-video task."}

        input_image = load_image(image_url)

        if model_id not in loaded_models:
            vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
            if model_id.startswith("Wan-AI/"):
                model = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
            else:
                model = AutoPipelineForText2Image.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
            model.to(device)
            loaded_models[model_id] = model

        model = loaded_models[model_id]

        max_area = 480 * 832
        aspect_ratio = image.height / image.width
        mod_value = model.vae_scale_factor_spatial * model.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))

        frames = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
        ).frames[0]
        output_path = "output.mp4"
        export_to_video(output, output_path, fps=16)
        s3_url = upload_to_s3(output_path, "video/mp4")

        return { "url": s3_url }

    else:
        return { "error": f"Unsupported task: {task}" }

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
