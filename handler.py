import runpod
import os
import torch
import uuid
import boto3
import os
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    DiffusionPipeline,
    FluxPipeline,
    WanPipeline,
    AutoencoderKLWan
)
from diffusers.utils import (
    export_to_video,
    load_image
)
import huggingface_hub

# Global model cache
loaded_models = {}

def handler(event):
    output_name = f"outputs/{uuid.uuid4()}"
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        return {
            "url": None,
            "error": "HF_TOKEN is not set in environment variables."
        }

    huggingface_hub.login(token=hf_token, add_to_git_credential=False)

    input = event['input']
    task = input.get('task')      # text-to-image, image-to-image, image-to-video
    model_id = input.get('model') # e.g., runwayml/stable-diffusion-v1-5

    if not task or not model_id:
        return {
            "url": None,
            "error": "Missing required fields: 'task' and 'model'"
        }

    model = get_model(model_id, task)

    prompt = input.get('prompt')       # Prompt text (sometimes required)
    image_url = input.get("image_url") # Image URL input (sometimes required)

    if task.startswith("text-") and not prompt:
        return {
            "url": None,
            "error": "Missing required field 'prompt'"
        }

    if task.startswith("image-") and not image_url:
        return {
            "url": None,
            "error": "Missing required field 'image_url'"
        }

    seed = input.get("seed", -1)                       # Random seed (optional)
    height = input.get("height", 480)                  # Image height (optional)
    width = input.get("width", 832)                    # Image width (optional)
    fps = input.get("fps", 16)                         # Frames per second for video (optional)
    num_frames = input.get("num_frames", 33)           # Number of frames for video generation (optional)
    guidance_scale = input.get("guidance_scale", 5.0)  # Guidance scale for generation (optional)
    negative_prompt = input.get(                       # Negative prompt for image generation (optional)
        "negative_prompt",
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    )

    if task.startswith("image-"):
        input_image = load_image(image_url)

        max_area = 480 * 832 # Use Wan2.1 480p default "max_area"
        aspect_ratio = input_image.height / input_image.width
        mod_value = model.vae_scale_factor_spatial * model.transformer.config.patch_size[1]

        height = round(input_image.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(input_image.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        input_image = input_image.resize((width, height))
    else:
        input_image = None

    if task.endswith("-image"):
        output_path = f"{output_name}.png"

        output = model(
            prompt=prompt,
            image=input_image,
            # strength=0.75,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
        ).images[0]

        output.save(output_path)

    elif task.endswith("-video"):
        output_path = f"{output_name}.mp4"

        frames = model(
            seed=seed,
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
        ).frames[0]

        export_to_video(frames, output_path, fps)

    return {
        "url": upload_to_cloud(output_path),
        "error": None
    }

def get_model(model_id, task):

    if model_id in loaded_models:
        return loaded_models[model_id]

    if task.endswith("-image"):
        if model_id.startswith("black-forest-labs/FLUX"):
            model = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        elif task == "text-to-image":
            model = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        elif task == "image-to-image":
            model = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    elif task.endswith("-video"):
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        if model_id.startswith("Wan-AI/"):
            model = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        else:
            model = AutoPipelineForText2Image.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)

    model.to(torch.device("cuda"))

    loaded_models[model_id] = model

    return model

def upload_to_cloud(file_path):
    bucket = os.getenv("AWS_BUCKET")
    key = f"outputs/{file_path}"

    if file_path.endswith(".mp4"):
        mime_type = "video/mp4"
    elif file_path.endswith(".png"):
        mime_type = "image/png"

    # Upload to AWS S3 or DO Spaces
    session = boto3.session.Session()
    client = session.client(
        's3',
        region_name=os.getenv("AWS_REGION"),
        endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    # Upload the file
    client.upload_file(
        file_path,
        bucket,
        key,
        ExtraArgs={
            "ACL": "public-read",
            "ContentType": mime_type
        }
    )

    return f"{os.getenv('AWS_URL')}/{bucket}/{key}"

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
