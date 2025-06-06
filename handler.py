import runpod
import os
import torch
import uuid
import boto3
import os
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    FluxPipeline,
    WanPipeline,
    AutoencoderKLWan
)
from diffusers.utils import (
    export_to_video,
    load_image
)

# Global model cache
loaded_models = {}

def get_model(model_id, task):

    if model_id in loaded_models:
        return loaded_models[model_id]

    if task.endswith("-image"):
        if model_id.startswith("black-forest-labs/FLUX"):
            model = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
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

    loaded_models[model_id] = model
    return model

def upload_to_cloud(file_path, file_type="image/png"):
    bucket = os.getenv("AWS_BUCKET")
    key = f"outputs/{file_path}"

    if file_path.endswith('.mp4'):
        file_type = 'video/mp4'

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
    client.upload_file(file_path, bucket, key, ExtraArgs={'ACL': 'public-read', 'ContentType': file_type})

    return f"{os.getenv('AWS_URL')}/{bucket}/{key}"

def handler(event):
    task = event['input'].get('task')                           # text-to-image, image-to-image, image-to-video
    model_id = event['input'].get('model')                      # e.g., runwayml/stable-diffusion-v1-5

    prompt = event['input'].get('prompt')                       # Prompt text
    image_url = event['input'].get('image_url')                 # Image URL input (optional)
    seed = event['input'].get('seed', -1)                       # Random seed (optional)
    height = event['input'].get('height', 480)                  # Image height (optional)
    width = event['input'].get('width', 832)                    # Image height (optional)
    num_frames = event['input'].get('num_frames', 33)           # Number of frames for video generation (optional)
    guidance_scale = event['input'].get('guidance_scale', 5.0)  # Guidance scale for generation (optional)
    negative_prompt = event['input'].get('negative_prompt', "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")

    output_name = uuid.uuid4()
    output_path = f"{output_name}.png"

    if not task or not model_id or not prompt:
        return { "error": "Missing required fields: 'task', 'model', or 'prompt'" }

    model = get_model(model_id, task)
    model.to(torch.device("cuda"))

    if task == "text-to-image":
        output_path = f"{output_name}.png"
        output = model(prompt=prompt).images[0]
        output.save(output_path)

    elif task == "image-to-image":
        if not image_url:
            return { "error": "Missing 'image_url' input for image-to-video task." }

        input_image = load_image(image_url)
        output = model(prompt=prompt, image=input_image, strength=0.75, guidance_scale=7.5).images[0]
        output.save(output_path)

    elif task == "text-to-video":

        frames = model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
        ).frames[0]

        output_path = f"{output_name}.mp4"
        export_to_video(frames, output_path, fps=16)

    elif task == "image-to-video":

        if not image_url:
            return { "error": "Missing 'image_url' input for image-to-video task." }

        input_image = load_image(image_url)
        max_area = 480 * 832
        aspect_ratio = input_image.height / input_image.width
        mod_value = model.vae_scale_factor_spatial * model.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = input_image.resize((width, height))

        frames = model(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
        ).frames[0]

        output_path = f"{output_name}.mp4"
        export_to_video(frames, output_path, fps=16)

    url = upload_to_cloud(output_path)
    return { "url": url }

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
