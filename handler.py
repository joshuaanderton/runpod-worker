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
    WanPipeline
)

# Global model cache
loaded_models = {}

def decode_image(b64_string):
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def encode_image(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def encode_video(video_path):
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_valid_kwargs(pipe, input_data):
    """Filter input_data to only the kwargs accepted by the pipeline."""
    valid_keys = pipe.__call__.__code__.co_varnames
    return {k: v for k, v in input_data.items() if k in valid_keys}

def upload_to_s3(file_path, file_type):
    s3 = boto3.client("s3")
    bucket = os.environ["AWS_S3_BUCKET"]
    key = f"outputs/{uuid.uuid4()}{Path(file_path).suffix}"

    s3.upload_file(file_path, bucket, key, ExtraArgs={
        "ContentType": f"{file_type}",
        "ACL": "public-read"
    })

    region = os.getenv("AWS_REGION", "us-east-1")
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"

def handler(event):
    task = event['input'].get('task')       # text-to-image, image-to-image, image-to-video
    model_id = event['input'].get('model')  # e.g., runwayml/stable-diffusion-v1-5
    prompt = event['input'].get('prompt')   # Prompt text
    image_b64 = event['input'].get('image') # base64 image input (optional)
    seed = event['input'].get('seed', -1)   # Random seed (optional)
    #kwargs = get_valid_kwargs(pipe, event["input"])

    if not task or not model_id or not prompt:
        return {"error": "Missing required fields: 'task', 'model', or 'prompt'"}

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if task == "text-to-image":
        if model_id not in loaded_models:
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch_dtype)
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            loaded_models[model_id] = pipe
        pipe = loaded_models[model_id]
        result = pipe(prompt).images[0]
        out_path = "output.png"
        result.save(out_path)
        s3_url = upload_to_s3(out_path, "image/png")
        return { "s3_url": s3_url }

    elif task == "image-to-image":
        if not image_b64:
            return {"error": "Missing 'image' input for image-to-image task."}
        init_image = decode_image(image_b64)

        if model_id not in loaded_models:
            pipe = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=torch_dtype)
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            loaded_models[model_id] = pipe
        pipe = loaded_models[model_id]
        result = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
        out_path = "output.png"
        result.save(out_path)
        s3_url = upload_to_s3(out_path, "image/png")
        return { "s3_url": s3_url }

    elif task == "text-to-video" or task == "image-to-video":
        if model_id not in loaded_models:
            if model_id.startswith("Wan-AI/"):
                pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
            else:
                pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch_dtype)
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            loaded_models[model_id] = pipe
        pipe = loaded_models[model_id]

        output = pipe(prompt=prompt, num_frames=16)
        video_path = "output.mp4"
        output.save(video_path)
        s3_url = upload_to_s3(video_path, "video/mp4")
        return { "s3_url": s3_url }

    elif task == "image-to-video":
        if not image_b64:
            return {"error": "Missing 'image' input for image-to-video task."}
        init_image = decode_image(image_b64)

        # Save image locally for models that expect file input
        input_image_path = "input.png"
        init_image.save(input_image_path)

        if model_id not in loaded_models:
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch_dtype)
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            loaded_models[model_id] = pipe

        pipe = loaded_models[model_id]

        output = pipe(image=input_image_path, num_frames=16)
        video_path = "output.mp4"
        output.save(video_path)
        s3_url = upload_to_s3(video_path, "video/mp4")
        return { "s3_url": s3_url }

    else:
        return { "error": f"Unsupported task: {task}" }

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
