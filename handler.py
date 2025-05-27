import runpod
import os
import base64
import io
import torch
from PIL import Image
import boto3
import uuid
from pathlib import Path
# from huggingface_hub import login

from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    WanPipeline
)

# Global model cache
loaded_models = {}

# Login to Hugging Face Hub just in case
# login(token=os.environ["HUGGING_FACE_ACCESS_TOKEN"])

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
    task = event['input'].get('task')       # text-to-image, image-to-image, image-to-video
    model_id = event['input'].get('model')  # e.g., runwayml/stable-diffusion-v1-5
    prompt = event['input'].get('prompt')   # Prompt text
    image_b64 = event['input'].get('image') # base64 image input (optional)
    seed = event['input'].get('seed', -1)   # Random seed (optional)
    #kwargs = get_valid_kwargs(pipe, event["input"])
    input_image_path = "input.png"

    if not task or not model_id or not prompt:
        return {"error": "Missing required fields: 'task', 'model', or 'prompt'"}

    # Setup device
    device = torch.device("cuda")
    torch_dtype = torch.float16

    if task == "text-to-image":
        if model_id not in loaded_models:
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
        if not image_b64:
            return {"error": "Missing 'image' input for image-to-image task."}
        input_image = decode_image(image_b64)
        input_image.save(input_image_path)

        if model_id not in loaded_models:
            model = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=torch_dtype)
            model.to(device)
            loaded_models[model_id] = model

        model = loaded_models[model_id]
        output = model(prompt=prompt, image=input_image_path, strength=0.75, guidance_scale=7.5).images[0]

        out_path = "output.png"
        output.save(out_path)
        s3_url = upload_to_s3(out_path, "image/png")

        return { "url": s3_url }

    elif task == "text-to-video" or task == "image-to-video":
        if model_id not in loaded_models:
            if model_id.startswith("Wan-AI/"):
                model = WanPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
            else:
                model = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch_dtype)
            model.to(device)
            loaded_models[model_id] = model

        model = loaded_models[model_id]
        output = model(prompt=prompt, num_frames=16)

        out_path = "output.mp4"
        output.save(out_path)
        s3_url = upload_to_s3(out_path, "video/mp4")

        return { "url": s3_url }

    elif task == "image-to-video":
        if not image_b64:
            return {"error": "Missing 'image' input for image-to-video task."}
        input_image = decode_image(image_b64)
        input_image.save(input_image_path)

        if model_id not in loaded_models:
            model = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch_dtype)
            model.to(device)
            loaded_models[model_id] = model

        model = loaded_models[model_id]

        output = model(image=input_image_path, num_frames=16)
        output_path = "output.mp4"
        output.save(output_path)
        s3_url = upload_to_s3(output_path, "video/mp4")

        return { "url": s3_url }

    else:
        return { "error": f"Unsupported task: {task}" }

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
