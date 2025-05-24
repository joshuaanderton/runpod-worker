import runpod
import os
import base64
import io
import torch
from PIL import Image

from diffusers import (
    StableDiffusionPipeline,
    AutoPipelineForImage2Image,
    AutoPipelineForText2Video
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

def handler(event):
    task = event['input'].get('task')              # text-to-image, image-to-image, image-to-video
    model_id = event['input'].get('model')         # e.g., runwayml/stable-diffusion-v1-5
    prompt = event['input'].get('prompt')          # Prompt text
    image_b64 = event['input'].get('image')        # base64 image input (optional)
    seed = event['input'].get('seed', None)

    if not task or not model_id or not prompt:
        return {"error": "Missing required fields: 'task', 'model', or 'prompt'"}

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if task == "text-to-image":
        if model_id not in loaded_models:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            loaded_models[model_id] = pipe
        pipe = loaded_models[model_id]
        result = pipe(prompt).images[0]
        return { "image_base64": encode_image(result) }

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
        return { "image_base64": encode_image(result) }

    elif task == "text-to-video" or task == "image-to-video":
        if model_id not in loaded_models:
            pipe = AutoPipelineForText2Video.from_pretrained(model_id, torch_dtype=torch_dtype)
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            loaded_models[model_id] = pipe
        pipe = loaded_models[model_id]

        output = pipe(prompt=prompt, num_frames=16)
        video_path = "output.mp4"
        output.save(video_path)
        return { "video_base64": encode_video(video_path) }
        
    elif task == "image-to-video":
        if not image_b64:
            return {"error": "Missing 'image' input for image-to-video task."}
        init_image = decode_image(image_b64)

        # Save image locally for models that expect file input
        input_image_path = "input.png"
        init_image.save(input_image_path)

        if model_id not in loaded_models:
            pipe = AutoPipelineForText2Video.from_pretrained(model_id, torch_dtype=torch_dtype)
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            loaded_models[model_id] = pipe

        pipe = loaded_models[model_id]

        output = pipe(image=input_image_path, num_frames=16)
        video_path = "output.mp4"
        output.save(video_path)

        return { "video_base64": encode_video(video_path) }

    else:
        return { "error": f"Unsupported task: {task}" }
