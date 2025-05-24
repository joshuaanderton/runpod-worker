import runpod
import requests
import os

HF_TOKEN = os.getenv("HF_TOKEN")

def handler(event):
    model = event['input'].get('model')  # e.g. "gpt2" or "Salesforce/blip-image-captioning-base"
    task = event['input'].get('task')    # e.g. "text-generation" or "image-to-text"
    data = event['input'].get('data')    # e.g. {"inputs": "Hello world"}

    if not model or not task or not data:
        return { "error": "Missing 'model', 'task', or 'data' in input." }

    # Call Hugging Face Inference API
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = { "Authorization": f"Bearer {HF_TOKEN}" }

    response = requests.post(url, headers=headers, json=data)

    return response.json()