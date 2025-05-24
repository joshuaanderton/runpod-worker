import runpod
import requests
import os
import base64

HF_TOKEN = os.getenv("HF_TOKEN")

def handler(event):
    model = event['input'].get('model')
    task = event['input'].get('task')
    image_base64 = event['input'].get('image')

    if not model or not task or not image_base64:
        return {"error": "Missing 'model', 'task', or 'image' in input."}

    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    image_bytes = base64.b64decode(image_base64)
    files = {"file": ("image.jpg", image_bytes)}

    response = requests.post(url, headers=headers, files=files)

    if response.status_code != 200:
        return {"error": response.json()}

    return response.json()
