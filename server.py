import os
import requests
import io
import hashlib
import base64
from flask import Flask, request, jsonify, send_file
from faunadb import query as q
from faunadb.client import FaunaClient
from novita_client import NovitaClient, Txt2ImgRequest, Samplers, ModelType, save_image
from PIL import Image

active_requests = 0

# Constants
NGROK_URL = 'https://imagineit.ngrok.app/ImageGen'
MAX_ATTEMPTS = 3
neg_prompt = "bad quality, bad anatomy, worst quality, low quality, low resolutions, extra fingers, blur, blurry, ugly, wrongs proportions, watermark, image artifacts, lowres, ugly, jpeg artifacts, deformed, noisy image, ugly faces"
enhance_keywords = "graphic novel, vibrant, inked, High Quailty "
IMAGE_DIR = "/var/data/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Flask app initialization
app = Flask(__name__)

# FaunaDB client setup
FAUNA_SECRET = "fnAFNtTsG9AARFSwW429OKB31VOr71ICCRPWHvbI"
client = FaunaClient(secret=FAUNA_SECRET)

# Novita API Client
NOVITA_API_KEY = os.getenv("NOVITA_API_KEY", "8f347388-1d5b-4153-8c3f-704f17bdb45e")
novita_client = NovitaClient(NOVITA_API_KEY)

# FaunaDB Functions

def get_server_ref_by_url(server_url):
    server_data = client.query(q.get(q.match(q.index("servers_by_url"), server_url)))
    return server_data["ref"]

def mark_server_as_in_use(server_url, in_use_status):
    server_ref = get_server_ref_by_url(server_url)
    client.query(q.update(server_ref, {"data": {"in_use": in_use_status}}))

def save_image_to_fauna(image_data, hash_id):
    image_record = {
        "hash_id": hash_id,
        "data": image_data
    }
    result = client.query(q.create(q.collection("images"), {"data": image_record}))
    return result["ref"]

def get_image_from_fauna(hash_id):
    try:
        image_record = client.query(q.get(q.match(q.index("images_by_hash_id"), hash_id)))
        return image_record["data"]["data"]
    except:
        return None

def get_available_servers():
    try:
        refs = client.query(q.paginate(q.match(q.index("all_servers"), False)))["data"]
        servers_data = client.query(q.map_(lambda ref: q.get(ref), refs))
        servers = [{"ref": server["ref"], "data": server["data"]} for server in servers_data]
        return servers
    except Exception as e:
        return []

def remove_server(server_ref):
    client.query(q.delete(server_ref))

# Helper functions for image handling

def save_image_to_disk(image_data, file_name, image_format):
    file_path = os.path.join(IMAGE_DIR, f"{file_name}.{image_format}")

    # Directly save the bytes data to disk
    with open(file_path, 'wb') as image_file:
        image_file.write(base64.b64decode(image_data))

    return file_path


def get_image_from_disk(hash_id):
    for ext in [".png", ".webp"]:
        file_path = os.path.join(IMAGE_DIR, hash_id + ext)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as image_file:
                return image_file.read(), ext.lstrip('.')
    
    print(f"File not found for hash: {hash_id}")  # Simple logging
    return None, None
# Image Generation and Backup Functions

def send_task_to_ngrok_server(prompt, width="768", height="768"):
    global active_requests
    if active_requests < 20:
        active_requests += 1
        payload = {
            "prompt": enhance_keywords + prompt,
            "steps": 20,
            "negative_prompt": neg_prompt,
            "width": width,
            "height": height
        }
        
        try:
            response = requests.post(NGROK_URL, json=payload, timeout=10)
            active_requests -= 1
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type')
                if content_type == 'image/webp':
                    image_format = 'webp'
                    image_data = base64.b64encode(response.content).decode()
                elif content_type == 'image/png':
                    image_format = 'png'
                    image_data = base64.b64encode(response.content).decode()
                else:
                    raise ValueError(f"Unexpected content type: {content_type}")
                
                return image_data, image_format
            else:
                print(f"Failed to generate image from ngrok server with status code {response.status_code}. Using backup...")
                return backup_image_generation(prompt, width, height)
        except requests.RequestException as e:
            print(f"Error occurred while connecting to ngrok server: {e}. Using backup...")
            active_requests -= 1
            return backup_image_generation(prompt, width, height)
    else:
        print(f"Queue full. Redirecting to backup...")
        return backup_image_generation(prompt, width, height)

def backup_image_generation(prompt, width="512", height="512"):
    req = Txt2ImgRequest(
        model_name='revAnimated_v122.safetensors',
        prompt=enhance_keywords + prompt,
        negative_prompt=neg_prompt,
        width=int(width),
        height=int(height),
        sampler_name="DPM++ 2M Karras",
        cfg_scale=8.5,
        steps=20,
        batch_size=1,
        n_iter=1,
        seed=-1,
    )
    output_image_bytes = novita_client.sync_txt2img(req).data.imgs_bytes[0]
    hash_id = hashlib.md5(output_image_bytes).hexdigest()

    # Convert the image bytes to PIL Image
    image = Image.open(io.BytesIO(output_image_bytes))

    # Compress the image
    compressed_image_bytes = io.BytesIO()
    image.save(compressed_image_bytes, format='WebP', optimize=True, quality=80)
    compressed_image_bytes.seek(0)

    # Save the compressed image to disk
    webp_image_path = os.path.join(IMAGE_DIR, f"{hash_id}_v2.webp")
    with open(webp_image_path, 'wb') as image_file:
        image_file.write(compressed_image_bytes.getvalue())

    # Check the size of the compressed WebP image
    webp_image_size = os.path.getsize(webp_image_path)
    print("Compressed WebP image size (bytes):", webp_image_size)

    # Return the base64-encoded compressed image data and the image format
    return base64.b64encode(compressed_image_bytes.getvalue()).decode(), "webp"
# Updated generate_image function
@app.route('/generate-image', methods=['POST'])
def generate_image():
    if 'prompt' not in request.json:
        return jsonify({'error': 'Bad request data'}), 400

    prompt = request.json['prompt']
    width = request.json.get('width', "512")
    height = request.json.get('height', "512")

    image_data, image_format = send_task_to_ngrok_server(prompt, width, height)
    
    if image_data.startswith("http"):
        # Download the image if it's a URL
        response = requests.get(image_data)
        image_bytes = response.content
        hash_id = image_data.split('/')[-1].replace(f".{image_format}", "")
    else:
        # If it's base64 encoded image data
        image_bytes = base64.b64decode(image_data)
        hash_id = hashlib.md5(image_bytes).hexdigest()
        
    save_image_to_disk(image_data, hash_id, image_format)

    return jsonify({"image_url": f"https://image-labs.onrender.com/imagesV2/{hash_id}.{image_format}"})

def save_image_to_disk(image_data, hash_id, image_format):
    file_path = os.path.join(IMAGE_DIR, f"{hash_id}.{image_format}")

    with open(file_path, 'wb') as image_file:
        image_file.write(base64.b64decode(image_data))

    return file_path


@app.route('/images/<hash_id>', methods=['GET'])
def retrieve_image(hash_id):
    image_data = get_image_from_fauna(hash_id)
    if not image_data:
        return jsonify({"error": "Image not found"}), 404
    image_binary = io.BytesIO(base64.b64decode(image_data))
    return send_file(image_binary, mimetype='image/png')

@app.route('/imagesV2/<hash_id>', methods=['GET'])
def retrieve_image_v2(hash_id):
    # Remove the file extension from the hash_id
    hash_id = os.path.splitext(hash_id)[0]
    
    image_data, image_format = get_image_from_disk(hash_id)
    if image_data is None:
        return jsonify({"error": "Image not found"}), 404
    return send_file(io.BytesIO(image_data), mimetype=f'image/{image_format}')
# Main Application Execution

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
