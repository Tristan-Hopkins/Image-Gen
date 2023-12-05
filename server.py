import os
import requests
import io
import hashlib
import base64
from flask import Flask, request, jsonify, send_file
from faunadb import query as q
from faunadb.client import FaunaClient
from novita_client import NovitaClient, Txt2ImgRequest, Samplers, ModelType, save_image
active_requests = 0
# Constants
NGROK_URL = 'https://imagineit.ngrok.app/ImageGen'
MAX_ATTEMPTS = 3
neg_prompt = "nsfw, nudity, naked, breasts, (worst quality:1.2), (low quality:1.2), (lowres:1.1), multiple views, comic, sketch, (((bad anatomy))), (((deformed))), (((disfigured))), watermark, multiple_views, mutation hands, mutation fingers, extra fingers, missing fingers, watermark"
enhance_keywords = "((best quality)), ((masterpiece)), (detailed),"

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

def send_task_to_ngrok_server(prompt, width="512", height="512"):
    global active_requests
    if active_requests < 20:
        active_requests += 1
        payload = {
            "prompt": enhance_keywords + prompt,
            "steps": 20,
            "negative_prompt": neg_prompt,
            "width": width,   # Add width
            "height": height  # Add height
        }
        
        try:
            response = requests.post(NGROK_URL, json=payload, timeout=10)  # Added a timeout
            active_requests -= 1
            
            if response.status_code == 200 and 'images' in response.json():
                return response.json()['images'][0]
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
    image_ref = save_image_to_fauna(base64.b64encode(output_image_bytes).decode(), hash_id)
    return f"https://image-labs.onrender.com/images/{hash_id}"

@app.route('/generate-image', methods=['POST'])
def generate_image():
    if not 'prompt' in request.json:
        return jsonify({'error': 'Bad request data'}), 400
    
    # Extract width and height from request JSON
    width = request.json.get('width', "512")
    height = request.json.get('height', "512")

    image_data_or_url = send_task_to_ngrok_server(request.json['prompt'], width, height)

    # Check if the returned data is a URL from the backup generator
    if image_data_or_url.startswith('http'):
        return jsonify({"image_url": image_data_or_url})

    # If it's not a URL, then it's image data, so we store it in FaunaDB
    filename = hashlib.md5(image_data_or_url.encode()).hexdigest()
    image_ref = save_image_to_fauna(image_data_or_url, filename)
    return jsonify({"image_url": f"https://image-labs.onrender.com/images/{filename}"})

@app.route('/images/<hash_id>', methods=['GET'])
def retrieve_image(hash_id):
    image_data = get_image_from_fauna(hash_id)
    if not image_data:
        return jsonify({"error": "Image not found"}), 404
    image_binary = io.BytesIO(base64.b64decode(image_data))
    return send_file(image_binary, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
