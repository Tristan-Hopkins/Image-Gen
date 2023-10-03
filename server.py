from flask import Flask, request, jsonify, send_file
import requests
import io
import hashlib
import base64
from faunadb import query as q
from faunadb.client import FaunaClient

# Constants
BACKUP_API_URL = 'https://stablediffusionapi.com/api/v3/text2img'
BACKUP_API_KEY = 'Kep5yIY9sU58fd2ZXciZKc37bnLdRfjpu0fUvur74bVaz73wJVerd6BJjXYe'
NGROK_URL = 'https://imagineit.ngrok.app/ImageGen'
MAX_ATTEMPTS = 3

# Flask app initialization
app = Flask(__name__)

neg_prompt = "(worst quality:1.2), (low quality:1.2), (lowres:1.1), multiple views, comic, sketch, (((bad anatomy))), (((deformed))), (((disfigured))), watermark, multiple_views, mutation hands, mutation fingers, extra fingers, missing fingers, watermark"
enhance_keywords = "((best quality)), ((masterpiece)), (detailed),"

# Global counter for active requests
active_requests = 0

# FaunaDB client setup
FAUNA_SECRET = "fnAFNtTsG9AARFSwW429OKB31VOr71ICCRPWHvbI"
client = FaunaClient(secret=FAUNA_SECRET)

# The FaunaDB functions from the original code

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

def send_task_to_ngrok_server(prompt):
    global active_requests
    if active_requests < 2:
        active_requests += 1
        payload = {"prompt": enhance_keywords + prompt, "steps": 20, "negative_prompt": neg_prompt}
        
        try:
            response = requests.post(NGROK_URL, json=payload, timeout=10)  # Added a timeout
            active_requests -= 1
            
            if response.status_code == 200 and 'images' in response.json():
                return response.json()['images'][0]
            else:
                print(f"Failed to generate image from ngrok server with status code {response.status_code}. Using backup...")
                return backup_image_generation(prompt)
        except requests.RequestException as e:
            print(f"Error occurred while connecting to ngrok server: {e}. Using backup...")
            active_requests -= 1
            return backup_image_generation(prompt)
    else:
        print(f"Queue full. Redirecting to backup...")
        return backup_image_generation(prompt)


def backup_image_generation(prompt):
    attempt = 0
    enhancedPrompt = enhance_keywords + prompt
    while attempt < MAX_ATTEMPTS:
        options = {
            'headers': {'Content-Type': 'application/json'},
            'json': {
                "key": BACKUP_API_KEY,
                "prompt": enhancedPrompt,
                "width": "512",
                "height": "512",
                "samples": "1",
                "num_inference_steps": "20",
                "guidance_scale": 7.5,
                "safety_checker": "yes",
                "embeddings_model": "rev-animated",
                "negative_prompt": neg_prompt,
            }
        }
        response = requests.post(BACKUP_API_URL, **options)
        if response.status_code == 200 and 'output' in response.json() and response.json()['output']:
            return response.json()['output'][0]
        else:
            attempt += 1
    return 'https://i.imgur.com/tdGdu9l.png'

@app.route('/generate-image', methods=['POST'])
def generate_image():
    if not 'prompt' in request.json:
        return jsonify({'error': 'Bad request data'}), 400
    
    image_data_or_url = send_task_to_ngrok_server(request.json['prompt'])

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
