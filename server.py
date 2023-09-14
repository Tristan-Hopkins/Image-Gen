from flask import Flask, request, jsonify, send_file
import requests
import io
import os
import hashlib
from PIL import Image
import base64
import time
from faunadb import query as q
from faunadb.client import FaunaClient

# Constants
BACKUP_API_URL = 'https://stablediffusionapi.com/api/v3/text2img'
BACKUP_API_KEY = 'Kep5yIY9sU58fd2ZXciZKc37bnLdRfjpu0fUvur74bVaz73wJVerd6BJjXYe'
MAX_ATTEMPTS = 3

# Flask app initialization
app = Flask(__name__)

neg_prompt = "(worst quality:1.2), (low quality:1.2), (lowres:1.1), multiple views, comic, sketch, (((bad anatomy))), (((deformed))), (((disfigured))), watermark, multiple_views, mutation hands, mutation fingers, extra fingers, missing fingers, watermark"
enhance_keywords = "((best quality)), ((masterpiece)), (detailed),"


# FaunaDB client setup
FAUNA_SECRET = "fnAFNtTsG9AARFSwW429OKB31VOr71ICCRPWHvbI"
client = FaunaClient(secret=FAUNA_SECRET)

def get_server_ref_by_url(server_url):
    server_data = client.query(q.get(q.match(q.index("servers_by_url"), server_url)))
    return server_data["ref"]

def mark_server_as_in_use(server_url, in_use_status):
    """Marks a server as in use or free."""
    server_ref = get_server_ref_by_url(server_url)
    print(f"Marking server {server_url} as {'in use' if in_use_status else 'available'}.")
    client.query(
        q.update(server_ref, {"data": {"in_use": in_use_status}})
    )

def save_image_to_fauna(image_data, hash_id):
    """Saves the image data to FaunaDB and returns the ref."""
    print(f"Storing image with hash {hash_id} to FaunaDB...")
    image_record = {
        "hash_id": hash_id,
        "data": image_data
    }
    result = client.query(q.create(q.collection("images"), {"data": image_record}))
    return result["ref"]

def get_image_from_fauna(hash_id):
    """Retrieve the image data from FaunaDB using the hash id."""
    try:
        image_record = client.query(q.get(q.match(q.index("images_by_hash_id"), hash_id)))
        return image_record["data"]["data"]
    except:
        print(f"Error fetching image with hash {hash_id} from FaunaDB.")
        return None


def get_available_servers():
    """Fetches available servers from FaunaDB and returns them."""
    try:
        refs = client.query(q.paginate(q.match(q.index("all_servers"), False)))["data"]

        # Get data for all refs using map
        servers_data = client.query(
            q.map_(
                lambda ref: q.get(ref),
                refs
            )
        )

        servers = [{"ref": server["ref"], "data": server["data"]} for server in servers_data]

        print(f"Available servers: {servers}")
        return servers
    except Exception as e:
        print(f"Error querying available servers: {e}")
        return []


def remove_server(server_ref):
    """Removes a server from FaunaDB."""
    print(f"Removing server with Ref: {server_ref}")
    client.query(q.delete(server_ref))


def assign_task_to_server(prompt):
    """Assign the image generation task to a server."""
    servers = get_available_servers()
    for server in servers:
        server_url = server['data']['url']
        server_ref = server['ref']
        image_data = send_task_to_server(prompt, server_url, server_ref)
        if image_data:
            return image_data
    return None


def send_task_to_server(prompt, server_url, server_ref):
    """Send the image generation task to the given server and return the image data if successful."""
    mark_server_as_in_use(server_url, True)  # Set in_use to True

    payload = {"prompt": enhance_keywords + prompt, "steps": 20, "negative_prompt": neg_prompt,}
    response = requests.post(url=f'{server_url}sdapi/v1/txt2img', json=payload)

    if response.status_code != 200:
        print(f"Error: Server {server_url} responded with status code {response.status_code}.")
        print("Response data:", response.text)
    elif 'images' not in response.json():
        print(f"Error: 'images' key not found in server {server_url} response.")
        print("Response data:", response.json())
    else:
        print(f"Image generated successfully from server: {server_url}")
        mark_server_as_in_use(server_url, False)  # Set in_use to False after success
        return response.json()['images'][0]

    print(f"Failed to generate image from server: {server_url}. Removing server from available servers.")
    remove_server(server_ref)
    return None


def decrement_server_thread(server_url):
    """Decrements thread count of a server."""
    print(f"Attempting to reduce thread count for server: {server_url}")

    # Fetch the server details from the database
    server = client.query(q.get(q.match(q.index("servers_by_url"), server_url)))['data']

    # Check if the server has more than 1 thread, otherwise remove it from available servers
    if server['threads'] > 1:
        new_thread_count = server['threads'] - 1
        client.query(
            q.update(q.ref(q.collection("servers"), server['id']),
                     {"data": {"threads": new_thread_count}})
        )
        print(f"Server {server_url} now has {new_thread_count} threads available.")
    else:
        remove_server(q.ref(q.collection("servers"), server['id']))



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

        # If the image is still processing, we will retry after the provided ETA
        if response.status_code == 200 and 'status' in response.json() and response.json()['status'] == "processing":
            eta = response.json().get('eta', 3)  # Using 3 seconds as default if ETA is not provided
            print(f"Image is still processing. Will retry after {eta} seconds.")
            time.sleep(eta)

        # If the image generation was successful, return the generated image URL
        elif response.status_code == 200 and 'output' in response.json() and response.json()['output']:
            return response.json()['output'][0]
        else:
            print(f"Attempt {attempt + 1} failed. Retrying...")
            attempt += 1
            time.sleep(3)  # Wait for 3 seconds before retrying

    return 'https://i.imgur.com/tdGdu9l.png'  # Default backup image if all attempts fail

@app.route('/generate-image', methods=['POST'])
def generate_image():
    print(request.json)
    if not 'prompt' in request.json:
        return jsonify({'error': 'Bad request data'}), 400

    image_data = assign_task_to_server(request.json['prompt'])

    if not image_data:
        print("All servers failed. Attempting backup...")
        backup_image_url = backup_image_generation(request.json['prompt'])
        return jsonify({"image_url": backup_image_url}), 200

    # Check and extract image data based on the presence of MIME type prefix
    image_data_parts = image_data.split(",", 1)
    if len(image_data_parts) == 2:
        image_data = image_data_parts[1]
    else:
        image_data = image_data  # Use the entire data if no prefix

    # Generate filename using hash
    filename = hashlib.md5(image_data.encode()).hexdigest()

    # Directly save to FaunaDB without saving to local storage
    image_ref = save_image_to_fauna(image_data, filename)

    return jsonify({"image_url": f"https://image-labs.onrender.com/images/{filename}"})


@app.route('/images/<hash_id>', methods=['GET'])
def retrieve_image(hash_id):
    image_data = get_image_from_fauna(hash_id)
    if not image_data:
        return jsonify({"error": "Image not found"}), 404

    # Convert the base64 string back to binary
    image_binary = io.BytesIO(base64.b64decode(image_data))
    return send_file(image_binary, mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)

