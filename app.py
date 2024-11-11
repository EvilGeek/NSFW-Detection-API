import requests, urllib3, json, re, io, os
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify
import base64
from nsfw_detector import predict

urllib3.disable_warnings()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "NSFW_IMAGE_DETECTOR")  # Secret key from env
MAX_IMAGE_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", 16))
app.config['MAX_CONTENT_LENGTH'] = MAX_IMAGE_SIZE * 1024 * 1024  # 16 MB limit



try:
    model = predict.load_model('nsfw_detector/nsfw_model.h5')
except Exception as e:
    print(f"Failed to load the model: {e}")
    model = None


# Optimized prediction function
def prediction(image_bytes):
    try:
        results = predict.classify_bytes(model, image_bytes)
        hentai = results['data']['hentai']
        sexy = results['data']['sexy']
        porn = results['data']['porn']
        drawings = results['data']['drawings']
        neutral = results['data']['neutral']
        
        nsfw_score = sexy + porn + hentai
        is_nsfw = nsfw_score >= 70 and neutral < 25 and drawings < 40

        results['data']['is_nsfw'] = is_nsfw
        results['data']['predominant_class'] = max(results['data'], key=results['data'].get)  # Adding predominant class
        return results
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None


def image_content(url):
    try:
        resp = requests.get(url)
        if resp.status_code != 200:
            return {"error": "Unexpected response received from server."}
        if not resp.headers['Content-Type'].startswith('image/'):
            return {"error": "Not a valid image response from server."}

        # Load the image bytes and identify format
        image_bytes = io.BytesIO(resp.content)

        try:
            img = Image.open(image_bytes)
            if img.format not in ['JPEG', 'PNG', 'WEBP']:  # Supported formats
                return {"error": f"Unsupported image format: {img.format}"}
            
            # Convert to RGB in case it's a format like PNG or WebP with transparency
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Resize image to the size expected by the model (224x224 in this case)
            img = img.resize((224, 224))
            
            # Save image to bytes again (model expects bytes)
            img_byte_array = io.BytesIO()
            img.save(img_byte_array, format='JPEG')  # Saving as JPEG since model works on pixels
            img_byte_array = img_byte_array.getvalue()
            
            return {"content": img_byte_array}

        except UnidentifiedImageError:
            return {"error": "Cannot identify image file format."}

    except Exception as e:
        return {"error": "Internal Server Error."}

def process_raw_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.format not in ['JPEG', 'PNG', 'WEBP']:  # Supported formats
            return {"error": f"Unsupported image format: {img.format}"}
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize the image to 224x224 (expected size for the model)
        img = img.resize((224, 224))

        # Convert image to bytes (JPEG format for processing)
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format='JPEG')
        img_byte_array = img_byte_array.getvalue()
        return img_byte_array

    except UnidentifiedImageError:
        return {"error": "Cannot identify image format."}

# Basic URL validation function
def is_valid_url(url):
    regex = re.compile(
        r'^(https?|ftp):\/\/'                # http://, https://, or ftp://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'  # Domain or subdomain
        r'[A-Z]{2,6}\.?|'                    # .com, .org, .net, etc.
        r'localhost|'                        # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IPv4
        r'(?::\d+)?(\/[^\s]*)?$',            # Optional port and path
        re.IGNORECASE                        # Case insensitive
    )
    return re.match(regex, url) is not None


@app.route("/")
def home_api():
    """Health check endpoint that lists all available endpoints."""
    endpoints = {
        "/predict": "Predict if an image is NSFW using a URL or uploaded file.",
        "/ping": "Check server status.",
    }
    return jsonify({"status": "OK", "available_endpoints": endpoints})


@app.route("/ping/")
@app.route("/ping")
def ping():
    return jsonify({"message": "pong", "status": "success"}), 200


@app.route("/predict/", methods=['GET', 'POST'])
@app.route("/predict", methods=['GET', 'POST'])
def predict_api():
    image_bytes = None  

    if request.method == 'GET':
        url = request.args.get('url')
        if not url:
            return jsonify({"status": False, "error": "No Image URL provided."}), 400
        if not is_valid_url(url):
            return jsonify({"status": False, "error": "Not a valid Image URL provided."}), 400
        content_response = image_content(url)
        if "error" in content_response:
            return jsonify({"status": False, "error": content_response['error']}), 400
        image_bytes = content_response['content']

    elif request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"status": False, "error": "No image file provided."}), 400

        try:
            image_file = request.files['image']
            image_bytes = image_file.read()

            image_bytes = process_raw_image(image_bytes)
            if "error" in image_bytes:
                return jsonify({"status": False, "error": image_bytes['error']}), 400

        except Exception as e:
            return jsonify({"status": False, "error": "Error processing the image: " + str(e)}), 400

    if image_bytes is None:
        return jsonify({"status": False, "error": "Image data could not be processed."}), 500

    try:
       # print(image_bytes)
        predictions = prediction(image_bytes)
        if predictions:
            predictions["status"] = True
            return jsonify(predictions)
        return jsonify({"status": False, "error": "The request cannot be processed."}), 500
    except Exception as e:
        return jsonify({"status": False, "error": "Error processing image: " + str(e)}), 500



@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"status": False, "error": "Not Found", "message": "The requested resource could not be found."}), 404



if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", threaded=True)
