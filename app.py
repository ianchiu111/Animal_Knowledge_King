
import logging
import requests
import pytz
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

from backend.src.vitTrainModel import vitTraining

from dotenv import load_dotenv
load_dotenv(".env")

# ====================================================
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

taiwan_tz = pytz.timezone("Asia/Taipei")
current_time_taiwan = datetime.now(taiwan_tz)
version = current_time_taiwan.strftime("v%Y-%m%d-%H%M")

# ====================================================
vitModel = vitTraining()
vitModel.fineTuned(existed=False)

class APIResponse:
    """Static class for generating API responses"""

    @staticmethod
    def success(
        data: Any = None, message: str = "Success", meta: Optional[Dict] = None
    ):
        response = {
            "success": True,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        if data is not None:
            response["data"] = data
        if meta:
            response["meta"] = meta
        return response

    @staticmethod
    def error(
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Optional[str] = None,
    ):
        response = {
            "success": False,
            "error": {
                "message": message,
                "code": error_code,
                "timestamp": datetime.now().isoformat(),
            },
        }
        if details:
            response["error"]["details"] = details
        return response, status_code

@app.route("/api/vitModel/inference", methods=["POST"])
def inference():
    """
    Endpoint to receive an image file from the client, perform inference using the ViT model to predict the label.

    body: multipart/form-data with a file field named 'image'
    {
        "image": <image file>
    }
    """
    # check if the file is in the api request
    if 'image' in request.files:
        image = request.files['image']
    else:
        return jsonify({"error": "Missing the image file)"}), 400

    try:
        image_bytes = image.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        _, label = vitModel.inference(image)

        return jsonify({
            "status": "success",
            "prediction": label
        }), 200

    except Exception as e:
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500

@app.get("/")
def health_check():
    return {"status": f"server is running {version}"}

@app.get("/health")
def health():
    return {"status": f"server is running {version}"}


if __name__ == "__main__":
    port = 8080  # Default port
    logger.info(f"Starting Flask app on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)