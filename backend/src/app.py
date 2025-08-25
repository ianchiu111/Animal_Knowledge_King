from flask import Flask, request, jsonify
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image
import io
from flask_cors import CORS


# 初始化模型
model = AutoModelForImageClassification.from_pretrained("model/vit-finetuned").eval()
image_processor = AutoImageProcessor.from_pretrained("model/vit-finetuned")

app = Flask(__name__)
CORS(app)  ## 讓前端能跨網域存取 API

@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    inputs = image_processor(img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = logits.argmax(-1).item()
    label = model.config.id2label[pred]
    return jsonify({"label": label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
