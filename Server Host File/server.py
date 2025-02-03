from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import cv2
import numpy as np
from ultralytics import YOLO
import os

app = Flask(__name__)

# url model
try:
    url_model_path = os.path.abspath("/Users/devsriakash/Desktop/Telegram Sentinel/URL Separation/url-classification-model")
    tokenizer_url_model = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    url_model = AutoModelForSequenceClassification.from_pretrained(url_model_path)
    print("✅ URL Classification Model Loaded Successfully")
except Exception as e:
    url_model = None
    print(f"❌ Error Loading URL Classification Model: {e}")

# yolo
try:
    yolo_model_path = os.path.abspath("/Users/devsriakash/Desktop/Telegram Sentinel/YOLO_Model/detect/YOLO_Model/train2/weights/best.pt")
    yolo_model = YOLO(yolo_model_path)
    print("✅ YOLO Object Detection Model Loaded Successfully")
except Exception as e:
    yolo_model = None
    print(f"❌ Error Loading YOLO Model: {e}")

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
if url_model:
    url_model.to(device)

# url endpoint
@app.route("/predict_url", methods=["POST"])
def predict_url():
    if not url_model:
        return jsonify({"error": "URL classification model is not available"}), 500

    data = request.json
    if "url" not in data:
        return jsonify({"error": "Missing 'url' field"}), 400

    url = [data["url"]]
    inputs = tokenizer_url_model(url, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = url_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

    result = "Benign" if prediction == 0 else "Malicious"
    return jsonify({"url": data["url"], "prediction": result})

# yolo endpoint
@app.route("/predict_image", methods=["POST"])
def predict_image():
    if not yolo_model:
        return jsonify({"error": "YOLO model is not available"}), 500

    if "image" not in request.files:
        return jsonify({"error": "Missing image file"}), 400

    image_file = request.files["image"]
    image_np = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    results = yolo_model(image)

    detections = []
    for result in results:
        for box in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, confidence, class_id = box[:6]
            detections.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "confidence": float(confidence),
                "class_id": int(class_id)
            })

    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True) # I use Port 5001 usually