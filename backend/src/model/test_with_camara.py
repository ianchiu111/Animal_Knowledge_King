import cv2
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image

# 載入 fine‑tuned 模型
model = AutoModelForImageClassification.from_pretrained("backend/src/model/vit-finetuned").eval()
image_processor = AutoImageProcessor.from_pretrained("backend/src/model/vit-finetuned")

cam = cv2.VideoCapture(0)
cv2.namedWindow("Press SPACE to capture, ESC to exit")

while True:
    ret, frame = cam.read()
    if not ret:
        break
    cv2.imshow("Press SPACE to capture, ESC to exit", frame)
    key = cv2.waitKey(1)
    if key % 256 == 27:  # ESC
        break
    elif key % 256 == 32:  # SPACE
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = image_processor(img, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = logits.argmax(-1).item()
        label = model.config.id2label[pred]
        print("Predicted:", label)

cam.release()
cv2.destroyAllWindows()
