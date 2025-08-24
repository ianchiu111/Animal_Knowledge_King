import tkinter as tk
from tkinter import filedialog
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch

# 載入模型
model = AutoModelForImageClassification.from_pretrained("backend/src/model/vit-finetuned").eval()
image_processor = AutoImageProcessor.from_pretrained("backend/src/model/vit-finetuned")

root = tk.Tk()
root.withdraw()  # 隱藏主視窗
file_path = filedialog.askopenfilename(
    title="Select Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

if file_path:
    img = Image.open(file_path).convert("RGB")
    inputs = image_processor(img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = logits.argmax(-1).item()
    label = model.config.id2label[pred]
    print("Predicted:", label)
else:
    print("No image selected.")
