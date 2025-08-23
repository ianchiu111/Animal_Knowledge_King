
import torch
import requests
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

## ============== Transformers Download Model ==============
image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224",
    use_fast=True,
    force_download=True
)

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    torch_dtype=torch.float16,
    # device_map="auto",
    attn_implementation="sdpa",  ## Scaled Dot-Product Attention
    force_download=True
).to("cuda")                     ## .to(cuda) ➡️ use GPU


## ================== Data Pre-processing ==================

### Pre-trained Model Fine-tuning Documents [https://huggingface.co/blog/fine-tune-vit?utm_source=chatgpt.com]
# 1. Image must transfer to PIL
# 2. Label Issue ➡️ 可透過 Hugging Face 的 datasets.ImageFolder 自動生成 label
from datasets import load_dataset

# drop_labels = False ➡️ use folder name as label
dataset = load_dataset("imagefolder", data_dir="backend/src/model/images/", drop_labels=False) 
print(dataset["train"][0])                       # -> 包含 'image' 和 'label' 欄位
print(dataset["train"].features["label"].names)  # -> label name




## ============= Pre-trained Model Fine-tuning =============




## ============= Image Classification Testing =============

img = Image.open("backend/src/model/images/yellow_shiba_inu/1.jpg")
inputs = image_processor(img, return_tensors="pt").to("cuda")

with torch.no_grad():
  logits = model(**inputs).logits
predicted_class_id = logits.argmax(dim=-1).item()

class_labels = model.config.id2label
predicted_class_label = class_labels[predicted_class_id]
print(f"The predicted class label is: {predicted_class_label}")