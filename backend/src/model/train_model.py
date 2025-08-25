
from sklearn.metrics import accuracy_score
import torch
import numpy as np
import requests
import random
from PIL import ImageDraw, ImageFont, Image
from datasets import load_dataset
from transformers import AutoModelForImageClassification, AutoImageProcessor, Trainer, TrainingArguments

## ================== Data Pre-processing ==================

### Pre-trained Model Fine-tuning Documents [https://huggingface.co/blog/fine-tune-vit?utm_source=chatgpt.com]
# 1. Image must transfer to PIL
# 2. Label Issue ➡️ 可透過 Hugging Face 的 datasets.ImageFolder 自動生成 label

# drop_labels = False ➡️ use folder name as label
dataset = load_dataset("imagefolder", data_dir="backend/src/model/images/", drop_labels=False) 
labels = dataset['train'].features['label'].names   # 這邊改成單數 label
num_labels = len(labels)
print(f"Label Name: {labels}")

# ---------------------------------------------------------
def show_image_with_label(dataset, examples_per_class: int = 3, size=(350, 350)):
    seed = 20
    w, h = size

    labels = dataset['train'].features['label'].names   # 這邊改成單數 label
    grid = Image.new('RGB', size=(examples_per_class * w, len(labels) * h)) # build grid to store images
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default(size = 20.0)

    for label_id, label in enumerate(labels):
        # 選出指定 label 的圖片
        dataset_slice = dataset['train'].filter(lambda ex: ex['label'] == label_id).shuffle(seed).select(range(examples_per_class))

        for i, example in enumerate(dataset_slice):
            image = example['image']
            box = (i * w, label_id * h)  # 這邊要修正排版
            grid.paste(image.resize(size), box=box)
            # 在左上角貼 label 文字
            draw.text((box[0] + 10, box[1] + 10), label, (255, 255, 255), font=font)

    return grid

# img = show_image_with_label(dataset)
# img.show()  # Uncomment to preview image locally


## ============== Transformers Download Model ==============
image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224",
    use_fast=True,
    force_download=True
)

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    ignore_mismatched_sizes=True,                            # -> to ignore the default size (1000,768) vs (N,768), when N != 1000 / 替換分類 head
    id2label={i: label for i, label in enumerate(labels)},   # -> index and label 的映射
    label2id={label: i for i, label in enumerate(labels)},
    num_labels=num_labels,                                   # -> 基於照片種類建立分類 head / 若無此設定，預設會用 ImageNet 的原始分類 head
    force_download=True,
)

## ============= Pre-trained Model Fine-tuning =============

### Transfer PIL Image to model-used pixel_values
def preprocess(example):
    inputs = image_processor(example["image"], return_tensors="pt")
    example["pixel_values"] = inputs["pixel_values"].squeeze(0).to(torch.float32)  # 移除 batch 維度
    return example

### 對整個 dataset 執行 preprocess，
for split in dataset:
    dataset[split] = dataset[split].map(preprocess)
    dataset[split].set_format(type="torch", columns=["pixel_values", "label"])

### Define Data Collator
def collate_fn(examples):
    # 保險起見都轉 float32
    pixel_values = torch.stack([ex["pixel_values"].to(torch.float32) for ex in examples])
    labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.long)
    # Debug 型態
    print("--- [DEBUG BATCH INFO] ---")
    print("pixel_values dtype:", pixel_values.dtype, "shape:", pixel_values.shape)
    print("labels dtype:", labels.dtype, "shape:", labels.shape)
    print("--- END DEBUG ---")
    return {"pixel_values": pixel_values, "labels": labels}

### Fine-tuning
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

### Training
args = TrainingArguments(
    output_dir="vit-finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    num_train_epochs=3,
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["train"],
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=image_processor,
)

trainer.train()
trainer.save_model("backend/src/model/vit-finetuned")

print("模型已儲存！")