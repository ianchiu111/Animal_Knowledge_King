import os
from dotenv import load_dotenv
load_dotenv(".env")

import torch
from PIL import Image
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoModelForImageClassification, AutoImageProcessor, Trainer, TrainingArguments

class vitTraining:
    """
    step 1: _set_labels()
    step 2: _load_pretrained_model()
    step 3: fineTuned()
    step 4-1: _load_inference_model() -> inference()
    step 4-2: inference()
    """
    def __init__(self):
        raw_dataset = load_dataset(
            "imagefolder", 
            data_dir="backend/src/model/images/", 
            drop_labels=False,
        )   
        train_test_ds = raw_dataset['train'].train_test_split(test_size=0.3, seed=42)
        test_val_ds = train_test_ds['test'].train_test_split(test_size=0.5, seed=42)
        self.dataset = DatasetDict({
            'train': train_test_ds['train'], 
            'test': test_val_ds['test'],   
            'validate': test_val_ds['train']     
        })

        self.BACKBONE_ARCH = os.getenv("BACKBONE_ARCH")
        self.model_path = "backend/src/model/vit-finetuned"
        self.inf_model = None
        self.inf_processor = None

        self.args = TrainingArguments(
            output_dir="backend/src/model/vit-checkpoints",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            eval_strategy="epoch", 
            save_strategy="epoch",    
            num_train_epochs=3,
            learning_rate=5e-5,
            remove_unused_columns=False,    # 配合自定義的 collate_fn 使用
            load_best_model_at_end=True
        )
    
    def _set_labels(self):
        """
        concept: use Hugging Face datasets package to automatically generate labels based on folder names
        return: the label names as a list
        """
        labels = self.dataset['train'].features['label'].names
        return labels, len(labels)
    
    def _load_pretrained_model(self):
        """
        concept: download Hugging Face pre-trained ViT model and its corresponding image processor
        return: the image processor and the pre-trained model
        """
        labels, num_labels = self._set_labels()

        image_processor = AutoImageProcessor.from_pretrained(
            self.BACKBONE_ARCH,
            use_fast=True,
            force_download=False
        )

        pretrained_model = AutoModelForImageClassification.from_pretrained(
            self.BACKBONE_ARCH,
            ignore_mismatched_sizes=True,                            
            id2label={i: label for i, label in enumerate(labels)},   
            label2id={label: i for i, label in enumerate(labels)},
            num_labels=num_labels,                                  
            force_download=False,
        )

        device = torch.device("mps")
        pretrained_model.to(device)

        return image_processor, pretrained_model

    def _load_inference_model(self):
        
        if os.path.exists(self.model_path):
            try:
                print(f"Load model from {self.model_path} ...")
                self.inf_model = AutoModelForImageClassification.from_pretrained(self.model_path).eval()
                self.inf_processor = AutoImageProcessor.from_pretrained(self.model_path)
                print("Load model successfully!")
            except Exception as e:
                print(f"Failed to load model, the model files might be incomplete: {e}")
        else:
            print(f"Warning: Model path {self.model_path} not found, please run fine_tune() first.")

    def fineTuned(self, existed: bool):

        if existed:
            print("Model already exists. Skipping training.")
            return

        image_processor, pretrained_model = self._load_pretrained_model()

        ### Transfer PIL Image to model-used pixel_values
        def preprocess(example):
            inputs = image_processor(
                example["image"], 
                return_tensors="pt"
            )
            example["pixel_values"] = inputs["pixel_values"].squeeze(0).to(torch.float32)  # 移除 batch 維度
            return example
        
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

        
        ### 對整個 dataset 執行 preprocess，
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].map(preprocess)
            self.dataset[split].set_format(type="torch", columns=["pixel_values", "label"])

        ### Training
        trainer = Trainer(
            model=pretrained_model,
            args=self.args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            data_collator=collate_fn,
            tokenizer=image_processor,
        )

        trainer.train()
        trainer.save_model(self.model_path)

    def inference(self, img: Image.Image):
        """
        Input: image (PIL Image)
        Output: predicted labels
        """

        if self.inf_model is None or self.inf_processor is None:
            self._load_inference_model()

            if self.inf_model is None or self.inf_processor is None:
                print("Inference model is not available.")
                return None, None

        inputs = self.inf_processor(img, return_tensors="pt")
        with torch.no_grad():
            logits = self.inf_model(**inputs).logits
            
        pred = logits.argmax(-1).item()
        label = self.inf_model.config.id2label[pred]
        return pred, label

# if __name__ == "__main__":
#     vit_training = vitTraining()
#     vit_training.fineTuned(existed=False)
#     # vit_training.inference()