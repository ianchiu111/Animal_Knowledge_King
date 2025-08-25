
## Vision Transformer (ViT)

### 參數設定
1. device_map
    1. when = "auto": 自動決定模型該放在哪個裝置（CPU/GPU）上執行
        1. 使用 device_map="auto" 必須安裝 Accelerate
2. image
    1. ✴️image✴️ 應該要放圖片物件本身
    ```python
    image_processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224",
        use_fast=True,
    )

    inputs = image_processor(✴️image✴️, return_tensors="pt").to("cuda")
    ```

### Pipeline 🆚 AutoModel
1. Pipeline
    - Transformers 套件封裝了前處理、模型執行、後處理等流程
    - 執行較簡單
    - Example Code: Pipeline 插在模型應用流程的最外層，用於一行就完成圖像分類推論工作
    ```python 
    from transformers import pipeline

    pipeline = pipeline(
        task="image-classification",
        model="google/vit-base-patch16-224",
        torch_dtype=torch.float16,
        device=0
    )
    pipeline(images="...cat image url...")
    ```
2. AutoModel 
    - 較底層的方式，可自行控制輸入、處理、推論流程
    - 
    ```python 
    from transformers import AutoImageProcessor, ViTForImageClassification
    import torch

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    inputs = image_processor(image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits    
    ```

總結：用 Pipeline 去確認模型的辨識效果，再用 AutoModel 做開發