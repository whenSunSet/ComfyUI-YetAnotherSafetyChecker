from transformers import pipeline
from torchvision import transforms
import torch
import os
import folder_paths

class YetAnotherSafetyChecker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.8, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01
                }),
                "cuda": ("BOOLEAN", {"default": False}),
                "model_dir": ("STRING", {"default": "nsfw"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "process_images"

    CATEGORY = "image/processing"

    def process_images(self, image, threshold, cuda, model_dir):
        if cuda:
            device = "cuda"
        else:
            device = "cpu"
            
        # 构建模型路径
        MODELS_DIR = os.path.join(folder_paths.models_dir, model_dir)
        if not os.path.exists(MODELS_DIR):
            raise Exception(f"Model directory {MODELS_DIR} not found.")
            
        # 检查模型目录是否包含必要的文件
        required_files = ["config.json"]
        for file in required_files:
            if not os.path.exists(os.path.join(MODELS_DIR, file)):
                raise Exception(f"Required model file {file} not found in {MODELS_DIR}.")
                
        predict = pipeline("image-classification", model=MODELS_DIR, device=device)
        result = (predict(transforms.ToPILImage()(image[0].cpu().permute(2, 0, 1)))) #Convert to expected format
        score = next(item['score'] for item in result if item['label'] == 'nsfw')
        output = image
        if(float(score) > threshold):
            output = torch.zeros(1, 512, 512, dtype=torch.float32) #create black image tensor
        return (output, str(score))
    
NODE_CLASS_MAPPINGS = {
    "YetAnotherSafetyChecker": YetAnotherSafetyChecker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YetAnotherSafetyChecker": "Intercept NSFW Outputs"
}
