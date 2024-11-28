import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import os
from sentence_transformers import SentenceTransformer
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
device = "cuda" if torch.cuda.is_available() else "cpu"

# BLIP 모델 로드
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_model = blip_model.to(device)
blip_model.eval()

# Sentence-BERT 모델 로드
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
sbert_model.to(device)
sbert_model.eval()


def generate_image_embedding(image_input):
    """BLIP으로 이미지 임베딩 벡터 생성"""
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input)
    else:
        raise ValueError("Input should be a file path or a numpy array")

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        # 이미지 임베딩 생성
        image_embedding = blip_model.vision_model(**inputs).pooler_output
    return image_embedding.squeeze().cpu().numpy()


def generate_text_embedding(caption):
    """Sentence-BERT로 텍스트 임베딩 벡터 생성"""
    if not isinstance(caption, str):
        raise ValueError("Input should be a string")
    text_embedding = sbert_model.encode(caption)
    return text_embedding
