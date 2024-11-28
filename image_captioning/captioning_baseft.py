import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration,AutoProcessor, AutoConfig

import numpy as np
import warnings

warnings.filterwarnings("ignore")


weight_path = "./image_captioning/epoch8_val_loss1.4723037481307983.pt"

# 디바이스 설정
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.empty_cache()

# 모델 및 Processor 로드
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft" ,torch_dtype=torch.bfloat16, trust_remote_code=True).to(device) 
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft",trust_remote_code=True)


# Fine-tuned 가중치 로드 (선택 사항)
if weight_path:
    print(f"Loading fine-tuned weights from: {weight_path}")
    trained_weight = torch.load(weight_path, map_location=device)
    model.load_state_dict(trained_weight)
model.eval()


@torch.no_grad()
def captioning(image_input):
    """
    이미지 ndarray를 입력받아 캡션을 생성.
    """
    # numpy 배열을 PIL 이미지로 변환
    image = Image.fromarray(image_input).convert("RGB")

    
    task = "<MORE_DETAILED_CAPTION>"
    text = "<MORE_DETAILED_CAPTION>"

    # Processor로 입력 전처리
    inputs = processor(text=text, images=image, return_tensors="pt").to(device,torch.bfloat16)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
        )
    # 모델로 캡션 생성
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    caption = processor.post_process_generation(generated_text, task=task, image_size=(image.width, image.height))

    # 결과 반환 (캡션 문자열)
    return caption


if __name__ == "__main__":

    # 테스트용 이미지 불러오기
    test_image_path = (
        "/home/jmkim/dev/capstone/test_data/tower.jpg"  # 실제 이미지 경로 입력
    )

    # 이미지를 numpy 배열로 변환
    test_image = Image.open(test_image_path).convert("RGB")
    test_image_array = np.array(test_image)

    # 이미지 ndarray를 main에 전달
    caption = captioning(test_image_array)
