import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# 디바이스 설정
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.empty_cache()

# 모델 및 Processor 로드
model_id = "./image_captioning/epoch12_val_loss1.102"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
config.vision_config.model_type = "davit"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
).to(device)
processor = AutoProcessor.from_pretrained(
    model_id, trust_remote_code=True, config=config
)


@torch.no_grad()
def captioning(image_input):
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:  # numpy 배열
        image = Image.fromarray(image_input).convert("RGB")

    task = "<MORE_DETAILED_CAPTION>"
    text = "<MORE_DETAILED_CAPTION>"

    # Processor로 입력 전처리
    inputs = processor(text=text, images=image, return_tensors="pt").to(
        device, torch.bfloat16
    )
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
    )

    # 모델로 캡션 생성
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
    )[0]
    caption = processor.post_process_generation(
        generated_text,
        task=task,
        image_size=(
            image.width,
            image.height,
        ),
    )
    caption = caption[task].replace("<pad>", "")

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
    print(caption)
