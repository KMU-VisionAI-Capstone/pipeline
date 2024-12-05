import torch
from torchvision import models, transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import numpy as np

# 전역 변수
MODEL_ARCH = "efficientnet-b1"  # 모델 아키텍처
CHECKPOINT_PATH = (
    "./input_classification/model_best_20241129-005608.pth.tar"  # 체크포인트 파일 경로
)
CLASS_MAPPING_FILE = (
    "./input_classification/class_mapping.txt"  # 클래스 이름 매핑 파일 경로
)
IMAGE_SIZE = (224, 224)  # 입력 이미지 크기
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 사용 여부


def load_model(checkpoint_path, model_arch):
    """모델을 초기화하고 체크포인트에서 가중치를 로드합니다."""
    if "efficientnet" in model_arch:
        model = EfficientNet.from_name(model_arch)
    else:
        model = models.__dict__[model_arch]()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(DEVICE)
    model.eval()  # 평가 모드로 전환
    return model


def load_class_mapping(class_mapping_file):
    """클래스 이름 매핑 파일을 로드합니다."""
    idx_to_class = {}
    with open(class_mapping_file, "r") as f:
        for line in f:
            idx, class_name = line.strip().split(": ")
            idx_to_class[int(idx)] = class_name
    return idx_to_class


# 모델 로드
model = load_model(CHECKPOINT_PATH, MODEL_ARCH)

# 클래스 이름 매핑 로드
idx_to_class = load_class_mapping(CLASS_MAPPING_FILE)


def preprocess_image(input_data, image_size):
    """이미지를 불러와 전처리합니다."""
    image_transforms = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # NumPy 배열을 PIL 이미지로 변환
    if isinstance(input_data, np.ndarray):
        if input_data.ndim == 2:  # 흑백 이미지 처리
            input_data = np.stack([input_data] * 3, axis=-1)
        input_data = Image.fromarray(input_data.astype("uint8"))

    # 파일 경로일 경우 처리
    elif isinstance(input_data, str):  # 이미지 파일 경로
        input_data = Image.open(input_data).convert("RGB")

    # PIL 이미지를 전처리
    input_tensor = image_transforms(input_data).unsqueeze(0)  # 배치 차원 추가
    return input_tensor.to(DEVICE)


def predict(input_data, top_k=5):
    """모델로 이미지를 예측하고 상위 k개 클래스를 반환합니다."""
    input_tensor = preprocess_image(input_data, IMAGE_SIZE)
    with torch.no_grad():
        output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    topk_probs, topk_indices = torch.topk(probabilities, k=top_k)

    results = []
    for i in range(topk_probs.size(0)):
        class_name = idx_to_class.get(topk_indices[i].item(), "Unknown")
        results.append((class_name, topk_probs[i].item() * 100))
    return results


def classify(input_data):
    """메인 실행 함수."""
    # 예측 수행
    predictions = predict(input_data)

    # 결과 출력
    for class_name, prob in predictions:
        print(f"  {class_name}: {prob:.2f}%")

    return predictions


if __name__ == "__main__":

    # 이미지 파일 경로 테스트
    classify(
        input_data="/home/jmkim/dev/capstone/test_data/tower.jpg"
    )  # 실제 이미지 경로 입력
