import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(query_embedding, target_embedding):
    """코사인 유사도 계산"""
    query_embedding = np.expand_dims(query_embedding, axis=0)
    target_embedding = np.expand_dims(target_embedding, axis=0)
    return cosine_similarity(query_embedding, target_embedding)[0][0]


def rank_matches(image_embedding, text_embedding, matching_data, alpha):
    """매칭된 데이터를 순회하며 코사인 유사도 계산"""
    ranked_results = []
    for data in matching_data:
        
        if alpha == 0:
            # 데이터에서 임베딩 추출
            target_text_embedding = np.array(data["text_embedding"])
            text_similarity = compute_cosine_similarity(
                text_embedding, target_text_embedding
            )
            ranked_results.append(
                (text_similarity, data["file_path"])
            )
        elif alpha == 1:
            target_image_embedding = np.array(data["image_embedding"])
            # 이미지-이미지 및 텍스트-텍스트 코사인 유사도 계산
            image_similarity = compute_cosine_similarity(
                image_embedding, target_image_embedding
            )
            ranked_results.append(
                (image_similarity, data["file_path"])
            )
        else:
            target_image_embedding = np.array(data["image_embedding"])
            target_text_embedding = np.array(data["text_embedding"])
            image_similarity = compute_cosine_similarity(
                image_embedding, target_image_embedding
            )
            text_similarity = compute_cosine_similarity(
                text_embedding, target_text_embedding
            )

            # 가중치 조합
            combined_similarity = alpha * image_similarity + (1 - alpha) * text_similarity
            ranked_results.append(
                (combined_similarity, data["file_path"])
            )

    # 유사도 기준 내림차순 정렬
    ranked_results.sort(key=lambda x: x[0], reverse=True)
    return ranked_results
