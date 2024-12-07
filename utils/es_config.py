import os
from elasticsearch import Elasticsearch
import dotenv

dotenv.load_dotenv()

PASSWORD = os.getenv("ELASTIC_PASSWORD", None)

# Elasticsearch 클라이언트 설정
es = Elasticsearch(
    "https://localhost:9200",
    verify_certs=False,
    ssl_show_warn=False,
    basic_auth=("elastic", PASSWORD),
)


def fetch_matching_data(alpha=None, predictions=None, index=None):
    """Elasticsearch에서 predictions 카테고리에 해당하는 데이터 가져오기 (Scroll API 사용)"""

    results = []

    if predictions:
        categories = [label for label, _ in predictions]
        for category in categories:
            query = {
                "_source": ["file_path", "text_embedding", "image_embedding"],
                "query": {"wildcard": {"category": f"kor_{category}"}},
            }
            results.extend(scroll_search(index, query))
    else:
        if alpha == 0:
            query = {
                "_source": ["file_path", "text_embedding"],
                "query": {"match_all": {}},
            }
        elif alpha == 1:
            query = {
                "_source": ["file_path", "image_embedding"],
                "query": {"match_all": {}},
            }
        else:
            query = {
                "_source": ["file_path", "text_embedding", "image_embedding"],
                "query": {"match_all": {}},
            }
        results = scroll_search(index, query)

    return results


def scroll_search(index, query, scroll_time="1m", batch_size=1000):
    """Scroll API로 데이터를 가져오는 함수"""
    results = []
    try:
        # Scroll 초기화
        response = es.search(
            index=index, body=query, scroll=scroll_time, size=batch_size
        )
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]

        # Scroll 반복
        while hits:
            results.extend(hit["_source"] for hit in hits)

            # 다음 batch 가져오기
            response = es.scroll(scroll_id=scroll_id, scroll=scroll_time)
            scroll_id = response["_scroll_id"]
            hits = response["hits"]["hits"]

        # Scroll 컨텍스트 삭제
        es.clear_scroll(scroll_id=scroll_id)

    except Exception as e:
        print(f"Error during scroll: {e}")

    return results


if __name__ == "__main__":
    # 예시 predictions (분류 결과)
    predictions = [
        ("castle", 80.5),  # 91개
        # ("buddhist temple", 75.2),
        # ("museum", 60.1),
        # ("mountain", 50.0),
        # ("river", 45.3),
    ]

    matching_data = fetch_matching_data(predictions=predictions, alpha=0)

    print("Matching Data from Elasticsearch:")
    for data in matching_data:
        print(data)

    print(len(matching_data))
