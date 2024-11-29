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


def fetch_matching_data(predictions):
    """Elasticsearch에서 predictions 카테고리에 해당하는 데이터 가져오기"""
    categories = [label for label, _ in predictions]
    results = []
    for category in categories:
        query = {
            "size": 1000,
            "query": {"wildcard": {"category": f"kor_{category}"}},
        }
        try:
            response = es.search(index="embeddings", body=query)
            hits = response["hits"]["hits"]
            if hits:
                for hit in hits:
                    results.append(hit["_source"])
        except Exception as e:
            print(f"Error fetching category {category}: {e}")
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

    matching_data = fetch_matching_data(predictions)

    print("Matching Data from Elasticsearch:")
    for data in matching_data:
        print(data)

    print(len(matching_data))
