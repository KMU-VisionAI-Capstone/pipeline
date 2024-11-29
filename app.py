import gradio as gr
from input_classification.classification import classify
from image_captioning.captioning_baseft import captioning
from utils.es_config import fetch_matching_data
from utils.embedding import generate_image_embedding, generate_text_embedding
from utils.similarity import rank_matches


def get_classification_ui(predictions):
    """분류 결과 UI를 반환합니다."""
    html_result = "<h2>Top Predictions</h2>"
    for label, score in predictions:
        html_result += f"<p><strong>{label}:</strong> {score:.2f}%</p>"
    return html_result


def get_caption_ui(caption):
    """캡션 결과 UI를 반환합니다."""
    html_result = f"<h2>Generated Caption</h2><p>{caption}</p>"
    return html_result


def update_caption_ui(image_input):
    """이미지 캡션 결과를 HTML로 반환"""
    caption = captioning(image_input)
    caption_html = get_caption_ui(caption)
    return caption, caption_html


def on_click(image_input, alpha):
    """버튼 클릭 시 워크플로우 수행"""
    # 분류 수행
    predictions = classify(image_input)
    classification_html = get_classification_ui(predictions)

    # 캡셔닝 생성
    caption_text, caption_html = update_caption_ui(image_input)

    # 이미지 및 텍스트 임베딩 생성
    image_embedding = generate_image_embedding(image_input)
    text_embedding = generate_text_embedding(caption_text)
    print("Embeddings Generated")

    # Elasticsearch에서 매칭 데이터 가져오기
    matching_data = fetch_matching_data(predictions)
    print(f"Matching Data Length: {len(matching_data)}")

    # 매칭 데이터와 코사인 유사도 계산
    ranked_results = rank_matches(image_embedding, text_embedding, matching_data, alpha)

    # 결과 출력 HTML 생성
    gallery = []
    results_html = "<h2>Top Matching Results</h2>"
    for file_name, similarity, file_path in ranked_results[:5]:  # 상위 5개 결과만 표시
        results_html += f"<p><strong>{file_name}:</strong> {similarity:.4f}</p>"
        gallery.append(file_path)

    print("Workflow Completed\n")

    return classification_html, caption_html, results_html, gallery


# Gradio 인터페이스 생성
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Input Image")
            alpha_slider = gr.Slider(
                0, 1, value=0.5, step=0.1, label="Alpha (Image-Text Weight)"
            )
            predict_button = gr.Button("Run Workflow")
        with gr.Column():
            classification_output = gr.HTML(label="Classification Results")
            caption_output = gr.HTML(label="Caption Results")
            matching_results_output = gr.HTML(label="Matching Results")
    with gr.Column():
        gallery = gr.Gallery(
            interactive=False,
            label="Top Matching Images",
            columns=5,
            height=300,
        )

    predict_button.click(
        fn=on_click,
        inputs=[image_input, alpha_slider],
        outputs=[
            classification_output,
            caption_output,
            matching_results_output,
            gallery,
        ],
    )


demo.launch(share=True)
