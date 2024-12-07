from datetime import datetime
import gradio as gr
import time
from input_classification.classification import classify
import image_captioning.captioning_florence
import image_captioning.captioning_base_ft
from utils.es_config import fetch_matching_data
from utils.embedding import generate_image_embedding, generate_text_embedding
from utils.notion_config import push_to_notion
from utils.similarity import rank_matches


def update_classification_ui(predictions):
    """분류 결과 UI를 반환합니다."""
    classification_result = "<h2>Top Predictions</h2>"
    for label, score in predictions:
        classification_result += f"<p><strong>{label}:</strong> {score:.2f}%</p>"

    return classification_result


def update_caption_ui(image_input, index):
    """이미지 캡션 결과 UI를 반환합니다."""
    if index == "embeddings_florence":
        caption = image_captioning.captioning_florence.captioning(image_input)
    else:
        caption = image_captioning.captioning_base_ft.captioning(image_input)

    caption_result = f"<h2>Generated Caption</h2><p>{caption}</p>"

    return caption, caption_result


def update_gallery_ui(ranked_results):
    """갤러리 UI를 반환합니다."""
    gallery = []
    top_results = []
    results_html = "<h2>Top Matching Results</h2>"
    results_html_reverse = "<h2>Reverse Matching Results</h2>"

    for similarity, file_path in ranked_results[:5]:
        class_name = file_path.split("/")[-2].replace("kor_", "")
        file_name = file_path.split("/")[-1]
        results_html += (
            f"<p><strong>{class_name} - {file_name}:</strong> {similarity:.4f}</p>"
        )
        gallery.append(file_path)
        top_results.append(f"{class_name} - {file_name} : {similarity:.4f}")

    for similarity, file_path in ranked_results[::-1][:5]:
        results_html_reverse += f"<p><strong>{file_path.split('/')[-2].replace('kor_', '')} - {file_path.split('/')[-1]}:</strong> {similarity:.4f}</p>"

    return gallery, top_results, results_html, results_html_reverse


def on_click(image_input, alpha, checkbox, index):
    """버튼 클릭 시 워크플로우 수행"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_time = time.time()

    print(f"[INFO] Workflow Started at {timestamp}")

    # 분류 수행
    if checkbox:
        predictions = classify(image_input)
        classification_html = update_classification_ui(predictions)
        print(f"[INFO] Classification Completed")
    else:
        classification_html = ""
        print(f"[INFO] Classification Skipped")

    # 캡션 및 임베딩 생성
    if alpha == 0:
        caption_text, caption_html = update_caption_ui(image_input, index)
        text_embedding = generate_text_embedding(caption_text)
        image_embedding = None
    elif alpha == 1:
        caption_text, caption_html = "", ""
        image_embedding = generate_image_embedding(image_input)
        text_embedding = None
    else:
        caption_text, caption_html = update_caption_ui(image_input, index)
        image_embedding = generate_image_embedding(image_input)
        text_embedding = generate_text_embedding(caption_text)

    print("[INFO] Embeddings Generated")

    # Elasticsearch에서 매칭 데이터 가져오기
    if checkbox:
        matching_data = fetch_matching_data(
            alpha=alpha, predictions=predictions, index=index
        )
    else:
        matching_data = fetch_matching_data(alpha=alpha, index=index)

    print(f"[INFO] Matching Data Length: {len(matching_data)}")

    # 매칭 데이터와 코사인 유사도 계산
    ranked_results = rank_matches(image_embedding, text_embedding, matching_data, alpha)

    # 수행 시간 계산
    end_time = time.time()
    execution_time = end_time - start_time
    time_html = (
        f"<h2>Time Taken</h2> <p><strong>{execution_time:.4f} seconds</strong></p>"
    )

    # 결과 출력 HTML 생성
    (
        gallery,
        top_results,
        results_html,
        results_html_reverse,
    ) = update_gallery_ui(ranked_results)

    # Notion에 결과 저장
    response = push_to_notion(
        image_input,
        alpha,
        top_results,
        execution_time,
        checkbox,
        caption_text,
        classification_html,
        index,
        timestamp,
    )
    if response.status_code == 200:
        print("[INFO] Notion Push Success")
    else:
        print("[Warning] Notion Push Failed")
        print(response.json())

    print("[INFO] Workflow Completed\n")

    return (
        classification_html,
        caption_html,
        results_html,
        results_html_reverse,
        gallery,
        time_html,
    )


# Gradio 인터페이스 생성
with gr.Blocks() as demo:
    with gr.Row():

        with gr.Column():
            image_input = gr.Image(type="filepath")
            gr.Markdown(
                "### The closer it gets to 0, the more important text similarity becomes."
            )
            gr.Markdown(
                "### And the closer it gets to 1, the more important image similarity becomes."
            )
            with gr.Row():
                checkbox = gr.Checkbox(
                    label="Set Classifier?", value=True, show_label=False
                )
                index_input = gr.Dropdown(
                    choices=["embeddings", "embeddings_florence"],
                    value="embeddings",
                    show_label=False,
                )
            alpha_slider = gr.Slider(
                0,
                1,
                value=0.5,
                step=0.1,
                label="Alpha (Image-Text Weight)",
            )
            predict_button = gr.Button("Run Workflow")

        with gr.Row():
            gr.Markdown()
            with gr.Column(scale=3):
                classification_output = gr.HTML()
                caption_output = gr.HTML()
                matching_results_output = gr.HTML()
                matching_results_output_reverse = gr.HTML()
                time_output = gr.HTML()
            gr.Markdown()

    with gr.Column():
        gallery = gr.Gallery(
            interactive=False,
            columns=5,
            height=300,
        )

    predict_button.click(
        fn=on_click,
        inputs=[image_input, alpha_slider, checkbox, index_input],
        outputs=[
            classification_output,
            caption_output,
            matching_results_output,
            matching_results_output_reverse,
            gallery,
            time_output,
        ],
    )

demo.launch(share=True)
