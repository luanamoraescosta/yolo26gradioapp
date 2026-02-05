import gradio as gr
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from models import load_models
from inference import run_inference
from utils import create_heatmap, add_label
from plots import plot_counter

model_normal, model_open, device = load_models()

def build_heatmaps(heatmaps):
    return [add_label(create_heatmap(mask), label) for label, mask in heatmaps.items()]

def detect_normal(files, conf):
    if not files:
        return [], None, []
    imgs, counter, heatmaps = run_inference(model_normal, files, conf)
    return imgs, plot_counter(counter), build_heatmaps(heatmaps)

def detect_open(files, labels_text, conf):
    if not files or not labels_text.strip():
        return [], None, []
    labels = [l.strip() for l in labels_text.split(",") if l.strip()]
    open_path = hf_hub_download(
        repo_id="openvision/yoloe26-x-seg",
        filename="model.pt"
    )
    model_open = YOLO(open_path).to(device)
    imgs, counter, heatmaps = run_inference(model_open, files, conf, labels)
    return imgs, plot_counter(counter), build_heatmaps(heatmaps)

with gr.Blocks() as demo:
    gr.Markdown("# 🔍 YOLO26 / YOLOE26 – Segmentation, Count & Heatmaps")

    with gr.Tabs():
        with gr.Tab("Normal"):
            with gr.Row():
                with gr.Column(scale=1):
                    files_n = gr.File(file_types=["image"], file_count="multiple")
                    conf_n = gr.Slider(0.1, 1.0, 0.25, label="Confidence")
                    btn_n = gr.Button("Run")
                with gr.Column(scale=1):
                    gallery_n = gr.Gallery(columns=3, label="Segmented Images")

            with gr.Row():
                plot_n = gr.Plot()
                heat_n = gr.Gallery(columns=3, label="Heatmaps")

            btn_n.click(
                detect_normal,
                [files_n, conf_n],
                [gallery_n, plot_n, heat_n]
            )

        with gr.Tab("Open Vocabulary"):
            with gr.Row():
                with gr.Column(scale=1):
                    files_o = gr.File(file_types=["image"], file_count="multiple")
                    labels_o = gr.Textbox(value="person, car, bus", label="Objects")
                    conf_o = gr.Slider(0.1, 1.0, 0.25)
                    btn_o = gr.Button("Run")
                with gr.Column(scale=1):
                    gallery_o = gr.Gallery(columns=3, label="Segmented Images")

            with gr.Row():
                plot_o = gr.Plot()
                heat_o = gr.Gallery(columns=3, label="Heatmaps")

            btn_o.click(
                detect_open,
                [files_o, labels_o, conf_o],
                [gallery_o, plot_o, heat_o]
            )

demo.launch()
