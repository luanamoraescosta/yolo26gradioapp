import gradio as gr
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import tempfile
import os
import cv2
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

# =====================================================
# Load models
# =====================================================
def load_models():
    normal_path = hf_hub_download(
        repo_id="openvision/yolo26-l-seg",
        filename="model.pt"
    )
    open_path = hf_hub_download(
        repo_id="openvision/yoloe26-l-seg",
        filename="model.pt"
    )

    model_normal = YOLO(normal_path)
    model_open = YOLO(open_path)
    return model_normal, model_open


model_normal, model_open = load_models()

# =====================================================
# Helper functions
# =====================================================
def run_inference(model, files, conf, labels=None):
    object_counter = Counter()
    outputs = []

    if labels:
        model.set_classes(labels, model.get_text_pe(labels))

    for file in files:
        img = Image.open(file.name).convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            path = tmp.name

        results = model.predict(path, conf=conf)
        result = results[0]

        # Conta objetos
        if result.boxes is not None:
            for cls in result.boxes.cls:
                label = result.names[int(cls)]
                object_counter[label] += 1

        # Plot annotado
        annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
        outputs.append(annotated)

        os.remove(path)

    return outputs, object_counter


def plot_counter(counter: Counter):
    if not counter:
        return None
    df = pd.DataFrame(counter.items(), columns=["Object", "Count"])
    df = df.sort_values("Count", ascending=False)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(df["Object"], df["Count"], color="skyblue")
    ax.set_xlabel("Object")
    ax.set_ylabel("Count")
    ax.set_title("Object frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# =====================================================
# Detection functions
# =====================================================
def detect_normal(files, conf):
    if not files:
        return [], None
    images, counter = run_inference(model_normal, files, conf)
    fig = plot_counter(counter)
    return images, fig


def detect_open(files, labels_text, conf):
    if not files:
        return [], None
    labels = [l.strip() for l in labels_text.split(",") if l.strip()]
    if not labels:
        return [], None
    images, counter = run_inference(model_open, files, conf, labels)
    fig = plot_counter(counter)
    return images, fig

# =====================================================
# Gradio UI
# =====================================================
with gr.Blocks() as demo:
    gr.Markdown("# 🔍 YOLO26 / YOLOE-26 Detector with Object Count")

    with gr.Tabs():
        # Normal detection
        with gr.Tab("Normal"):
            files_n = gr.File(file_types=["image"], file_count="multiple")
            conf_n = gr.Slider(0.1, 1.0, 0.25)
            btn_n = gr.Button("Run")
            gallery_n = gr.Gallery(columns=3)
            plot_n = gr.Plot()

            btn_n.click(detect_normal, [files_n, conf_n], [gallery_n, plot_n])

        # Open-vocabulary detection
        with gr.Tab("Open-Vocabulary"):
            files_o = gr.File(file_types=["image"], file_count="multiple")
            labels_o = gr.Textbox(value="person, dog, bus")
            conf_o = gr.Slider(0.1, 1.0, 0.25)
            btn_o = gr.Button("Run")
            gallery_o = gr.Gallery(columns=3)
            plot_o = gr.Plot()

            btn_o.click(detect_open, [files_o, labels_o, conf_o], [gallery_o, plot_o])

demo.launch()
