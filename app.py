import gradio as gr
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import tempfile
import os
import cv2
import numpy as np
from collections import Counter
import altair as alt
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib

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
    return YOLO(normal_path), YOLO(open_path)

model_normal, model_open = load_models()

# =====================================================
# Utils
# =====================================================
def resize_image(img, max_size=640):
    w, h = img.size
    scale = max_size / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)))
    return img

def create_heatmap(mask):
    """
    Heatmap “quente” com gradiente laranja → vermelho → branco.
    """
    mask = mask / (mask.max() + 1e-6)
    colormap = matplotlib.colormaps["hot"]  # Gradiente: preto → vermelho → amarelo → branco
    heat = colormap(mask)[..., :3]  # descarta alpha
    heat = (heat * 255).astype(np.uint8)
    return Image.fromarray(heat)

def add_label(img, label):
    w, h = img.size
    out = Image.new("RGB", (w, h + 22), (255, 255, 255))
    out.paste(img, (0, 0))

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    draw.text((6, h + 2), label, fill="black", font=font)

    return out

# =====================================================
# Inference
# =====================================================
def run_inference(model, files, conf, labels=None):
    outputs = []
    heatmaps = {}
    counter = Counter()

    if labels:
        pe = model.get_text_pe(labels)
        model.set_classes(labels, pe)

    for file in files:
        img = Image.open(file.name).convert("RGB")
        img = resize_image(img)
        img_np = np.array(img)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            path = tmp.name

        result = model.predict(path, conf=conf, verbose=False)[0]

        # Contagem
        if result.boxes is not None:
            for cls in result.boxes.cls:
                label = result.names[int(cls)]
                counter[label] += 1

        # Heatmaps
        if result.masks is not None:
            for cls, mask in zip(result.boxes.cls, result.masks.data):
                label = result.names[int(cls)]
                mask_np = mask.cpu().numpy()
                mask_np = cv2.resize(mask_np, (img_np.shape[1], img_np.shape[0]))

                if label in heatmaps:
                    existing_shape = heatmaps[label].shape
                    mask_np_resized = cv2.resize(mask_np, (existing_shape[1], existing_shape[0]))
                    heatmaps[label] += mask_np_resized
                else:
                    heatmaps[label] = mask_np

        annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
        outputs.append(annotated)

        os.remove(path)

    return outputs, counter, heatmaps

# =====================================================
# Altair
# =====================================================
def plot_counter(counter):
    if not counter:
        return None

    df = pd.DataFrame(counter.items(), columns=["Object", "Count"])
    chart = (
        alt.Chart(df)
        .mark_bar(color="#4DA3FF")
        .encode(
            x=alt.X("Object:N", sort="-y"),
            y="Count:Q"
        )
        .properties(width=350, height=300, title="Object count")
    )
    return chart

def build_heatmaps(heatmaps):
    imgs = []
    for label, mask in heatmaps.items():
        heat = create_heatmap(mask)
        heat = add_label(heat, label)
        imgs.append(heat)
    return imgs

# =====================================================
# UI callbacks
# =====================================================
def detect_normal(files, conf):
    if not files:
        return [], None, []
    imgs, counter, heatmaps = run_inference(model_normal, files, conf)
    return imgs, plot_counter(counter), build_heatmaps(heatmaps)

def detect_open(files, labels_text, conf):
    labels = [l.strip() for l in labels_text.split(",") if l.strip()]
    if not files or not labels:
        return [], None, []
    imgs, counter, heatmaps = run_inference(model_open, files, conf, labels)
    return imgs, plot_counter(counter), build_heatmaps(heatmaps)

# =====================================================
# Gradio UI
# =====================================================
with gr.Blocks() as demo:
    gr.Markdown("# 🔍 YOLO26 / YOLOE26 – Segmentation, Count & Heatmaps")

    with gr.Tabs():
        with gr.Tab("Normal"):
            with gr.Row():
                with gr.Column(scale=1):
                    files_n = gr.File(file_types=["image"], file_count="multiple")
                    conf_n = gr.Slider(0.1, 1.0, 0.25, label="Confidence", step=0.05, info="Confidence threshold", minimum=0.1, maximum=1.0)
                    btn_n = gr.Button("Run")
                with gr.Column(scale=1):
                    gallery_n = gr.Gallery(columns=3, label="Segmented images")

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
                    conf_o = gr.Slider(0.1, 1.0, 0.25, label="Confidence", step=0.05)
                    btn_o = gr.Button("Run")
                with gr.Column(scale=1):
                    gallery_o = gr.Gallery(columns=3, label="Segmented images")

            with gr.Row():
                plot_o = gr.Plot()
                heat_o = gr.Gallery(columns=3, label="Heatmaps")

            btn_o.click(
                detect_open,
                [files_o, labels_o, conf_o],
                [gallery_o, plot_o, heat_o]
            )

demo.launch()
