import numpy as np
import cv2
from PIL import Image
import matplotlib

def resize_image(img, max_size=640):
    w, h = img.size
    scale = max_size / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)))
    return img

def create_heatmap(mask):
    """
    Heatmap “hot” com gradiente laranja → vermelho → branco
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
