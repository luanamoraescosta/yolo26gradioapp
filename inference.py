import os
import cv2
import tempfile
import numpy as np
from collections import Counter
from PIL import Image
from logger import get_logger
from utils import resize_image

logger = get_logger(__name__)

def run_inference(model, files, conf, labels=None):
    outputs = []
    heatmaps = {}
    counter = Counter()

    if labels:
        pe = model.get_text_pe(labels)
        model.set_classes(labels, pe)

    for file in files:
        logger.info(f"Processing image: {file.name}")
        img = Image.open(file.name).convert("RGB")
        img = resize_image(img)
        img_np = np.array(img)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            path = tmp.name

        result = model.predict(path, conf=conf, verbose=False)[0]

        # contador
        if result.boxes is not None:
            for cls in result.boxes.cls:
                label = result.names[int(cls)]
                counter[label] += 1

        # heatmaps
        if result.masks is not None:
            for cls, mask in zip(result.boxes.cls, result.masks.data):
                label = result.names[int(cls)]
                mask_np = mask.cpu().numpy()
                mask_np = cv2.resize(mask_np, (img_np.shape[1], img_np.shape[0]))
                heatmaps[label] = heatmaps.get(label, 0) + mask_np

        annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
        outputs.append(annotated)
        os.remove(path)

    return outputs, counter, heatmaps
