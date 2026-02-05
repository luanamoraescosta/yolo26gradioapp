import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from logger import get_logger

logger = get_logger(__name__)

def get_device():
    if torch.cuda.is_available():
        logger.info("CUDA available — using GPU")
        return "cuda"
    else:
        logger.info("CUDA not available — using CPU")
        return "cpu"

def load_models():
    device = get_device()

    normal_path = hf_hub_download(
        repo_id="openvision/yolo26-x-seg",
        filename="model.pt"
    )
    open_path = hf_hub_download(
        repo_id="openvision/yoloe26-x-seg",
        filename="model.pt"
    )

    model_normal = YOLO(normal_path).to(device)
    model_open = YOLO(open_path).to(device)

    logger.info("Models loaded")
    return model_normal, model_open, device
