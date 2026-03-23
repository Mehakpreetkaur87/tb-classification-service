import os
import glob
import tempfile
import torch
from zipfile import ZipFile
from PIL import Image
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoModel, AutoProcessor, AutoConfig

import json
import shutil 
from src.configuration.config import DICOM_TEMP_PATH, outputDir

from model.model import CheXagentSigLIPBinary          
from utils.utils import dicom_to_image                  

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODEL_NAME    = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"
CHECKPOINT    = "/media/omen/392571a8-91fe-4ff9-aac6-79d0409b1b3f/home/omen/Documents/Megha/tb_work/training_with_attention_loss/best_model_attention_loss.pth"
DICOM_TEMP_PATH = "/tmp/dicom_uploads"           # change to your actual temp path
dtype         = torch.float32
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(DICOM_TEMP_PATH, exist_ok=True)

# ──────────────────────────────────────────────
# Load model ONCE at module import (not per request)
# ──────────────────────────────────────────────
print("[startup] Loading XraySigLIP processor ...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

print("[startup] Loading vision encoder ...")
_config      = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
_vision_full = AutoModel.from_pretrained(
    MODEL_NAME, config=_config, trust_remote_code=True
).to(device, dtype)
vision_encoder = _vision_full.vision_model
del _vision_full                                  # free the text tower

print("[startup] Loading CheXagentSigLIPBinary checkpoint ...")
_model = CheXagentSigLIPBinary(vision_encoder=vision_encoder)
_ckpt  = torch.load(CHECKPOINT, map_location=device)
_model.load_state_dict(_ckpt["model_state"])
_model.to(device).eval()
print("[startup] Model ready.")

# ──────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────
router = APIRouter()


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Open a PNG/JPG chest X-ray, handle 16-bit DICOM artifacts,
    convert to RGB, run through XraySigLIP processor.

    Returns pixel_values of shape (1, C, H, W) on the correct device.
    """
    image = Image.open(image_path)

    # ── Handle 16-bit DICOM grayscale (mode "I;16") ──────────────────────
    # if image.mode == "I;16":
    #     image = image.convert("I")                              # 32-bit int
    #     image = image.point(lambda p: p * (255.0 / 65535.0))   # normalise
    #     image = image.convert("L")                              # 8-bit gray

    # ── Convert any mode (L, RGBA, P …) to RGB ───────────────────────────
    image = image.convert("RGB")

    # ── XraySigLIP processor ─────────────────────────────────────────────
    inputs       = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]            # (1, C, H, W)
    pixel_values = pixel_values.to(device, dtype)

    return pixel_values


@router.post("/predictdiseasev2/")
async def predict_disease_v2(file: UploadFile = File(...)):
    """
    Upload a ZIP containing one or more DICOM files.
    Returns TB prediction with label, finding string, and probability.
    """
    temp_dir = tempfile.mkdtemp(dir=DICOM_TEMP_PATH)

    try:
        # ── 1. Save uploaded ZIP ─────────────────────────────────────────
        temp_file = os.path.join(temp_dir, file.filename)
        with open(temp_file, "wb") as out_file:
            out_file.write(await file.read())

        # ── 2. Extract ZIP ───────────────────────────────────────────────
        with ZipFile(temp_file, "r") as zip_ref:
            root_dir = zip_ref.namelist()[0].split("/")[0]
            zip_ref.extractall(temp_dir)

        root_dir_path = os.path.join(temp_dir, root_dir)

        # ── 3. Find first DICOM file ─────────────────────────────────────
        file_paths = glob.glob(root_dir_path + "/**/*.dcm", recursive=True)
        file_paths += glob.glob(os.path.join(temp_dir, '**', '*.dicom'), recursive=True)
        if not file_paths:
            return JSONResponse(
                status_code=400,
                content={"error": "No .dcm file found inside the ZIP."}
            )

        dicom_path  = file_paths[0]
        print("DICOM FILE:", dicom_path)
        
        output_path = os.path.splitext(dicom_path)[0] + ".png"
        # ── 4. DICOM → PNG ───────────────────────────────────────────────
        dicom_to_image(dicom_path, output_path, format="png")

        # ── 5. Preprocess image ──────────────────────────────────────────
        pixel_values = preprocess_image(output_path)   # (1, C, H, W)

        # ── 6. Inference ─────────────────────────────────────────────────
        with torch.no_grad():
            logits, attention, pooling_attn_weights = _model(pixel_values)
            # logits shape: (1, 1)

        # ── 7. Post-process ──────────────────────────────────────────────
        prob    = torch.sigmoid(logits).squeeze().item()   # scalar float
        pred    = 1 if prob >= 0.5 else 0
        finding = "TB positive" if pred == 1 else "Normal"

        return JSONResponse({
            "finding":     finding,
            # "label":       pred,             # 0 = Normal, 1 = TB positive
            # "probability": round(prob, 4)
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

    finally:
        # ── Clean up temp files ──────────────────────────────────────────
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)