# TB Detection AI Service

A FastAPI-based AI service for Tuberculosis detection from chest X-ray DICOM files using the XraySigLIP vision model.

---

## Project Structure

```
project/
├── main.py                          ← FastAPI entry point
├── api.py                           ← endpoint logic
├── model.py                         ← CheXagentSigLIPBinary neural network
├── utils.py                         ← DICOM to image converter
└── best_model_attention_loss.pth    ← trained model checkpoint (on your drive)
```

---

## What Each File Does

### `main.py`
Entry point for the FastAPI server. Registers the router from `api.py`.
```python
from fastapi import FastAPI
from api import router

app = FastAPI()
app.include_router(router)
```

---

### `api.py`
Core service file. Does the following at startup and per request:

**At server startup (runs once):**
- Loads the XraySigLIP processor from HuggingFace
- Loads the vision encoder (XraySigLIP backbone)
- Loads your trained checkpoint `best_model_attention_loss.pth`
- Keeps the model in GPU/CPU memory for all incoming requests

**Per request:**
- Receives uploaded ZIP file
- Extracts the DICOM `.dcm` file from the ZIP
- Converts DICOM to PNG
- Preprocesses the image (handles 16-bit, converts to RGB, runs through processor)
- Runs inference through the model
- Returns prediction as JSON

---

### `model.py`
Defines `CheXagentSigLIPBinary` — a binary classification model built on top of the XraySigLIP vision encoder.

- Takes the vision encoder as input
- Adds a classifier head: `Linear(hidden_size → 256) → ReLU → Dropout → Linear(256 → 1)`
- Registers a forward hook to capture pooling attention weights
- `forward()` returns: `logits, attention, pooling_attn_weights`

---

### `utils.py`
Contains `dicom_to_image()` which:
- Reads a `.dcm` DICOM file using `pydicom`
- Extracts the raw pixel array
- Normalizes pixel values to 0–255 range (min-max scaling)
- Inverts image if `PhotometricInterpretation == MONOCHROME1`
- Saves as PNG or JPG

---

## Installation

### Requirements
```bash
pip install fastapi uvicorn torch transformers pillow pydicom opencv-python
```

### Verify checkpoint path
Your checkpoint is stored at:
```
/media/omen/392571a8-91fe-4ff9-aac6-79d0409b1b3f/home/omen/Documents/Megha/tb_work/training_with_attention_loss/best_model_attention_loss.pth
```
Make sure the drive is **mounted** before starting the server. Verify with:
```bash
ls "/media/omen/392571a8-91fe-4ff9-aac6-79d0409b1b3f/home/omen/Documents/Megha/tb_work/training_with_attention_loss/best_model_attention_loss.pth"
```

---

## Running the Server

### Step 1 — Navigate to project folder
```bash
cd /path/to/your/project
```

### Step 2 — Start the server
```bash
uvicorn main:app --reload
```

### Step 3 — Confirm startup logs
```
[startup] Loading XraySigLIP processor ...
[startup] Loading vision encoder ...
[startup] Loading CheXagentSigLIPBinary checkpoint ...
[startup] Model ready on cuda
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

## Using the Service

### Option 1 — Browser UI (easiest)
1. Open `http://127.0.0.1:8000/docs`
2. Click `POST /predictdiseasev2/`
3. Click **Try it out**
4. Click **Choose File** → select your ZIP file
5. Click **Execute**
6. See result in the **Response body**

### Option 2 — Postman
- Method: `POST`
- URL: `http://127.0.0.1:8000/predictdiseasev2/`
- Body: `form-data` → key: `file`, value: your ZIP file

### Option 3 — curl
```bash
curl -X POST "http://127.0.0.1:8000/predictdiseasev2/" \
     -F "file=@/path/to/your/dicom.zip"
```

---

## Input Format

| Field | Type   | Description                              |
|-------|--------|------------------------------------------|
| file  | ZIP    | ZIP file containing one DICOM (.dcm) file |

**ZIP structure example:**
```
patient001.zip
└── patient001/
    └── scan.dcm
```

---

## Output Format

```json
{
  "finding":     "TB positive",
  "label":       1,
  "probability": 0.8732
}
```

| Field       | Type   | Values                        |
|-------------|--------|-------------------------------|
| finding     | string | `"Normal"` or `"TB positive"` |
| label       | int    | `0` = Normal, `1` = TB positive |
| probability | float  | `0.0` to `1.0`                |

---

## Internal Workflow

```
User uploads ZIP
        │
        ▼
Extract .dcm from ZIP
        │
        ▼
dicom_to_image()          ← utils.py
  - pydicom reads .dcm
  - normalize pixel array
  - save as PNG
        │
        ▼
preprocess_image()        ← api.py
  - handle 16-bit (I;16)
  - convert to RGB
  - XraySigLIP processor
  - tensor shape: (1, 3, 384, 384)
        │
        ▼
CheXagentSigLIPBinary     ← model.py
  - vision encoder forward pass
  - pooler_output → classifier head
  - returns logits, attention, pooling_attn_weights
        │
        ▼
sigmoid(logits)
  - probability = sigmoid(logit)
  - pred = 1 if probability >= 0.5 else 0
        │
        ▼
JSON response
  { finding, label, probability }
```

---

## Configuration (in `api.py`)

| Variable         | Value                                              | Description                  |
|------------------|----------------------------------------------------|------------------------------|
| `MODEL_NAME`     | `StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli` | HuggingFace model name    |
| `CHECKPOINT`     | absolute path to `.pth` file                       | Your trained weights         |
| `DICOM_TEMP_PATH`| `/tmp/dicom_uploads`                               | Temp folder for uploads      |
| `device`         | `cuda` if GPU available, else `cpu`                | Inference device             |
| `dtype`          | `torch.float32`                                    | Model precision              |

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError` on checkpoint | Drive not mounted | Mount the drive before starting server |
| `No .dcm file found` | ZIP structure wrong | Ensure ZIP contains a `.dcm` file |
| `CUDA out of memory` | GPU memory full | Set `device = "cpu"` in `api.py` |
| `key model_state not found` | Wrong checkpoint key | Print `torch.load(CHECKPOINT).keys()` and update key name in `api.py` |
| Server not starting | Port in use | Run `uvicorn main:app --port 8001` |

---

## Stopping the Server

Press `Ctrl + C` in the terminal.
