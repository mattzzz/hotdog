import io
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cnnmodels import ResNN

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model" / "model.pt"
CLASS_PATH = APP_DIR / "model" / "class_names.json"

IMG_SIZE = 192

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

model = ResNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

class_names = json.loads(CLASS_PATH.read_text(encoding="utf-8"))
class0, class1 = class_names[0], class_names[1]

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    return tensor.to(device)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((APP_DIR / "static" / "index.html").read_text(encoding="utf-8"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        return JSONResponse(status_code=400, content={"error": "Unsupported file type"})

    image_bytes = await file.read()
    x = preprocess_image(image_bytes)

    with torch.no_grad():
        logits = model(x)
        prob_class1 = torch.sigmoid(logits).item()
        print(f"logit={logits.item():.3f}  sigmoid={prob_class1:.3f}")

    pred_class = class1 if prob_class1 >= 0.5 else class0
    score = prob_class1 if pred_class == class1 else (1.0 - prob_class1)

    is_hotdog = pred_class.lower() == "hotdog"

    return {
        "pred_class": pred_class,
        "is_hotdog": is_hotdog,
        "display_label": "HOTDOG ✅" if is_hotdog else "NOT HOTDOG ❌",
        "score": score,
    }