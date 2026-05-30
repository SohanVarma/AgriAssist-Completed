from pathlib import Path
import sys
import shutil
from importlib import import_module

from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
WEB_DIR = BASE_DIR / "web"
UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)

sys.path.append(str(SRC_DIR))

rag_module = import_module("09_full_rag_pipeline")
predict_module = import_module("13_predict_uploaded_image")

AgriculturalRAG = rag_module.AgriculturalRAG
KB_PATH = rag_module.KB_PATH
predict_image = predict_module.predict_image

app = FastAPI(title="AgriAssist AI Advisory")

app.mount(
    "/static",
    StaticFiles(directory=str(WEB_DIR / "static")),
    name="static"
)

templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))

rag = AgriculturalRAG(KB_PATH)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "result": None,
            "query": "",
            "error": None,
            "prediction": None,
        },
    )


@app.post("/advisory", response_class=HTMLResponse)
def advisory(request: Request, query: str = Form(...)):
    query = query.strip()

    if not query:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "result": None,
                "query": "",
                "error": "Please enter an agricultural query.",
                "prediction": None,
            },
        )

    try:
        result = rag.generate_advisory(query, top_k=5)

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "result": result,
                "query": query,
                "error": None,
                "prediction": None,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "result": None,
                "query": query,
                "error": f"Error while generating advisory: {str(e)}",
                "prediction": None,
            },
        )


@app.post("/predict-image", response_class=HTMLResponse)
def predict_uploaded_image(request: Request, image: UploadFile = File(...)):
    try:
        if not image.filename:
            raise ValueError("No image uploaded.")

        file_extension = Path(image.filename).suffix.lower()

        if file_extension not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            raise ValueError("Please upload a valid image file: jpg, jpeg, png, webp, or bmp.")

        saved_path = UPLOAD_DIR / image.filename

        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        predicted_label, confidence = predict_image(saved_path)

        query = f"{predicted_label} disease treatment management"

        result = rag.generate_advisory(query, top_k=5)

        prediction = {
            "label": predicted_label,
            "confidence": confidence,
            "image_name": image.filename,
        }

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "result": result,
                "query": query,
                "error": None,
                "prediction": prediction,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "result": None,
                "query": "",
                "error": f"Image prediction error: {str(e)}",
                "prediction": None,
            },
        )
