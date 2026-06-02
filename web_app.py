from pathlib import Path
import sys
import shutil
import uuid
from importlib import import_module

from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
WEB_DIR = BASE_DIR / "web"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = WEB_DIR / "static" / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(SRC_DIR))

rag_module = import_module("09_full_rag_pipeline")
predict_module = import_module("13_predict_uploaded_image")
gradcam_module = import_module("15_gradcam_uploaded_image")
# backbone_module = import_module("16_backbone_comparison_web")

AgriculturalRAG = rag_module.AgriculturalRAG
KB_PATH = rag_module.KB_PATH

predict_image = predict_module.predict_image
generate_gradcam_for_image = gradcam_module.generate_gradcam_for_image
# generate_backbone_comparison_report = backbone_module.generate_backbone_comparison_report

app = FastAPI(title="AgriAssist AI Advisory")

app.mount(
    "/static",
    StaticFiles(directory=str(WEB_DIR / "static")),
    name="static"
)

templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))

rag = AgriculturalRAG(KB_PATH)


def save_upload(upload: UploadFile) -> Path:
    if not upload.filename:
        raise ValueError("No image uploaded.")

    extension = Path(upload.filename).suffix.lower()

    if extension not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        raise ValueError("Please upload a valid image file: jpg, jpeg, png, webp, or bmp.")

    safe_name = f"{uuid.uuid4().hex}{extension}"
    saved_path = UPLOAD_DIR / safe_name

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)

    return saved_path


def render_page(
    request: Request,
    query: str = "",
    result=None,
    error=None,
    prediction=None,
    gradcam=None,
    # backbone=None,
):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "query": query,
            "result": result,
            "error": error,
            "prediction": prediction,
            "gradcam": gradcam,
            # "backbone": backbone,
        },
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return render_page(request)


@app.post("/advisory", response_class=HTMLResponse)
def advisory(
    request: Request,
    query: str = Form(...),
    top_k: int = Form(5),
):
    query = query.strip()

    if not query:
        return render_page(
            request,
            error="Please enter an agricultural query."
        )

    try:
        result = rag.generate_advisory(query, top_k=top_k)
        return render_page(
            request,
            query=query,
            result=result
        )

    except Exception as e:
        return render_page(
            request,
            query=query,
            error=f"Error while generating advisory: {str(e)}"
        )


@app.post("/predict-image", response_class=HTMLResponse)
def predict_uploaded_image(
    request: Request,
    image: UploadFile = File(...),
    generate_advisory: str = Form("yes"),
):
    try:
        saved_path = save_upload(image)

        predicted_label, confidence = predict_image(saved_path)

        copied_image_path = OUTPUT_DIR / saved_path.name
        shutil.copyfile(saved_path, copied_image_path)

        prediction = {
            "label": predicted_label,
            "confidence": confidence,
            "image_name": image.filename,
            "image_url": f"/static/outputs/{saved_path.name}",
        }

        result = None
        query = f"{predicted_label} disease treatment management"

        if generate_advisory == "yes":
            result = rag.generate_advisory(query, top_k=5)

        return render_page(
            request,
            query=query,
            result=result,
            prediction=prediction
        )

    except Exception as e:
        return render_page(
            request,
            error=f"Image prediction error: {str(e)}"
        )


@app.post("/gradcam", response_class=HTMLResponse)
def gradcam_explainability(
    request: Request,
    image: UploadFile = File(...),
):
    try:
        saved_path = save_upload(image)

        output_name = f"gradcam_{saved_path.stem}.png"
        output_path = OUTPUT_DIR / output_name

        gradcam_result = generate_gradcam_for_image(
            image_path=saved_path,
            output_path=output_path
        )

        original_output = OUTPUT_DIR / saved_path.name
        shutil.copyfile(saved_path, original_output)

        gradcam = {
            "image_name": image.filename,
            "label": gradcam_result["label"],
            "confidence": gradcam_result["confidence"],
            "image_url": f"/static/outputs/{saved_path.name}",
            "gradcam_url": f"/static/outputs/{output_name}",
            "explanation": (
                "The Grad-CAM heatmap highlights the visual regions that contributed most "
                "to the CNN model prediction. Brighter/redder regions indicate stronger "
                "influence on the predicted class."
            ),
        }

        return render_page(
            request,
            gradcam=gradcam
        )

    except Exception as e:
        return render_page(
            request,
            error=f"Grad-CAM error: {str(e)}"
        )


# @app.post("/backbone-comparison", response_class=HTMLResponse)
# def backbone_comparison(request: Request):
#     try:
#         comparison = generate_backbone_comparison_report()

#         backbone = {
#             "markdown": comparison["markdown"],
#             "json_path": comparison["json_path"],
#             "markdown_path": comparison["markdown_path"],
#         }

#         return render_page(
#             request,
#             backbone=backbone
#         )

#     except Exception as e:
#         return render_page(
#             request,
#             error=f"Backbone comparison error: {str(e)}"
#         )
