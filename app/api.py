from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from app.schemas import PredictionResponse, ReportRequest
from genai.workflow import KidneyReportWorkflow
from ml.predictor import class_names, predict_image


app = FastAPI(
    title="Kidney Disease Detection API",
    description="CNN image prediction with LangChain prompts and LangGraph-style GenAI reporting.",
    version="1.0.0",
)

workflow = KidneyReportWorkflow()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "classes": class_names,
        "features": ["keras_prediction", "fastapi", "langchain_prompting", "langgraph_workflow"],
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> dict:
    img = await _read_upload_image(file)
    try:
        prediction = predict_image(img)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return _prediction_response(prediction)


@app.post("/report")
def report(request: ReportRequest) -> dict:
    return workflow.run(request.model_dump())


@app.post("/predict-report")
async def predict_report(file: UploadFile = File(...)) -> dict:
    img = await _read_upload_image(file)
    try:
        prediction = predict_image(img)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "prediction": _prediction_response(prediction),
        "report": workflow.run(prediction),
    }


def _prediction_response(prediction: dict) -> dict:
    return {
        "diagnosis": prediction["diagnosis"],
        "confidence": prediction["confidence"],
        "probabilities": prediction["probabilities"],
        "image_size": prediction["image_size"],
    }


async def _read_upload_image(file: UploadFile) -> Image.Image:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    try:
        content = await file.read()
        return Image.open(BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
