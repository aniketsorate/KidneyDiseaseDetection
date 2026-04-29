# Kidney Disease Detection

FastAPI-based kidney CT image classification project with a TensorFlow model,
GenAI reporting, and LangChain/LangGraph workflow support.

Training files are kept separately in `training/`.

## Run FastAPI

```bash
uvicorn main:app --reload
```

Open API docs at `http://127.0.0.1:8000/docs`.

## Run Streamlit Demo

```bash
streamlit run streamlit_app.py
```

## Current Structure

```text
main.py                  FastAPI entry point
app/api.py               API routes
app/schemas.py           API request/response schemas
ml/predictor.py          TensorFlow model loading and prediction
genai/prompts.py         System and user prompt templates
genai/prompt_service.py  LangChain/OpenRouter prompt execution
genai/workflow.py        LangGraph workflow orchestration
training/                Kaggle GPU training files
artifacts/               Trained model files
```
