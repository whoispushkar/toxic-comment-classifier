from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import joblib


app = FastAPI(
    title="Toxic Comment Classifier",
    description="Multi-label toxic comment classification API",
    version="1.0"
)



BASE_DIR = Path(__file__).resolve().parent.parent

tfidf_vectorizer = joblib.load(
    BASE_DIR / "models" / "tfidf_vectorizer.pkl"
)

model_svc = joblib.load(
    BASE_DIR / "models" / "toxic_comment_model.pkl"
)

label_names = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


class CommentRequest(BaseModel):
    comment: str


@app.get("/")
def home():
    return {
        "message": "Toxic Comment Classifier API is running"
    }


@app.post("/predict")
def predict(request: CommentRequest):

    
    comment_tfidf = tfidf_vectorizer.transform(
        [request.comment]
    )

    
    prediction = model_svc.predict(
        comment_tfidf
    )[0]

    
    result = {
        label: bool(value)
        for label, value in zip(label_names, prediction)
    }
    is_toxic = any(result.values())

    return {
        "comment": request.comment,
        "is_toxic": is_toxic,
        "predictions": result
    }