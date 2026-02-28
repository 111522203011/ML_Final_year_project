from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from scipy.sparse import hstack
import uvicorn
from nlp_analyzer import analyze_reviews

app = FastAPI()

# ✅ Enable CORS (like @CrossOrigin in Spring Boot)
origins = [
    "*",  # For testing (allow all)
    # "http://localhost:3000",
    # "https://your-frontend-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder (if needed)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Load model AND vectorizer
model = joblib.load("fake_review_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


class ReviewInput(BaseModel):
    account_age_days: int
    rating: int
    review_velocity: float
    is_verified: int
    review_text: str


@app.get("/")
def home():
    return {"message": "Fake Review Detection API is running"}


@app.post("/predict")
def predict_review(data: ReviewInput):

    # 1️⃣ Convert text to TF-IDF features
    text_features = tfidf.transform([data.review_text])

    # 2️⃣ Prepare numeric features
    numeric_features = np.array([[
        data.account_age_days,
        data.review_velocity,
        data.is_verified,
        data.rating
    ]])

    # 3️⃣ Combine text + numeric features
    final_features = hstack([text_features, numeric_features])

    # 4️⃣ Make prediction
    prediction = model.predict(final_features)[0]
    confidence = model.predict_proba(final_features).max()

    return {
        "prediction": "Fake Review" if prediction == 1 else "Genuine Review",
        "confidence": f"{confidence * 100:.2f}%"
    }


@app.post("/analyze-feedback")
def analyze_feedback(reviews: list[str]):
    return analyze_reviews(reviews)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)