from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 🔥 Allow frontend requests (IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for now allow all (later restrict)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Input format (JSON body)
class TextInput(BaseModel):
    text: str

# 🔥 Load model ONCE at startup
@app.on_event("startup")
def load_model():
    global classifier

    classifier = pipeline(
        "text-classification",
        model="pranaymaheshwaram/Hindi-News-Classification",
        model_kwargs={"low_cpu_mem_usage": True}
    )

# 🧠 Label mapping
labels = {
    "LABEL_0": "India",
    "LABEL_1": "International",
    "LABEL_2": "Entertainment",
    "LABEL_3": "Sport",
    "LABEL_4": "News",
    "LABEL_5": "Science and Technology"
}

# ✅ Health check
@app.get("/")
def home():
    return {"message": "API is running 🚀"}

# 🚀 Prediction endpoint
@app.post("/predict")
def predict(input: TextInput):
    try:
        result = classifier(input.text)

        label = result[0]['label']
        score = result[0]['score']

        return {
            "prediction": labels.get(label, label),
            "confidence": float(score)
        }

    except Exception as e:
        return {"error": str(e)}