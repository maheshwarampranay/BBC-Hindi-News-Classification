from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

classifier = pipeline(
    "text-classification",
    model="pranaymaheshwaram/Hindi-News-Classification"
)

labels = {
    "LABEL_0": "India",
    "LABEL_1": "International",
    "LABEL_2": "Entertainment",
    "LABEL_3": "Sport",
    "LABEL_4": "News",
    "LABEL_5": "Science and Technology"
}

@app.post("/predict")
def predict(text: str):
    result = classifier(text)

    label = result[0]['label']
    score = result[0]['score']

    return {
        "prediction": labels[label],
        "confidence": float(score)
    }