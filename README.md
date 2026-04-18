# BBC Hindi News Classification

A multi-approach project for classifying Hindi BBC news articles into topics using traditional machine-learning and transformer-based methods.

---

## 📌 Overview

This project tackles the task of **automatic topic classification of Hindi news articles**. Given a Hindi news article, the goal is to predict which topic it belongs to (e.g., India, International, Sports, Entertainment, Science).

We explored **three different approaches**, ranging from classical ML to state-of-the-art transformer models.

---

## 📊 Dataset

**IndicGLUE – BBC Hindi (bbca.hi)**  
Source: https://huggingface.co/datasets/ai4bharat/indic_glue/viewer/bbca.hi

| Split | Samples |
|-------|---------|
| Train | 3,470   |
| Test  | 886     |
| **Total** | **4,356** |

- Dataset contains Hindi news articles labeled by topic  
- Significant **class imbalance** present  

---

## ⚙️ Solution Approaches

### 🔹 Approach 1 – TF-IDF + Classical ML (Pranay)

**Key Steps:**
- Handled class imbalance via **sampling**
- Selected **6 major classes** and down-sampled dominant ones:

| Class | Samples Used |
|-------|-------------|
| India | 400 |
| International | 400 |
| Entertainment | 285 |
| Sport | 258 |
| News | 230 |
| Science | 194 |

- Feature Extraction: **TF-IDF (Unigrams + Bigrams)**
- Models Used:
  - Logistic Regression  
  - Support Vector Machine (SVM)

**Results:**

| Model | Accuracy |
|------|----------|
| Logistic Regression | 73.6% |
| SVM | **75.3%** |

---

### 🔹 Approach 2 – FastText, Autoencoder & XLM-RoBERTa (Bharani)

**Key Steps:**
- Used **full dataset** with **class weighting**
- Preprocessing:
  - Text cleaning  
  - Tokenization  
  - Stopword removal  

**Experiments:**
- TF-IDF → SVM  
- FastText → Logistic Regression / SVM  
- Autoencoder (PyTorch) → Neural Network  
- **XLM-RoBERTa (fine-tuned)**:
  - max_length = 384  
  - 6 epochs  
  - weighted cross-entropy loss  

**Results:**

| Method | Model | Accuracy |
|--------|-------|----------|
| TF-IDF | SVM | 72.51% |
| FastText | Logistic Regression | 53.34% |
| FastText | SVM | 45.03% |
| FastText + Autoencoder | Neural Network | 24.45% |
| XLM-RoBERTa | Fine-tuned | **76.55%** |

---

### 🔹 Approach 3 – IndicBERT Embeddings (P. Sai Teja)

**Key Steps:**
- Used **IndicBERT v2 pretrained model**
- Generated **sentence embeddings (mean pooling)**
- Applied feature scaling
- Model: Logistic Regression

**Results:**

| Model | Accuracy |
|-------|----------|
| IndicBERT + Logistic Regression | **71.59%** |

---

## 📈 Results Summary

| Approach | Method | Model | Accuracy |
|----------|--------|-------|----------|
| 1 | TF-IDF | Logistic Regression | 73.6% |
| 1 | TF-IDF | SVM | 75.3% |
| 2 | TF-IDF | SVM | 72.51% |
| 2 | FastText | Logistic Regression | 53.34% |
| 2 | FastText | SVM | 45.03% |
| 2 | FastText + Autoencoder | Neural Network | 24.45% |
| 2 | XLM-RoBERTa | Fine-tuned | **76.55%** |
| 3 | IndicBERT | Logistic Regression | 71.59% |

> 🏆 **Best Model: XLM-RoBERTa (Fine-tuned) – 76.55% Accuracy**

---

## 🧠 Conclusion

- **TF-IDF + SVM** provides a strong baseline (~75%)
- **FastText** captures semantics but underperforms in this setup
- **IndicBERT** works well without fine-tuning (~71.6%)
- **XLM-RoBERTa (fine-tuned)** performs best due to deep contextual understanding

👉 Clear trade-off:
- Simpler models → faster, cheaper  
- Transformers → better accuracy, higher compute  

---

## 🌐 Live Deployment

🎉 **The XLM-RoBERTa model is now live!**

**Try it here:** [https://maheshwarampranay.github.io/BBC-Hindi-News-Classification/](https://maheshwarampranay.github.io/BBC-Hindi-News-Classification/)

### Deployment Architecture

- **Model Hosting:** [Hugging Face Spaces](https://huggingface.co/spaces/pranaymaheshwaram/HINDI-NEWS-CLASSIFIER)
- **UI Hosting:** GitHub Pages  
- **Communication:** RESTful API (POST requests)

### API Endpoint

```
POST https://pranaymaheshwaram-hindi-news-classifier.hf.space/predict

Request Body:
{
  "text": "your Hindi news text here"
}

Response:
{
  "prediction": "category_name",
  "confidence": 0.85
}
```

### Performance Notes

- **First request:** ~20–60 sec (model loading / cold start)
- **Subsequent requests:** ~0.5–2 sec  
- **Max input length:** 512 tokens (long inputs are truncated)

### Local Setup (Optional)

To run the model locally:

```bash
pip install transformers torch sentencepiece fastapi uvicorn

uvicorn app:app --reload
```

---

## 🚀 Future Work

- Data augmentation (back-translation, paraphrasing)
- Better hyperparameter tuning for transformers
- Ensemble methods (ML + Transformers)
- Cross-lingual transfer learning across Indic languages

---
