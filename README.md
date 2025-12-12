# 🧠 AI Resume Screening System  
A machine learning system that automatically classifies resumes as **Fit** or **Not Fit** for Data/ML roles using **TF-IDF + Logistic Regression**.

This project includes a full training pipeline, Colab notebook, dataset, model file, and a **live Gradio demo**.

---

## 🚀 Live Demo  
Paste a resume or summary and see the prediction:

👉 https://bc0655af6e7d532ec1.gradio.live

---

## ▶ Run This Project on Google Colab  

Click the badge to open and run the full notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Malaiyarasan/ai-resume-screening-system/blob/main/notebooks/resume_screening_ai.ipynb)

---

## 🧩 How It Works  
The system performs:

1. **Text Preprocessing**  
   - Lowercasing  
   - Stopword removal  
   - TF-IDF transformation (1–2 grams)

2. **Modeling**  
   - Logistic Regression classifier  
   - Tuned for high precision on "Fit" class  

3. **Evaluation**  
   - Accuracy  
   - Classification report  
   - Confusion matrix  

4. **Deployment**  
   - Gradio UI for instant testing  
   - Saved ML pipeline (`resume_model.joblib`)

---
ai-resume-screening-system/
│
├── data/
│ └── resumes_demo.csv # Sample dataset
│
├── models/
│ └── resume_model.joblib # Trained ML pipeline
│
├── notebooks/
│ └── resume_screening_ai.ipynb # Full Colab notebook
│
├── src/
│ └── train_resume_model.py # Script version of training pipeline
│
└── README.md

---

## 🛠 Tech Stack  
- Python  
- Pandas, NumPy  
- scikit-learn  
- TF-IDF Vectorizer  
- Logistic Regression  
- Joblib  
- Gradio  

---

## 📌 Future Improvements  
- Add BERT-based classifier  
- Add PDF resume parser  
- Multi-class role prediction  
- Deployment via FastAPI + Docker  

---

## 👤 Author  
**Malaiyarasan M**  
AI & Data Engineer  
👉 GitHub: https://github.com/Malaiyarasan  
👉 Portfolio: *(to be updated after all projects)*  



## 📁 Project Structure  

