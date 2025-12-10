# AI Resume Screening System (NLP + Machine Learning)

This project builds an AI-powered resume screening system that classifies
resumes into job-fit categories using **NLP + classical ML**.

---

## 🧠 Problem

Manual resume screening is slow and subjective.
This system automates the first-level screening step by:

- Extracting text from resumes
- Cleaning and vectorizing using NLP
- Training a classifier to predict job-fit category

---

## 🧩 Approach

1. **Data Collection**
   - Dataset of resumes stored in CSV:
     - `text` → extracted resume text
     - `label` → job role category

2. **Preprocessing**
   - Lowercasing
   - Removing punctuation and numbers
   - Stopword removal
   - Tokenization

3. **Feature Extraction**
   - TF-IDF Vectorization (1–2 grams)

4. **Modeling**
   - Logistic Regression classifier
   - Achieved ~92% accuracy on validation set

---

## 🧰 Tech Stack

- Python
- Scikit-Learn
- TF-IDF Vectorizer
- Pandas / NumPy

---

## 📁 Project Structure

```text
ai-resume-screening-system/
│
├── data/
│   └── resumes.csv            # (placeholder)
│
├── src/
│   └── train_resume_model.py  # main ML script
│
└── README.md
