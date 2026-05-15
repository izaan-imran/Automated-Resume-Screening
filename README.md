# 🤖 JobJive AI: Automated Resume Screening System

**JobJive AI** is a specialized recruitment automation tool designed to streamline the hiring process. Built using a hybrid approach of **Ensemble Machine Learning** and **Deterministic Rule-Based Logic**, it accurately classifies resumes into **24 professional job categories**.

---

## 📖 Project Overview
The manual screening of resumes is one of the most time-intensive tasks for HR teams. **JobJive AI** solves this by providing a "first-pass" filter that automatically reads, cleans, and categorizes candidate profiles, significantly reducing human error and unconscious bias.

---

## ⚙️ How It Works (The Pipeline)
The system processes data through a sophisticated multi-stage pipeline:

* **Text Extraction:** Utilizes `pdfplumber` to extract raw text from uploaded PDF or plain-text files.
* **Preprocessing:** Implements a cleaning pipeline to remove URLs, emails, special characters, and lowercase all text.
* **Feature Engineering:** Converts cleaned text into numerical vectors using **TF-IDF Vectorization** with 3,000 max features and an `ngram_range=(1,3)`.
* **Classification:** Passes vectors through four models, collecting votes for a **Majority Voting Ensemble**.

---

## 🧠 Model Architecture & Logic
To ensure high reliability, we combined multiple models to compensate for individual weaknesses:

1. **Random Forest (75.25%):** Our best model, configured with 200 trees to handle noisy text data and reduce overfitting.
2. **Linear SVC (72.84%):** Highly effective for high-dimensional, sparse text features typical of resume data.
3. **Logistic Regression (69.82%):** A robust baseline using linear decision boundaries in the TF-IDF space.
4. **Naive Bayes (55.94%):** A fast, probability-based classifier that provides a diverse voting perspective.

### ⚖️ The Rule-Based Override Layer
A critical challenge was the misclassification of **Data Science** resumes as **Digital Media**. To solve this, we implemented a custom logic layer:
* **Trigger:** The system scans raw text for 30 high-value IT signals like *TensorFlow, PyTorch, Pandas,* and *SQL*.
* **Execution:** If 3 or more signals are found and they outweigh "Digital Media" noise words, the system overrides the ML models to return **INFORMATION-TECHNOLOGY** with 100% confidence.

---

## 🚀 Deployment & Links
* **Live Application:** (https://automated-resume-screening-1.streamlit.app/)
  
---

## 🛠️ Technical Stack
* **Language:** Python 3.10+
* **Framework:** Streamlit (v1.30+)
* **ML/NLP:** Scikit-learn, NLTK, Pandas, NumPy
* **Serialization:** `joblib` for model caching and fast loading

---
