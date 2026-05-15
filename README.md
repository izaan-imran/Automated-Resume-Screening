# 🤖 JobJive AI: Automated Resume Screening System

[cite_start]**JobJive AI** is a specialized recruitment automation tool designed to streamline the hiring process[cite: 122]. [cite_start]Built using a hybrid approach of **Ensemble Machine Learning** and **Deterministic Rule-Based Logic**, it accurately classifies resumes into **24 professional job categories**[cite: 5, 27].

---

## 📖 Project Overview
[cite_start]The manual screening of resumes is one of the most time-intensive tasks for HR teams[cite: 14, 15]. [cite_start]**JobJive AI** solves this by providing a "first-pass" filter that automatically reads, cleans, and categorizes candidate profiles, significantly reducing human error and unconscious bias[cite: 16, 132].

---

## ⚙️ How It Works (The Pipeline)
[cite_start]The system processes data through a sophisticated multi-stage pipeline[cite: 19]:

* [cite_start]**Text Extraction:** Utilizes `pdfplumber` to extract raw text from uploaded PDF or plain-text files[cite: 126, 146].
* [cite_start]**Preprocessing:** Implements a cleaning pipeline to remove URLs, emails, special characters, and lowercase all text[cite: 32, 33].
* [cite_start]**Feature Engineering:** Converts cleaned text into numerical vectors using **TF-IDF Vectorization** with 3,000 max features and an `ngram_range=(1,3)`[cite: 58, 61].
* [cite_start]**Classification:** Passes vectors through four models, collecting votes for a **Majority Voting Ensemble**[cite: 108, 109].

---

## 🧠 Model Architecture & Logic
[cite_start]To ensure high reliability, we combined multiple models to compensate for individual weaknesses[cite: 72]:

1. [cite_start]**Random Forest (75.25%):** Our best model, configured with 200 trees to handle noisy text data and reduce overfitting[cite: 88, 89, 92].
2. [cite_start]**Linear SVC (72.84%):** Highly effective for high-dimensional, sparse text features typical of resume data[cite: 84, 85, 87].
3. [cite_start]**Logistic Regression (69.82%):** A robust baseline using linear decision boundaries in the TF-IDF space[cite: 74, 75, 76].
4. [cite_start]**Naive Bayes (55.94%):** A fast, probability-based classifier that provides a diverse voting perspective[cite: 80, 81, 82].

### ⚖️ The Rule-Based Override Layer
[cite_start]A critical challenge was the misclassification of **Data Science** resumes as **Digital Media**[cite: 113]. To solve this, we implemented a custom logic layer:
* [cite_start]**Trigger:** The system scans raw text for 30 high-value IT signals like *TensorFlow, PyTorch, Pandas,* and *SQL*[cite: 115].
* [cite_start]**Execution:** If 3 or more signals are found and they outweigh "Digital Media" noise words, the system overrides the ML models to return **INFORMATION-TECHNOLOGY** with 100% confidence[cite: 117, 119].

---

## 🚀 Deployment & Links
* **Live Application:** [Paste your Streamlit link here]
* **GitHub Repository:** [Paste your GitHub link here]

---

## 🛠️ Technical Stack
* [cite_start]**Language:** Python 3.10+ [cite: 146]
* [cite_start]**Framework:** Streamlit (v1.30+) [cite: 146]
* [cite_start]**ML/NLP:** Scikit-learn, NLTK, Pandas, NumPy [cite: 146]
* [cite_start]**Serialization:** `joblib` for model caching and fast loading [cite: 146]

---

### 👥 Development Team
* **Shiraz Asif** (019) | **Izan Imran** (041) | [cite_start]**Dayyan Hashmi** (037) [cite: 3]
* [cite_start]**Institution:** Dawood University of Engineering & Technology (DUET) [cite: 1]
* [cite_start]**Supervised by:** Sir Jamal Khanzada [cite: 3]

[cite_start]*Submission Date: May 14, 2026* [cite: 3]
