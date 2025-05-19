<h1 align="center">🧠 Hate Speech Detection using ML Algorithms</h1>

<p align="center">
  <em>Classifying toxic, abusive, and offensive content using machine learning and NLP techniques.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NLP-Natural%20Language%20Processing-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-ML-yellow?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Scikit--learn-Classification-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/TF--IDF-Vectorization-brightgreen?style=for-the-badge" />
</p>

---

## 📘 Overview

This project focuses on building a robust pipeline to detect hate speech from social media content using five different machine learning models. The dataset is preprocessed using advanced NLP techniques and then fed into models for training, evaluation, and comparison.

---

## 🚀 Objectives

- Classify text as hate speech, offensive language, or neither.
- Compare the performance of 5 different machine learning algorithms.
- Use TF-IDF and CountVectorizer for feature extraction.
- Evaluate models using metrics like accuracy, precision, recall, and F1-score.

---

## 🛠️ Tech Stack

| Category        | Tools & Libraries                              |
|----------------|--------------------------------------------------|
| Language        | Python                                           |
| NLP             | NLTK, re (Regex), Stopwords, Lemmatization      |
| Vectorization   | CountVectorizer, TF-IDF                         |
| ML Algorithms   | Logistic Regression, Naive Bayes, Random Forest, SVM, KNN |
| Evaluation      | Confusion Matrix, Classification Report, ROC-AUC |
| Visualization   | Seaborn, Matplotlib                             |
| Development     | Jupyter Notebook / Google Colab                 |

---

## 🧾 Dataset

- **Source**: [Kaggle – Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/fizzbuzz/hate-speech-offensive-language-dataset)
- **Features**:
  - `tweet`: the text content
  - `class`: 0 = Hate Speech, 1 = Offensive Language, 2 = Neither

---

## 📊 Models Used

- **Logistic Regression**
- **Multinomial Naive Bayes**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**

---

## 📈 Performance Metrics

| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Logistic Regression    | 91.2%    | 0.90      | 0.89   | 0.90     |
| Naive Bayes            | 88.5%    | 0.87      | 0.85   | 0.86     |
| Random Forest          | 93.1%    | 0.92      | 0.91   | 0.92     |
| SVM                    | 92.7%    | 0.92      | 0.90   | 0.91     |
| KNN                    | 85.4%    | 0.83      | 0.81   | 0.82     |

---
