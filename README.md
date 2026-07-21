# 🧠 Mental Health Emotion Prediction using BiLSTM

## 📌 Project Overview

This project is a Deep Learning-based Natural Language Processing (NLP) application that predicts the mental health emotion of a given text statement. The model classifies statements into one of seven mental health categories using a BiLSTM network and Word2Vec embeddings.

---

## 🎯 Features

- Predicts mental health emotions from text.
- Classifies statements into **7 categories**:
  - Anxiety
  - Bipolar
  - Depression
  - Normal
  - Personality Disorder
  - Stress
  - Suicidal
- Uses Word2Vec for text embedding.
- BiLSTM model built using TensorFlow/Keras.
- Supports external validation using new datasets.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Word2Vec (Gensim)
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## 📂 Project Structure

```
Mental-Health-Emotion-Prediction/
│
├── Mental_Health_Emotion_Prediction.ipynb
├── bilstm_model.keras
├── word2vec.model
├── label_encoder.pkl
├── validate.xlsx
├── Combined Data.csv.zip
└── README.md
```

---

## 📊 Dataset

The dataset contains mental health-related text statements labeled into seven emotion categories.

### Classes

- Anxiety
- Bipolar
- Depression
- Normal
- Personality Disorder
- Stress
- Suicidal

---

## ⚙️ Methodology

1. Data Cleaning
2. Text Preprocessing
3. Tokenization
4. Word2Vec Embedding
5. BiLSTM Model Training
6. Model Evaluation
7. External Validation

---

## 📈 Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

**Test Accuracy:** **~75%**

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Mental-Health-Emotion-Prediction.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the Notebook

```bash
jupyter notebook
```

Open:

```
Mental_Health_Emotion_Prediction.ipynb
```

---

## 📌 Future Improvements

- Develop a Flask/Streamlit web application.
- Deploy the model on Render or Hugging Face Spaces.
- Improve model accuracy using Transformer-based models such as BERT or RoBERTa.
- Expand the dataset for better generalization.

---

## 👩‍💻 Author

**Ananthula Varshitha**

B.Tech – Computer Science and Engineering (AI & ML)

GitHub: https://github.com/Varshitha14-web

---

## ⭐ Acknowledgements

This project was developed for educational and research purposes to explore the application of Deep Learning and Natural Language Processing in mental health emotion prediction.