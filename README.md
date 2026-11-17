# 📧 NLP Spam Email Classifier Using Machine Learning

<p align="center">
  <strong>A Natural Language Processing and Machine Learning Project</strong><br>
  An intelligent web application that classifies email messages as Spam or Ham using NLP preprocessing and Gaussian Naive Bayes.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Flask-Web%20Application-black?logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/NLP-Text%20Processing-purple" alt="NLP">
  <img src="https://img.shields.io/badge/Machine%20Learning-Gaussian%20Naive%20Bayes-orange" alt="Machine Learning">
  <img src="https://img.shields.io/badge/TensorFlow-Embedding-orange?logo=tensorflow" alt="TensorFlow">
</p>

---

## 📌 Project Overview

The **NLP Spam Email Classifier** is a Machine Learning project that identifies whether an email message is Spam or Ham (Legitimate Email).

The project uses Natural Language Processing (NLP) techniques to preprocess text data and convert textual messages into numerical representations using the One-Hot Encoding technique.

The processed data is passed to a Gaussian Naive Bayes classifier, which learns patterns from the training dataset and predicts whether a new email message belongs to the Spam or Ham category.

A Flask web application provides an interactive interface where users can enter an email message and check its classification result.

### 🎯 Project Objectives

- Understand Natural Language Processing fundamentals.
- Perform text preprocessing using NLP techniques.
- Convert textual data into numerical representations.
- Implement One-Hot Encoding and Padding Sequences.
- Train a Machine Learning classification model.
- Evaluate model performance using classification metrics.
- Save and reuse the trained model using Pickle.
- Develop an interactive Flask web application.
- Predict whether user-entered messages are Spam or Ham.

---

## 🔄 Project Workflow

The project follows the following Machine Learning pipeline:

```text
             SPAM DATASET
                  |
                  |
           DATA PREPROCESSING
                  |
                  |
          NLP PREPROCESSING
                  |
          -------------------
          |        |        |
        Lower   Remove   Remove
        Case   Punctuation Stopwords
                  |
                  |
             Lemmatization
                  |
                  |
          ONE-HOT ENCODING
                  |
                  |
           PADDING SEQUENCES
                  |
                  |
          TRAIN-TEST SPLIT
                  |
                  |
        GAUSSIAN NAIVE BAYES
                  |
                  |
          MODEL EVALUATION
                  |
                  |
          SAVE MODEL (.pkl)
                  |
                  |
           FLASK APPLICATION
                  |
                  |
          USER ENTERS EMAIL
                  |
                  |
          NLP PREPROCESSING
                  |
                  |
           EMBEDDING
                  |
                  |
             PREDICTION
                  |
             -----------
             |         |
            SPAM      HAM
```

---

## 📂 Dataset Information

The project uses the SMS Spam Collection dataset.

The dataset contains text messages classified into two categories:

| Category | Description |
|---|---|
| Ham | Normal or legitimate messages |
| Spam | Unwanted or promotional messages |

### Dataset Preprocessing

The original dataset contains the following columns:

- `v1` – Message classification.
- `v2` – Email/message text.
- `Unnamed: 2`
- `Unnamed: 3`
- `Unnamed: 4`

Unwanted columns are removed, and the remaining columns are renamed.

```python
df = df.drop(
    ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],
    axis=1
)

df = df.rename(
    columns={
        'v1': 'Target',
        'v2': 'Gmails'
    }
)
```

### Target Encoding

The categorical target values are converted into numerical values.

```python
df['Target'] = df['Target'].map({
    'ham': 1,
    'spam': 0
}).astype(int)
```

Target mapping:

- `0` = Spam
- `1` = Ham

---

## 🧠 Natural Language Processing (NLP)

NLP is used to process textual data and convert it into a format suitable for Machine Learning algorithms.

This project uses two preprocessing stages.

### Level 1: Text Preprocessing

The following techniques are applied to the email messages.

**1. Convert Text into Lowercase**

Converts all uppercase characters into lowercase.

```python
text = text.lower()
```

**2. Remove Punctuation**

Removes punctuation symbols from the messages.

```python
import string

text = ''.join([
    i for i in text
    if i not in string.punctuation
])
```

**3. Remove Stopwords**

Removes commonly occurring English words that are excluded during preprocessing.

```python
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
```

**4. Lemmatization**

Converts words into their base or dictionary forms.

```python
from nltk.stem import WordNetLemmatizer

lemm_obj = WordNetLemmatizer()

text = ' '.join([
    lemm_obj.lemmatize(i)
    for i in text.split()
    if i not in stop_words
])
```

### Level 2: Text Embedding

After preprocessing, textual messages are converted into numerical representations.

**One-Hot Encoding**

TensorFlow's `one_hot()` function converts text into numerical token representations.

```python
from tensorflow.keras.preprocessing.text import one_hot

dic_size = 9000

values = [
    one_hot(i, dic_size)
    for i in original_text
]
```

**Padding Sequences**

Padding ensures that all input messages have the same sequence length.

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

final_inde_data = pad_sequences(
    values,
    maxlen=s,
    padding='post'
)
```

This produces numerical input data suitable for the Machine Learning model.

---

## 🤖 Machine Learning Algorithm

### Gaussian Naive Bayes

This project uses the Gaussian Naive Bayes classification algorithm.

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It assumes that features follow a Gaussian (normal) distribution within each class.

### Model Training

The dataset is divided into training and testing portions.

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    final_inde_data,
    df['Target'],
    test_size=0.2,
    random_state=42
)
```

The Gaussian Naive Bayes model is trained using:

```python
from sklearn.naive_bayes import GaussianNB

reg = GaussianNB()

reg.fit(x_train, y_train)
```

### Model Evaluation

The following evaluation metrics are used:

- Confusion Matrix
- Accuracy Score
- Classification Report
- Precision
- Recall
- F1-Score

```python
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

y_pred = reg.predict(x_test)

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
```

### Cross-Validation

Cross-validation is also explored to evaluate model performance across multiple data splits.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    estimator=reg,
    X=X,
    y=y,
    cv=30
)

print("Cross Validation Accuracy:", scores.mean())
```

---

## 💾 Model Serialization Using Pickle

After training the model, it is saved using the Pickle library.

```python
import pickle

with open("NLP_ML_Model.pkl", "wb") as f:
    pickle.dump(reg, f)
```

The saved model is loaded into the Flask application.

```python
with open("NLP_ML_Model.pkl", "rb") as f:
    model = pickle.load(f)
```

This allows the application to reuse the trained Machine Learning model without retraining it whenever Flask starts.

**Important:** The Pickle file must be generated using trusted training code and placed in the project root directory.

---

## 🌐 Flask Web Application

The project includes an interactive web application developed using Flask, HTML, CSS and JavaScript.

### Frontend Features

The web interface provides:

1. Modern and responsive user interface.
2. Spam Classifier information section.
3. Email message input area.
4. Check Email button.
5. Reset button.
6. Spam classification result.
7. Ham classification result.
8. Error message handling.
9. Developer information in the footer.

### Backend Features

The Flask backend performs the following operations:

1. Loads the trained Pickle model.
2. Displays the HTML frontend.
3. Accepts email text through an HTML form.
4. Receives submitted data using the POST method.
5. Converts the email into lowercase.
6. Removes punctuation.
7. Removes stopwords.
8. Applies lemmatization.
9. Performs One-Hot Encoding.
10. Applies Padding Sequences.
11. Passes the processed data to the Gaussian Naive Bayes model.
12. Generates the prediction.
13. Returns the classification result to the frontend.

### Flask Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Displays the homepage |
| `/` | POST | Receives email text and predicts its classification |

### Prediction Logic

The model produces one of two numerical outputs.

```python
sol = model.predict(results)[0]

if sol == 0:

    prediction = "SPAM EMAIL"

else:

    prediction = "HAM EMAIL"
```

---

## 📁 Project Folder Structure

```text
Nlp_pipeline_project/
│
├── .venv/
│
├── templates/
│   └── index.html
│
├── app.py
│
├── main.py
│
├── NLP_ML_Model.pkl
│
├── requirements.txt
│
└── README.md
```

### File Descriptions

| File | Description |
|---|---|
| `main.py` | NLP preprocessing, Machine Learning training and model saving |
| `app.py` | Flask backend application |
| `index.html` | Frontend user interface |
| `NLP_ML_Model.pkl` | Saved Gaussian Naive Bayes model |
| `requirements.txt` | Project dependencies |
| `README.md` | Project documentation |

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.x
- PyCharm or Visual Studio Code
- pip
- Basic knowledge of Python and Machine Learning

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd Nlp_pipeline_project
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Required Libraries

```bash
pip install flask numpy pandas scikit-learn tensorflow nltk
```

### 4. Download NLTK Resources

The Flask application downloads the required NLTK resources automatically.

Alternatively, execute:

```python
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 5. Check Project Files

Ensure that the following files exist:

```text
app.py
NLP_ML_Model.pkl
templates/index.html
```

The trained model must be downloaded from Google Colab and copied into the project root.

### 6. Run the Flask Application

```bash
python app.py
```

### 7. Open the Application

Visit the following address in your browser:

```text
http://127.0.0.1:5000
```

Enter an email message and click the **CHECK EMAIL** button to view the classification result.

---

## 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical operations |
| Pandas | Dataset handling |
| NLTK | NLP preprocessing |
| TensorFlow / Keras | One-Hot Encoding and Padding |
| Scikit-learn | Machine Learning and evaluation |
| Gaussian Naive Bayes | Classification algorithm |
| Pickle | Model serialization |
| Flask | Backend web framework |
| HTML5 | Frontend structure |
| CSS3 | Frontend styling |
| JavaScript | Frontend interactions |

---

## 🚀 Future Enhancements

Possible improvements for this project include:

- Experimenting with Multinomial Naive Bayes.
- Implementing TF-IDF Vectorization.
- Comparing multiple Machine Learning algorithms.
- Improving performance on imbalanced datasets.
- Adding email confidence or probability scores.
- Improving spam detection using advanced NLP techniques.
- Deploying the application to a cloud platform.

---

## 👨‍💻 Developer Information

<p align="center">
  <strong>M. Yuva Teja</strong><br>
  Machine Learning & NLP Project
</p>

<p align="center">
  Developed using Python, Natural Language Processing, Machine Learning and Flask.
</p>

---

## 📌 Disclaimer

This project is developed for educational and Machine Learning practice purposes.

The classification result depends on the dataset, preprocessing techniques and trained model. It should not be considered a guaranteed email security or spam detection solution.