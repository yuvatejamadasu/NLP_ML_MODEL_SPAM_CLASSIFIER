# Import libraries

from flask import Flask, render_template, request

import numpy as np
import pickle
import string
import nltk
import warnings
import os

from pathlib import Path

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Ignore warnings
warnings.filterwarnings('ignore')


# Create Flask object
app = Flask(__name__)


# Download NLTK datasets
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


# Initialize NLP objects
lemm_obj = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))


# Load Machine Learning model

MODEL_PATH = Path(__file__).resolve().parent / "NLP_ML_Model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# Embedding configuration

dic_size = 9000

max_length = model.n_features_in_


# NLP preprocessing function

def preprocess_text(text):

    # Convert into lowercase
    text = text.lower()

    # Remove punctuation
    text = ''.join(
        [i for i in text if i not in string.punctuation]
    )

    # Remove stopwords and apply lemmatization
    words = []

    for i in text.split():

        if i not in stop_words:

            words.append(lemm_obj.lemmatize(i))

    text = ' '.join(words)

    return text


# Embedding function

def embedding_text(text):

    # Convert text into numbers
    values = [one_hot(text, dic_size)]

    # Make dimensions equal
    results = pad_sequences(
        values,
        maxlen=max_length,
        padding='post'
    )

    return results


# Home route

@app.route("/", methods=["GET", "POST"])

def home():

    prediction = None

    submitted_text = ""

    error = None

    if request.method == "POST":

        # Get email from frontend
        submitted_text = request.form.get("message", "").strip()

        if not submitted_text:

            error = "Please enter an email message."

        else:

            try:

                # NLP preprocessing
                cleaned_text = preprocess_text(submitted_text)

                # Embedding technique
                results = embedding_text(cleaned_text)

                # Machine Learning prediction
                sol = model.predict(results)[0]

                # Classification result
                if sol == 0:

                    prediction = "SPAM EMAIL"

                else:

                    prediction = "HAM EMAIL"

            except Exception as e:

                error = "Prediction failed. Please check the model configuration."

                app.logger.exception("Prediction Error")


    return render_template(
        "index.html",
        prediction=prediction,
        submitted_text=submitted_text,
        error=error
    )


# Main function

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port, debug=False)