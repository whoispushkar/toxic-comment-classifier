# Toxic Comment Classifier

A multi-label NLP machine learning project that detects different
types of toxic comments using TF-IDF and Linear SVC.

## Features

The model predicts six categories:

- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

## Tech Stack

- Python
- Pandas
- Scikit-learn
- NLP / TF-IDF
- Linear SVC
- One-vs-Rest Classification
- FastAPI

## Model

Text is converted into numerical features using TF-IDF.

A balanced Linear SVC model with One-vs-Rest classification
is then used to predict multiple toxicity labels.

## API

The trained model is deployed through FastAPI.

### Endpoint

POST `/predict`

Example input:

{
    "comment": "You are a stupid idiot"
}
