# Toxic Comment Classifier

A multi-label NLP machine learning project that detects different types of toxic comments using TF-IDF vectorization and a Linear SVC classifier.

The trained machine learning model is exposed through a FastAPI REST API, allowing users to submit a comment and receive toxicity predictions through an API endpoint.

## Features

The model predicts six different categories:

- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

The classifier supports multi-label prediction, meaning a single comment can belong to multiple categories at the same time.

## Machine Learning Pipeline

The project follows this pipeline:

Raw Text
   ↓
Text Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Balanced Linear SVC
   ↓
One-vs-Rest Classification
   ↓
Multi-label Predictions
   ↓
FastAPI REST API

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Linear SVC
- One-vs-Rest Classification
- FastAPI
- Uvicorn
- Jupyter Notebook

## Model

### TF-IDF Vectorization

TF-IDF (Term Frequency-Inverse Document Frequency) converts text comments into numerical feature vectors.

The trained TF-IDF vectorizer is saved and reused during inference so that new comments are transformed using the same vocabulary as the training data.

### Linear SVC

A Linear Support Vector Classifier is used as the underlying classification algorithm.

Because the dataset contains imbalanced classes, `class_weight="balanced"` is used to give more importance to minority classes.

### One-vs-Rest Classification

The problem is multi-label, since a comment can belong to multiple toxicity categories.

One-vs-Rest classification trains a separate binary classifier for each toxicity category.

This allows the model to independently predict:

- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

## Model Evaluation

The model was evaluated using classification reports containing:

- Precision
- Recall
- F1-score
- Support

The model achieves different performance levels across the six toxicity categories, with minority classes such as `threat`, `severe_toxic`, and `identity_hate` being more challenging due to class imbalance.

## FastAPI

The trained model is served using FastAPI.

### Home Endpoint

`GET /`

Returns a message confirming that the API is running.

### Prediction Endpoint

`POST /predict`

#### Request

```json
{
  "comment": "You are a stupid idiot person and I hate you"
}

### Example Response

```json
{
  "comment": "You are a stupid idiot person and I hate you",
  "is_toxic": true,
  "predictions": {
    "toxic": true,
    "severe_toxic": false,
    "obscene": true,
    "threat": false,
    "insult": true,
    "identity_hate": false
  }
}
