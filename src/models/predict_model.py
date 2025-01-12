
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import sys
import json

# Load your saved model
lasso = joblib.load("./models/trained_model.joblib")
X_test = pd.read_csv('./data/processed_data//X_test.csv')
y_test = pd.read_csv('./data/processed_data//y_test.csv')
y_test = np.ravel(y_test)

def main():
    y_pred = predict_model(X_test)
    score = evaluate_model(y_pred)
    save_prediction(y_pred)
    save_score(score)

def predict_model(X_test):
    y_pred = lasso.predict(X_test)
    return y_pred

def save_prediction(y_pred):
    prediction = pd.DataFrame({
        'y_test':y_test,
        'y_pred':y_pred
    })
    prediction_path = './data/prediction.csv'
    prediction.to_csv(prediction_path, index=False)
    print(f"Prédictions sauvegardées dans '{prediction_path}'.")


def evaluate_model(y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    score = {
        "mse": mse,
        "r2": r2
    }
    return score

def save_score(score):
    score_path = './metrics/scores.json'
    with open(score_path, 'w') as f:
        json.dump(score, f)

if __name__ == "__main__":
    main()