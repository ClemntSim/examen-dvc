
import pandas as pd 
from sklearn.linear_model import Lasso
import joblib
import numpy as np

print(joblib.__version__)

X_train = pd.read_csv('./data/processed_data/X_train.csv')
X_test = pd.read_csv('./data/processed_data//X_test.csv')
y_train = pd.read_csv('./data/processed_data//y_train.csv')
y_test = pd.read_csv('./data/processed_data//y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Importation des paramètres du modèle
param_lasso = joblib.load('./models/lasso_best_model.pkl')

lasso = Lasso(**param_lasso)

#Train the model
lasso.fit(X_train, y_train)

#Save the trained model to a file
model_filename = './models/trained_model.joblib'
joblib.dump(lasso, model_filename)
print("Model trained and saved successfully.")