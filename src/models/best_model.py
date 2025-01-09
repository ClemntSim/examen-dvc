import pandas as pd
import joblib
import os
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso


train_data_path = "/home/ubuntu/exam_SIMONIN_dvc/examen-dvc/data/processed_data/"
model_save_path = "/home/ubuntu/exam_SIMONIN_dvc/examen-dvc/models/"


param_grid = {'alpha': [0.1, 1, 10, 50, 100]}

def main():
    
    print("Début de la selection")
    best_models = find_best_model(train_data_path, model_save_path)
    print("Modele selectionné.")

def find_best_model(train_data_path, model_save_path):

    X_train, y_train = charge_data(train_data_path)
    
    best_params = gridsearch(X_train, y_train, param_grid)
    

    save_model(best_params, model_save_path)
    return best_params

def charge_data(train_data_path):

# Chargement des données d'entraînement depuis un fichier CSV
    X_train = pd.read_csv(os.path.join(train_data_path, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(train_data_path, 'y_train.csv'))

#Modification du format de y_train pour éviter le message d'erreur 2D en 1D:
    y_train = y_train.values.ravel()

    return X_train, y_train

def gridsearch(X_train, y_train, param_grid):

    #Initialisation du test de Lasso
    lasso = Lasso()

    #Entrainement du Gridsearch
    grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    #Extraction du meilleur modèle
    ##best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Les meilleurs paramètres de lasso sont: {best_params}")
    return best_params

def save_model(best_params, model_save_path):
    # Sauvegarde du modèle au format .pkl
    best_model_path = os.path.join(model_save_path, 'lasso_best_model.pkl')
    joblib.dump(best_params, best_model_path)
    print(f"Sauvegarde du modèle de lasso: {best_model_path}")

if __name__ == '__main__':
    main()




