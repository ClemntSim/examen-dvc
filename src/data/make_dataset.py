import pandas as pd
import os
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file_path = "/home/ubuntu/exam_SIMONIN_dvc/examen-dvc/data/raw_data/raw.csv"
output_folderpath = "/home/ubuntu/exam_SIMONIN_dvc/examen-dvc/data/processed_data/"

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    process_data(file_path, output_folderpath)

def process_data(file_path, output_folderpath):
    #Compilation des différentes fonctions de nettoyage, entrainement, enregistrement ...

    df = import_dataset(file_path, sep = ',', encoding = 'utf-8')
    df = remove_column(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test = standardizer(X_train, X_test)
    save_train_test(X_train, X_test, y_train, y_test, output_folderpath)
    return X_train, X_test, y_train, y_test

def import_dataset(file_path, **kwargs):
    #Importation des données
    return pd.read_csv(file_path, **kwargs)


def remove_column(dataframe):
    # Supprimer la colonne de date
    dataframe = dataframe.drop(columns = 'date')
    return dataframe

def split_data(df):
    # Jeu d'entrainement et jeu de test
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def standardizer(X_train, X_test):
    #Normalisation des données
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)

    # Convertir les numpy.ndarray en DataFrame
    X_train_sc = pd.DataFrame(X_train_sc, columns=X_train.columns)
    X_test_sc = pd.DataFrame(X_test_sc, columns=X_test.columns)
    return X_train_sc, X_test_sc


def save_train_test(X_train, X_test, y_train, y_test, output_folderpath):
    # Sauvegarde des données
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()

    




