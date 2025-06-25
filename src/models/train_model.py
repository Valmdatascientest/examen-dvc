import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def train_model(X_train_path, y_train_path, params_path):
    # Charger les données
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    # Assurez-vous que y_train est un tableau 1D
    y_train = y_train.values.ravel()

    # Charger les meilleurs paramètres
    with open(params_path, 'rb') as file:
        params = pickle.load(file)

    # Convertir les paramètres en types natifs Python
    valid_params = {}
    for key, value in params.items():
        if key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
            if isinstance(value, (np.int64, pd.Series)):
                # Extraire la première valeur si c'est une série
                value = value.iloc[0] if isinstance(value, pd.Series) else int(value)
            valid_params[key] = value

    # Vérifier que max_depth est valide
    if 'max_depth' in valid_params:
        max_depth = valid_params['max_depth']
        if isinstance(max_depth, int) and max_depth < 1:
            valid_params['max_depth'] = None

    # Créer et entraîner le modèle
    model = RandomForestRegressor(**valid_params)
    model.fit(X_train, y_train)

    # Sauvegarder le modèle entraîné
    with open('models/trained_model.pkl', 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    train_model('data/processed_data/X_train_scaled.csv', 'data/processed_data/y_train.csv', 'models/best_params.pkl')
