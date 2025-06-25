import pandas as pd
import pickle
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
    valid_params = {k: int(v) if isinstance(v, np.int64) else v for k, v in params.items() if k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']}

    # Créer et entraîner le modèle
    model = RandomForestRegressor(**valid_params)
    model.fit(X_train, y_train)

    # Sauvegarder le modèle entraîné
    with open('models/trained_model.pkl', 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    import numpy as np
    train_model('data/processed_data/X_train_scaled.csv', 'data/processed_data/y_train.csv', 'models/best_params.pkl')
