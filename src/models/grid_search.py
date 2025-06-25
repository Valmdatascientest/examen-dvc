import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

# Charger les données
X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')

# Assurez-vous que y_train est un tableau 1D
y_train = y_train.values.ravel()

# Définir le modèle
model = GradientBoostingRegressor()

# Définir la grille de paramètres
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}

# Configurer GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Effectuer la recherche sur grille
grid_search.fit(X_train, y_train)

# Sauvegarder les meilleurs paramètres
best_params = grid_search.best_params_
pd.DataFrame([best_params]).to_pickle('models/best_params.pkl')