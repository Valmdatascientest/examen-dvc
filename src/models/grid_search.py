from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import joblib

def grid_search(X_train_path, y_train_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
    }
    grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    joblib.dump(grid_search.best_params_, 'models/best_params.pkl')

if __name__ == "__main__":
    grid_search('data/processed_data/X_train_scaled.csv', 'data/processed_data/y_train.csv')
