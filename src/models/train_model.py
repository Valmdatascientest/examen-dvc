from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib

def train_model(X_train_path, y_train_path, params_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    params = joblib.load(params_path)
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/trained_model.pkl')

if __name__ == "__main__":
    train_model('data/processed_data/X_train_scaled.csv', 'data/processed_data/y_train.csv', 'models/best_params.pkl')
