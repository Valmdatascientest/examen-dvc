from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

def normalize_data(X_train_path, X_test_path):
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
     # Supprimer la colonne 'date' originale
    X_train = X_train.drop(columns=['date'])
    X_test = X_test.drop(columns=['date'])
     # Sélectionner uniquement les colonnes numériques
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train_numeric = X_train[numeric_cols]
    X_test_numeric = X_test[numeric_cols]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pd.DataFrame(X_train_scaled).to_csv('data/processed_data/X_train_scaled.csv', index=False)
    pd.DataFrame(X_test_scaled).to_csv('data/processed_data/X_test_scaled.csv', index=False)
    joblib.dump(scaler, 'models/scaler.pkl')

if __name__ == "__main__":
    normalize_data('data/processed_data/X_train.csv', 'data/processed_data/X_test.csv')
