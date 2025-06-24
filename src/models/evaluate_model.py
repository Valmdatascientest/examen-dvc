from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import joblib
import json

def evaluate_model(X_test_path, y_test_path, model_path):
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {'mse': mse, 'r2': r2}
    with open('metrics/scores.json', 'w') as f:
        json.dump(metrics, f)
    pd.DataFrame(y_pred).to_csv('data/processed_data/y_pred.csv', index=False)

if __name__ == "__main__":
    evaluate_model('data/processed_data/X_test_scaled.csv', 'data/processed_data/y_test.csv', 'models/trained_model.pkl')
