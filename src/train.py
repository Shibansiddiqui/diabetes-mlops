import os
import joblib
import mlflow
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_data, split_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Load and split data
df = load_data()
X_train, X_test, y_train, y_test = split_data(df)

# Set experiment
mlflow.set_experiment("Diabetes Predictor")

with mlflow.start_run():

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log parameters
    mlflow.log_param("n_estimators", 100)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "model")

    # Save model locally
    joblib.dump(model, "models/model.pkl")

    print(f"Model trained and saved.")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
