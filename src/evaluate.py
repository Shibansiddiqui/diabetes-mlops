import joblib
from preprocess import load_data, split_data
from sklearn.metrics import accuracy_score

model = joblib.load("model.pkl")
df = load_data()
X_train, X_test, y_train, y_test = split_data(df)

preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds)}")
