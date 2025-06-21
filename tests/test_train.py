from src.preprocess import load_data, split_data
import os

def test_load_data():
    path = os.path.join("data", "diabetes.csv")
    df = load_data(path)
    assert df.shape[0] > 0
    assert "Outcome" in df.columns

def test_split_data():
    path = os.path.join("data", "diabetes.csv")
    df = load_data(path)
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) > 0 and len(X_test) > 0
