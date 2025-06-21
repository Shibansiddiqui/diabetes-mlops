# test/test_train.py

import sys
import os

# Add the root folder (i.e., "ml ops") to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import load_data, split_data  # ✅ Correct import path
def test_load_data():
    df = load_data(path=r"C:\Users\Lenovo\Desktop\projects\ml ops\data\diabetes.csv")
    assert df.shape[0] > 0
    assert "Outcome" in df.columns

def test_split_data():
    df = load_data(path=r"C:\Users\Lenovo\Desktop\projects\ml ops\data\diabetes.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) > 0 and len(X_test) > 0
