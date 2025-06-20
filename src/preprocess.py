import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path=r"C:\Users\Lenovo\Desktop\projects\ml ops\data\diabetes.csv"):
    # Just load the file normally if it already has headers
    df = pd.read_csv(path)
    return df

def split_data(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return train_test_split(X, y, test_size=0.2, random_state=42)