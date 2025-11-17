import pandas as pd
from sklearn.model_selection import train_test_split

def cargar_datos(path):
    df = pd.read_excel(path)
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
