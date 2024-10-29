from app import app, db, Combination  # Asegúrate de que estás importando los modelos y db de app.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def get_data():
    with app.app_context():  # Agrega el contexto de aplicación aquí
        combinations = Combination.query.all()
        
        X = []
        y = []

        for comb in combinations:
            numbers = list(map(int, comb.numbers.split(',')))
            special = comb.special
            X.append(numbers)
            y.append(special)

        return pd.DataFrame(X), y

def train_and_save_model():
    X, y = get_data()
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    joblib.dump(model, 'trained_model.pkl')
    print("Modelo entrenado y guardado como 'trained_model.pkl'.")

if __name__ == '__main__':
    train_and_save_model()
