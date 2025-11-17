import joblib
import pandas as pd

def predecir(nueva_fila: dict):
    model = joblib.load('modelo_entrenado.joblib')
    df = pd.DataFrame([nueva_fila])
    pred = model.predict(df)[0]
    return int(pred)

if __name__ == '__main__':
    ejemplo = {
        'Age': 60,
        'Sex': 'M',
        'ChestPainType': 'ATA',
        'RestingBP': 130,
        'Cholesterol': 250,
        'FastingBS': 0,
        'RestingECG': 'Normal',
        'MaxHR': 150,
        'ExerciseAngina': 'N',
        'Oldpeak': 1.0,
        'ST_Slope': 'Up'
    }
    print(predecir(ejemplo))
