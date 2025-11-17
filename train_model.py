import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from preprocess import cargar_datos

def entrenar():
    X_train, X_test, y_train, y_test = cargar_datos('data/heart.xlsx')
    num = X_train.select_dtypes(include=['number']).columns
    cat = X_train.select_dtypes(exclude=['number']).columns

    pre = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat)
    ])

    model = Pipeline([
        ('pre', pre),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    model.fit(X_train, y_train)
    joblib.dump(model, 'modelo_entrenado.joblib')
    print('Modelo guardado como modelo_entrenado.joblib')

if __name__ == "__main__":
    entrenar()
