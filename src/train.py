import pandas as pd
import xgboost as xgb
import pickle
import os
from sklearn.model_selection import train_test_split


# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    X_train = df.drop(['Class'],axis=1)
    y_train = df[['Class']]
    print(filename, ' cargado correctamente')
    # Entrenamos el modelo con toda la muestra
    xgb_model=XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb_model.fit(X_train, y_train)
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    filename='../models/best_model.pkl'
    pickle.dump(xgb_model, open(filename, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('credit_card_fraud_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()