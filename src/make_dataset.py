import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler

# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df


# Realizamos la transformación de datos
def data_preparation(filename):
    #Cargar los datos
    df=read_file_csv(filename)
    #Eliminar duplicados
    df.drop_duplicates(inplace=True)
    #Escalar los datos
    robust_scaler=RobustScaler()
    df[['Time', 'Amount']]=robust_scaler.fit_transform(df[['Time', 'Amount']])
    if filename!='validation_data.csv':
        X = df.drop('Class', axis=1)
        y = df['Class']
        rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        df = pd.concat([X_resampled, y_resampled], axis=0)
    print('Transformación de datos completa')
    return df


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, filename):
    df.to_csv(os.path.join('../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')


# Generamos las matrices de datos que se necesitan para la implementación

def main():
    # Matriz de Entrenamiento
    df1 = data_preparation('credit_card_fraud_data.csv')
    data_exporting(df1, 'credit_card_fraud_train.csv')
    # Matriz de Validación
    df2 = data_preparation('testing_data.csv')
    data_exporting(df2, 'credit_card_fraud_testing.csv')
    # Matriz de Scoring
    df3 = data_preparation('validation_data.csv')
    data_exporting(df2, 'final_score.csv')
    
if __name__ == "__main__":
    main()