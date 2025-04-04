import pandas as pd
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import os

def plot_and_save_confusion_matrix(model, X_test, y_test, save_path="../models/confusion_matrix_bestmodel.png"):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))  

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", ax=ax)

    ax.set_title("Matriz de Confusión", fontsize=14)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

def plot_and_save_roc_auc(model, X_test, y_test, save_path="../models/roc_curve_bestmodel.png"):
    y_probs = model.predict_proba(X_test)[:, 1]  
    
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Curva ROC-AUC")
    plt.legend(loc="lower right")
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

# Cargar la tabla transformada
def eval_model(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, ' cargado correctamente')
    # Predecimos sobre el set de datos de validación 
    X_test = df.drop(['Class'],axis=1)
    y_test = df[['Class']]
    # Leemos el modelo entrenado para usarlo
    filename = '../models/best_model.pkl'
    model = pickle.load(open(filename, 'rb'))
    print('Modelo importado correctamente')
    y_pred=model.predict(X_test)
    #Guardamos matris de confusión
    plot_and_save_confusion_matrix(model, X_test, y_test)
    #Guardamos curva roc
    plot_and_save_roc_auc(model, X_test, y_test )
    # Generamos métricas de diagnóstico
    cm_test = confusion_matrix(y_test,y_pred_test)
    print("Matriz de confusion: ")
    print(cm_test)
    accuracy_test=accuracy_score(y_test,y_pred_test)
    print("Accuracy: ", accuracy_test)
    precision_test=precision_score(y_test,y_pred_test)
    print("Precision: ", precision_test)
    recall_test=recall_score(y_test,y_pred_test)
    print("Recall: ", recall_test)


# Validación desde el inicio
def main():
    df = eval_model('credit_card_fraud_testing.csv')
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()