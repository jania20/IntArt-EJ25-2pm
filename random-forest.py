
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # Para balancear datos

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Cargar datos 
data = pd.read_csv("creditcard.csv")
print("Distribución de clases:\n", data['Class'].value_counts())


X = data.drop('Class', axis=1)
y = data['Class']


smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.3, random_state=42
)


model = RandomForestClassifier(
    n_estimators=100,      
    max_features='sqrt',   
    bootstrap=True,       
    oob_score=True,       
    n_jobs=-1,            
    random_state=42
)
model.fit(X_train, y_train)

# 5. Evaluación
y_pred = model.predict(X_test)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))


importances = model.feature_importances_
features = X.columns
plt.barh(features, importances)
plt.title("Importancia de características")
plt.show()