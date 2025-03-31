
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Cargar y explorar los datos
data = pd.read_csv("reviews_sentiment.csv", sep=';')
print("Primeras filas del dataset:")
print(data.head())
print("\nResumen estadístico:")
print(data.describe())

# 2. Preprocesamiento

X = data[['wordcount', 'sentimentValue']]  
y = data['Star Rating']                     

# Escalar los datos (K-NN es sensible a la escala)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3. Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Crear y entrenar 
knn = KNeighborsClassifier(
    n_neighbors=5,      
    metric='euclidean'   
)
knn.fit(X_train, y_train)

#Evaluar 
y_pred = knn.predict(X_test)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))


plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.title('Distribución de reviews por Word Count y Sentiment Value')
plt.xlabel('Word Count (escalado)')
plt.ylabel('Sentiment Value (escalado)')
plt.colorbar(label='Star Rating')
plt.show()

# 7. Función para encontrar el mejor k (opcional)
def find_best_k(max_k=20):
    k_values = range(1, max_k+1)
    accuracies = []
    
    for k in k_values:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        knn_temp.fit(X_train, y_train)
        accuracies.append(knn_temp.score(X_test, y_test))
    
    plt.plot(k_values, accuracies)
    plt.xlabel('Valor de k')
    plt.ylabel('Precisión')
    plt.title('Selección del mejor k')
    plt.show()
    
    best_k = k_values[np.argmax(accuracies)]
    print(f"Mejor valor de k: {best_k} con precisión {max(accuracies):.2f}")
    return best_k

best_k = find_best_k()