# regresion_lineal_multiple.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Cargar datos (reemplaza con tu archivo real)
# filtered_data = pd.read_csv('tus_datos.csv')

# Datos de ejemplo (similares a los resultados que mencionaste)
data = {
    'Word count': [500, 800, 1200, 650, 900, 1100, 750],
    '# of Links': [3, 5, 8, 2, 6, 7, 4],
    '# of comments': [2, 1, 5, 0, 3, 4, 1],
    '# Images video': [1, 2, 3, 1, 2, 3, 0],
    '# Shares': [1500, 2500, 1800, 1200, 3000, 2100, 1600]
}
filtered_data = pd.DataFrame(data)

# 2. Crear variable combinada
filtered_data['suma'] = (filtered_data["# of Links"] + 
                         filtered_data['# of comments'].fillna(0) + 
                         filtered_data['# Images video'])

# 3. Preparar datos para el modelo
X = filtered_data[['Word count', 'suma']]
y = filtered_data['# Shares']

# 4. Crear y entrenar modelo
modelo = LinearRegression()
modelo.fit(X, y)

# 5. Realizar predicciones
y_pred = modelo.predict(X)

# 6. Calcular métricas
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# 7. Mostrar resultados
print("\n=== RESULTADOS DEL MODELO ===")
print(f"Coeficientes: {modelo.coef_}")
print(f"Intercepto: {modelo.intercept_}")
print(f"\nMSE: {mse:.2f}")
print(f"R²: {r2:.2f}")

# 8. Generar gráfico 3D (opcional)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Puntos reales
ax.scatter(X['Word count'], X['suma'], y, c='blue', label='Datos reales')

# Plano de predicción
x1_range = np.linspace(X['Word count'].min(), X['Word count'].max(), 10)
x2_range = np.linspace(X['suma'].min(), X['suma'].max(), 10)
X1, X2 = np.meshgrid(x1_range, x2_range)
Y_pred = modelo.intercept_ + modelo.coef_[0]*X1 + modelo.coef_[1]*X2
ax.plot_surface(X1, X2, Y_pred, alpha=0.5, color='red')

ax.set_xlabel('Word count')
ax.set_ylabel('Suma de features')
ax.set_zlabel('# Shares')
ax.set_title('Regresión Lineal Múltiple en 3D')
plt.legend()
plt.savefig('grafico_resultados.png', dpi=300)
print("\nGráfico guardado como 'grafico_resultados.png'")

# 9. Exportar resultados para LaTeX
with open('resultados.tex', 'w') as f:
    f.write(f"% Resultados generados automáticamente\n")
    f.write(f"\\newcommand{{\\coefWord}}{{{modelo.coef_[0]:.4f}}}\n")
    f.write(f"\\newcommand{{\\coefSuma}}{{{modelo.coef_[1]:.4f}}}\n")
    f.write(f"\\newcommand{{\\intercepto}}{{{modelo.intercept_:.2f}}}\n")
    f.write(f"\\newcommand{{\\mse}}{{{mse:.2f}}}\n")
    f.write(f"\\newcommand{{\\rdo}}{{{r2:.2f}}}\n")

print("\nResultados exportados para LaTeX (resultados.tex)")