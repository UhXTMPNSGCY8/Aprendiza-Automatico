# ==============================================================================
# TAREA: EVALUACIÓN Y OPTIMIZACIÓN DE MODELOS DE CLASIFICACIÓN
# ==============================================================================

# --- IMPORTACIÓN DE LIBRERÍAS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay

# --- 1. CARGA DEL DATASET ---
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. ENTRENAMIENTO DE REGRESIÓN LOGÍSTICA ---
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- 3. MATRIZ DE CONFUSIÓN ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(cmap='Blues')
plt.title("Matriz de Confusión")
plt.show()
# NOTA: La matriz muestra los aciertos en la diagonal principal y los errores 
# (falsos positivos/negativos) en las celdas fuera de la diagonal.

# --- 4. CÁLCULO DE MÉTRICAS ---
print("--- MÉTRICAS DE EVALUACIÓN ---")
print(classification_report(y_test, y_pred))
y_prob = model.predict_proba(X_test)[:, 1]
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

# --- 5. K-FOLD CROSS-VALIDATION (k=5 y k=10) ---
cv5 = cross_val_score(model, X, y, cv=5)
cv10 = cross_val_score(model, X, y, cv=10)
print(f"\n--- K-FOLD CROSS-VALIDATION ---")
print(f"K-Fold (k=5) Promedio: {cv5.mean():.4f}")
print(f"K-Fold (k=10) Promedio: {cv10.mean():.4f}")
# Comparación: Un valor de K mayor suele dar una estimación más precisa pero computacionalmente más costosa.

# --- 6. OPTIMIZACIÓN DE HIPERPARÁMETROS ---
params = {'C': [0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs']}

# GridSearchCV (Exhaustivo)
grid = GridSearchCV(LogisticRegression(max_iter=10000), params, cv=5)
grid.fit(X_train, y_train)
print(f"\nMejor configuración GridSearchCV: {grid.best_params_}")

# RandomizedSearchCV (Eficiente)
rand = RandomizedSearchCV(LogisticRegression(max_iter=10000), params, n_iter=5, cv=5)
rand.fit(X_train, y_train)
print(f"Mejor configuración RandomizedSearchCV: {rand.best_params_}")

# --- 7. AJUSTE DEL UMBRAL DE DECISIÓN ---
umbral = 0.3
y_pred_ajustado = (model.predict_proba(X_test)[:, 1] > umbral).astype(int)
print(f"\n--- REPORTE CON UMBRAL AJUSTADO A {umbral} ---")
print(classification_report(y_test, y_pred_ajustado))
# Análisis: Al bajar el umbral, aumentamos la sensibilidad (Recall) a costa de la precisión.