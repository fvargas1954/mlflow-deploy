"""
Script de validación de modelo de detección de fraude.
Carga el modelo desde MLflow y evalúa con datos externos de test.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import mlflow
import mlflow.sklearn

print("🔍 Iniciando validación del modelo...")

# ============================================================
# CONFIGURACIÓN DE MLFLOW
# ============================================================

workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = f"file://{os.path.abspath(mlruns_dir)}"

mlflow.set_tracking_uri(tracking_uri)
print(f"📁 Tracking URI: {tracking_uri}")

# ============================================================
# OBTENER ÚLTIMO MODELO REGISTRADO
# ============================================================

def get_latest_run(experiment_name):
    """
    Obtiene el run más reciente de un experimento.
    
    Args:
        experiment_name (str): Nombre del experimento
        
    Returns:
        str: Run ID del último experimento
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"❌ No se encontró el experimento '{experiment_name}'")
        print("Ejecuta train.py primero.")
        sys.exit(1)
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )
    
    if runs.empty:
        print("❌ No hay runs disponibles para validar.")
        sys.exit(1)
    
    return runs.iloc[0]

experiment_name = "fraud-detection-pipeline"
latest_run = get_latest_run(experiment_name)
run_id = latest_run['run_id']

print(f"✅ Modelo encontrado - Run ID: {run_id}")
print(f"📅 Fecha: {latest_run['start_time']}")

# ============================================================
# CARGAR MODELO DESDE MLFLOW
# ============================================================

print(f"\n📦 Cargando modelo desde MLflow...")

model_uri = f"runs:/{run_id}/model"

try:
    loaded_model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    sys.exit(1)

# ============================================================
# CARGAR DATOS DE TEST EXTERNOS
# ============================================================

def load_test_data(filepath):
    """
    Carga datos de test desde CSV externo.
    
    Args:
        filepath (str): Ruta al archivo CSV de test
        
    Returns:
        tuple: (X_test, y_test)
    """
    if not os.path.exists(filepath):
        print(f"❌ No se encontró el archivo: {filepath}")
        sys.exit(1)
    
    df = pd.read_csv(filepath)
    print(f"📊 Datos de test cargados: {df.shape[0]:,} filas")
    
    # Separar features y target
    X_test = df.drop('Class', axis=1)
    y_test = df['Class']
    
    # Escalar Time y Amount (igual que en entrenamiento)
    scaler = StandardScaler()
    X_test_scaled = X_test.copy()
    X_test_scaled[['Time', 'Amount']] = scaler.fit_transform(X_test[['Time', 'Amount']])
    
    print(f"🎯 Distribución en test:")
    print(f"   - Legítimas (0): {(y_test==0).sum():,} ({(y_test==0).sum()/len(y_test)*100:.2f}%)")
    print(f"   - Fraudes (1): {(y_test==1).sum():,} ({(y_test==1).sum()/len(y_test)*100:.2f}%)")
    
    return X_test_scaled, y_test

print(f"\n📥 Cargando datos de test externos...")
X_test, y_test = load_test_data('data/test_data.csv')

# ============================================================
# HACER PREDICCIONES CON DATOS EXTERNOS
# ============================================================

print(f"\n🔮 Realizando predicciones en datos de test...")

try:
    y_pred = loaded_model.predict(X_test)
    y_pred_proba = loaded_model.predict_proba(X_test)[:, 1]
    print(f"✅ Predicciones completadas")
except Exception as e:
    print(f"❌ Error durante predicciones: {e}")
    sys.exit(1)

# ============================================================
# CALCULAR MÉTRICAS CON DATOS EXTERNOS
# ============================================================

print(f"\n📊 Evaluando desempeño en datos externos...")

# Calcular todas las métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n📈 MÉTRICAS DE VALIDACIÓN:")
print(f"   ✅ Accuracy:  {accuracy:.4f}")
print(f"   ✅ Precision: {precision:.4f}")
print(f"   ✅ Recall:    {recall:.4f}")
print(f"   ✅ F1-Score:  {f1:.4f}")
print(f"   ✅ ROC-AUC:   {roc_auc:.4f}")

# Reporte detallado
print(f"\n📋 Classification Report (Test Data):")
print(classification_report(y_test, y_pred, target_names=['Legítima', 'Fraude']))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print(f"\n🔢 Confusion Matrix:")
print(f"                 Predicted")
print(f"               Leg.  Fraud")
print(f"Actual Leg.  {cm[0][0]:6d} {cm[0][1]:6d}")
print(f"       Fraud {cm[1][0]:6d} {cm[1][1]:6d}")

# ============================================================
# VALIDACIÓN CON UMBRALES
# ============================================================

# Umbrales para validación
THRESHOLD_F1 = 0.10      # F1 mínimo aceptable
THRESHOLD_RECALL = 0.80  # Recall mínimo (importante detectar fraudes)
THRESHOLD_ROC_AUC = 0.90 # ROC-AUC mínimo

print(f"\n🎯 VALIDACIÓN DE UMBRALES:")
print(f"   {'Métrica':<15} {'Valor':<10} {'Umbral':<10} {'Estado'}")
print(f"   {'-'*50}")

checks = [
    ('F1-Score', f1, THRESHOLD_F1),
    ('Recall', recall, THRESHOLD_RECALL),
    ('ROC-AUC', roc_auc, THRESHOLD_ROC_AUC)
]

all_passed = True
for metric_name, value, threshold in checks:
    status = "✅ PASA" if value >= threshold else "❌ FALLA"
    if value < threshold:
        all_passed = False
    print(f"   {metric_name:<15} {value:<10.4f} {threshold:<10.2f} {status}")

# ============================================================
# DECISIÓN FINAL
# ============================================================

print(f"\n{'='*60}")
if all_passed:
    print(f"✅ MODELO VALIDADO EXITOSAMENTE")
    print(f"✅ El modelo cumple todos los criterios de calidad")
    print(f"✅ Apto para promoción a producción")
    sys.exit(0)
else:
    print(f"❌ MODELO NO CUMPLE CRITERIOS")
    print(f"❌ El modelo no alcanza los umbrales mínimos")
    print(f"❌ NO apto para producción - requiere reentrenamiento")
    sys.exit(1)