"""
Script de entrenamiento de modelo de detección de fraude.
Carga datos desde CSV, entrena LogisticRegression y registra en MLflow.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    classification_report
)
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

print("🚀 Iniciando entrenamiento del modelo de detección de fraude...")

# ============================================================
# CONFIGURACIÓN DE RUTAS Y MLFLOW
# ============================================================

workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = f"file://{os.path.abspath(mlruns_dir)}"

print(f"📁 Workspace: {workspace_dir}")
print(f"📁 MLRuns: {mlruns_dir}")
print(f"📁 Tracking URI: {tracking_uri}")

os.makedirs(mlruns_dir, exist_ok=True)
mlflow.set_tracking_uri(tracking_uri)

# ============================================================
# CREAR O USAR EXPERIMENTO
# ============================================================

experiment_name = "fraud-detection-pipeline"
try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=tracking_uri
    )
    print(f"✅ Experimento '{experiment_name}' creado con ID: {experiment_id}")
except mlflow.exceptions.MlflowException as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"✅ Usando experimento existente '{experiment_name}' con ID: {experiment_id}")
    else:
        print(f"❌ Error creando experimento: {e}")
        sys.exit(1)

# ============================================================
# CARGAR Y PREPARAR DATOS
# ============================================================

def load_data(filepath):
    """
    Carga el dataset desde un archivo CSV.
    
    Args:
        filepath (str): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: Dataset cargado
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"📊 Datos cargados: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    return df

def prepare_features(df):
    """
    Prepara las features para entrenamiento.
    Separa X (features) e y (target) y escala Time y Amount.
    
    Args:
        df (pd.DataFrame): Dataset completo
        
    Returns:
        tuple: (X_scaled, y, scaler)
    """
    # Separar features y target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Escalar solo Time y Amount (V1-V28 ya están normalizadas por PCA)
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
    
    print(f"📦 Features: {X_scaled.shape[1]}")
    print(f"🎯 Target distribution:")
    print(f"   - Legítimas (0): {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.2f}%)")
    print(f"   - Fraudes (1): {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.2f}%)")
    
    return X_scaled, y, scaler

# Cargar datos de entrenamiento
print("\n📥 Cargando datos de entrenamiento...")
train_df = load_data('data/train_data.csv')

# Preparar features
X_train, y_train, scaler = prepare_features(train_df)

# ============================================================
# ENTRENAR MODELO
# ============================================================

def train_model(X, y, class_weight='balanced'):
    """
    Entrena un modelo de regresión logística.
    
    Args:
        X (pd.DataFrame): Features de entrenamiento
        y (pd.Series): Target de entrenamiento
        class_weight (str): Estrategia para balancear clases
        
    Returns:
        LogisticRegression: Modelo entrenado
    """
    print(f"\n🤖 Entrenando modelo Logistic Regression...")
    print(f"   - Solver: liblinear")
    print(f"   - Class weight: {class_weight}")
    
    model = LogisticRegression(
        max_iter=1000,
        solver='liblinear',
        class_weight=class_weight,  # Importante para datos desbalanceados
        random_state=42
    )
    model.fit(X, y)
    
    return model

model = train_model(X_train, y_train)

# ============================================================
# EVALUAR MODELO
# ============================================================

def evaluate_model(model, X, y):
    """
    Evalúa el modelo y calcula múltiples métricas.
    
    Args:
        model: Modelo entrenado
        X (pd.DataFrame): Features de prueba
        y (pd.Series): Target real
        
    Returns:
        dict: Diccionario con todas las métricas
    """
    print(f"\n📊 Evaluando modelo...")
    
    # Predicciones
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Calcular métricas
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1_score': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred_proba)
    }
    
    # Mostrar resultados
    print(f"   ✅ Accuracy: {metrics['accuracy']:.4f}")
    print(f"   ✅ Precision: {metrics['precision']:.4f}")
    print(f"   ✅ Recall: {metrics['recall']:.4f}")
    print(f"   ✅ F1-Score: {metrics['f1_score']:.4f}")
    print(f"   ✅ ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Reporte de clasificación
    print(f"\n📋 Classification Report:")
    print(classification_report(y, y_pred, target_names=['Legítima', 'Fraude']))
    
    return metrics, y_pred_proba

metrics, y_pred_proba = evaluate_model(model, X_train, y_train)

# ============================================================
# REGISTRAR EN MLFLOW
# ============================================================

print(f"\n💾 Registrando modelo en MLflow...")

try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"📝 Run ID: {run_id}")
        
        # Registrar parámetros
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("solver", "liblinear")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples_train", X_train.shape[0])
        
        # Registrar métricas
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("precision", metrics['precision'])
        mlflow.log_metric("recall", metrics['recall'])
        mlflow.log_metric("f1_score", metrics['f1_score'])
        mlflow.log_metric("roc_auc", metrics['roc_auc'])
        
        # Inferir signature del modelo
        signature = infer_signature(X_train, y_pred_proba)
        
        # Input example (primera fila)
        input_example = X_train.iloc[:1]
        
        # Registrar modelo con signature e input_example
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )
        
        print(f"   ✅ Modelo registrado con signature e input_example")
        print(f"   ✅ Parámetros: 6 registrados")
        print(f"   ✅ Métricas: 5 registradas")
        
except Exception as e:
    print(f"❌ Error durante el registro en MLflow: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 Entrenamiento completado exitosamente!")
print(f"🔗 Ver resultados: mlflow ui")