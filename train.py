import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys

print("🚀 Iniciando entrenamiento del modelo...")

# Configurar rutas
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = f"file://{os.path.abspath(mlruns_dir)}"

print(f"📁 Workspace: {workspace_dir}")
print(f"📁 MLRuns: {mlruns_dir}")
print(f"📁 Tracking URI: {tracking_uri}")

# Crear directorio si no existe
os.makedirs(mlruns_dir, exist_ok=True)

# Configurar MLflow
mlflow.set_tracking_uri(tracking_uri)

# Crear o usar experimento
experiment_name = "CI-CD-Lab2"
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

# Cargar datos
print("📊 Cargando dataset de diabetes...")
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"📦 Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Entrenar modelo
print("🤖 Entrenando modelo de regresión lineal...")
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluar
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print(f"🎯 MSE: {mse:.4f}")

# Registrar en MLflow
try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"📝 Run ID: {run_id}")
        
        mlflow.log_metric("mse", mse)
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.sklearn.log_model(model, "model")
        
        print("💾 Modelo y métricas registrados en MLflow")
        
except Exception as e:
    print(f"❌ Error durante el registro en MLflow: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("✅ Entrenamiento completado con éxito!")