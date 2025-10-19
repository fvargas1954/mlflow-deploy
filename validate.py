import os
import mlflow
import sys
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("🔍 Iniciando validación del modelo...")

# Configurar rutas
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = f"file://{os.path.abspath(mlruns_dir)}"

print(f"📁 Tracking URI: {tracking_uri}")

# Configurar MLflow
mlflow.set_tracking_uri(tracking_uri)

# Obtener experimento
experiment_name = "CI-CD-Lab2"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    print(f"❌ No se encontró el experimento '{experiment_name}'")
    print("Ejecuta train.py primero.")
    sys.exit(1)

# Buscar el último run
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"]
)

if runs.empty:
    print("❌ No hay runs disponibles para validar.")
    sys.exit(1)

# Obtener MSE del último run
mse = runs.iloc[0]["metrics.mse"]

print(f"📊 MSE del modelo: {mse:.4f}")

# Validar umbral
THRESHOLD = 5000.0  # Umbral más alto que en Taller 2

if mse <= THRESHOLD:
    print(f"✅ MSE ({mse:.4f}) es aceptable (< {THRESHOLD})")
    print("✅ El modelo cumple los criterios de calidad")
    sys.exit(0)
else:
    print(f"❌ MSE ({mse:.4f}) supera el umbral ({THRESHOLD})")
    print("❌ Modelo NO apto para producción")
    sys.exit(1)