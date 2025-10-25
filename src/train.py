import os
import subprocess
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv


# --- Environment setup ---
load_dotenv()

os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"))
os.environ.setdefault("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID", "admin"))
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", "admin123"))

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "Housing_Price_Prediction"))


# --- DVC version utility ---
def get_dvc_version(file_path: str) -> str:
    """Return Git commit hash for a DVC-tracked dataset, or 'unknown' if unavailable."""
    dvc_file = f"{file_path}.dvc"
    if not os.path.exists(dvc_file):
        return "no_dvc_file"
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=%H", "--", dvc_file],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# --- Data load and preprocessing ---
DATA_PATH = "data/housing.csv"
data = pd.read_csv(DATA_PATH)
data = data.replace({"yes": 1, "no": 0}).infer_objects(copy=False)
data["furnishingstatus"] = data["furnishingstatus"].map(
    {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
)
data = data.astype(float)

# --- Train/test split ---
X = data.drop("price", axis=1)
y = data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Metadata ---
dataset_version = get_dvc_version(DATA_PATH)
hyperparams = {"fit_intercept": True}


# --- MLflow run ---
with mlflow.start_run(run_name="Linear_Regression_Run"):
    mlflow.sklearn.autolog(log_models=True)
    mlflow.log_params(hyperparams)
    mlflow.set_tags({
        "framework": "scikit-learn",
        "model_type": "LinearRegression",
        "dataset": os.path.basename(DATA_PATH),
        "dataset_version": dataset_version,
        "encoding": "manual_binary"
    })

    # Save split data as artifacts
    train_csv, test_csv = "train_split.csv", "test_split.csv"
    pd.concat([X_train, y_train], axis=1).to_csv(train_csv, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_csv, index=False)
    for f in (DATA_PATH, train_csv, test_csv):
        mlflow.log_artifact(f)

    # Train
    model = LinearRegression(**hyperparams)
    model.fit(X_train, y_train)

    # Evaluate
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    mlflow.log_metrics({"train_r2": train_r2, "test_r2": test_r2})

    # Persist model
    model_path = "model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    # Register model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    registered = mlflow.register_model(model_uri, "HousingPriceModel")

    print(f"Training R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    print(f"Registered version: {registered.version}")
    print(f"DVC dataset version: {dataset_version}")

# --- Cleanup ---
for f in (train_csv, test_csv):
    if os.path.exists(f):
        os.remove(f)
