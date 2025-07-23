import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

# Set MLflow experiment for tracking
mlflow.set_experiment("Housing_Price_Prediction")

# Load and preprocess the data
data = pd.read_csv("data/housing.csv")
data = data.replace({'yes': 1, 'no': 0})
data["furnishingstatus"] = data["furnishingstatus"].map({
    "furnished": 2,
    "semi-furnished": 1,
    "unfurnished": 0
})
# Convert all columns to float explicitly to avoid downcasting issues
data = data.astype(float)

# Separate features and target
X = data.drop("price", axis=1)
y = data["price"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save train and test splits to CSV files
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
train_data.to_csv("train_split.csv", index=False)
test_data.to_csv("test_split.csv", index=False)

# Define hyperparameters
hyperparams = {
    "fit_intercept": True
}

# Start MLflow run for experiment tracking
with mlflow.start_run(run_name="Linear_Regression_Run"):
    # Enable autologging for scikit-learn
    mlflow.sklearn.autolog(log_models=True)
    
    # Log hyperparameters
    mlflow.log_params(hyperparams)
    
    # Set tags for additional metadata
    mlflow.set_tags({
        "model_type": "LinearRegression",
        "dataset": "housing.csv",
        "preprocessor": "manual_encoding"
    })
    
    # Log the dataset and splits as artifacts
    mlflow.log_artifact("data/housing.csv")
    mlflow.log_artifact("train_split.csv")
    mlflow.log_artifact("test_split.csv")
    
    # Train the model
    model = LinearRegression(**hyperparams)
    model.fit(X_train, y_train)
    
    # Save the model locally
    model_path = "model.pkl"
    joblib.dump(model, model_path)
    
    # Log the model as an artifact
    mlflow.log_artifact(model_path)
    
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    # Log additional metrics
    mlflow.log_metric("train_score", train_score)
    mlflow.log_metric("test_score", test_score)
    
    # Register the model in MLflow Model Registry
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    registered_model = mlflow.register_model(model_uri, "HousingPriceModel")
    
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    print(f"Model registered with version: {registered_model.version}")

# Clean up temporary split files
os.remove("train_split.csv")
os.remove("test_split.csv") 