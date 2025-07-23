# src/train.py
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
import mlflow


# Enable autologging for scikit-learn
mlflow.sklearn.autolog()
# Load and preprocess the data
data = pd.read_csv("data/housing.csv")
data = data.replace({'yes': 1, 'no': 0})
data["furnishingstatus"] = data["furnishingstatus"].map({
    "furnished": 1,
    "semi-furnished": 0.5,
    "unfurnished": 0
})
# convert all columns to float
data = data.astype(float)
X = data.drop("price", axis=1)
y = data["price"]
# Get the format of the data
signature = infer_signature(X, y)
print(f"Data signature: {signature}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train your model - MLflow automatically logs everything
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, "model.pkl")

    # Evaluation metrics are automatically captured
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    mlflow.log_metric("score", test_score)

    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")