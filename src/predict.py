import joblib
import pandas as pd

def predict(input_data):
    feature_order = [
        "area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom",
        "basement", "hotwaterheating", "airconditioning", "parking", "prefarea",
        "furnishingstatus"
    ]
    # Validate input keys
    if not all(key in input_data for key in feature_order):
        raise KeyError("Missing required input features")
    # Validate negative values
    for key, value in input_data.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"Feature '{key}' must be a number")
        if value < 0:
            raise ValueError(f"Feature '{key}' cannot be negative")
    
    # Create DataFrame with the correct feature order
    df = pd.DataFrame([input_data], columns=feature_order)
    try:
        model = joblib.load("model.pkl")
        prediction = model.predict(df)
        return float(prediction[0])
    except FileNotFoundError:
        raise FileNotFoundError("Model file not found")
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")