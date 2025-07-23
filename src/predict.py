import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")

def predict(df):
    # Use the DataFrame directly
    prediction = model.predict(df)
    return prediction[0]  # Return the first (and only) prediction