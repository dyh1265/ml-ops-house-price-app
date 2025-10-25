import streamlit as st
from src.predict import predict

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction App")
st.write("Input house details below to predict the price.")

# Input fields
area = st.number_input("Area (sq units)", min_value=0.0, step=100.0)
bedrooms = st.number_input("Bedrooms", min_value=0.0, step=1.0)
bathrooms = st.number_input("Bathrooms", min_value=0.0, step=1.0)
stories = st.number_input("Stories", min_value=0.0, step=1.0)
mainroad = st.selectbox("Main road", [0, 1])
guestroom = st.selectbox("Guestroom", [0, 1])
basement = st.selectbox("Basement", [0, 1])
hotwaterheating = st.selectbox("Hot water heating", [0, 1])
airconditioning = st.selectbox("Air conditioning", [0, 1])
parking = st.number_input("Parking spaces", min_value=0.0, step=1.0)
prefarea = st.selectbox("Preferred area", [0, 1])
furnishingstatus = st.selectbox(
    "Furnishing status", 
    options=[0, 1, 2],
    format_func=lambda x: ["Unfurnished", "Semi-furnished", "Furnished"][x]
)

# Predict button
if st.button("Predict Price"):
    data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }

    try:
        price = predict(data)
        st.success(f"Predicted house price: ₹{price:,.2f}")
    except FileNotFoundError:
        st.error("Model file not found. Train the model first.")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
