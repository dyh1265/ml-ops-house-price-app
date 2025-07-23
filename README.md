Test code for MLOPS concepts
streamlit run app.py
pytest tests/
docker build -t house-price-app .
docker run -p 8000:8000 house-price-app
