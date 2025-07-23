Test code for MLOPS concepts
streamlit run app.py
pytest tests/
docker build -t house-price-app .
docker run -p 8000:8000 house-price-app

# How to initialize the dvc
dvc init
dvc add data/housing.csv
git add data/housing.csv.dvc .gitignore
git commit -m "Add housing.csv to DVC"