# 🧠 MLOps End-to-End Demo — House Price Prediction

## Overview
This project demonstrates essential **MLOps concepts** using a simple ML pipeline for housing price prediction.  
It integrates **Streamlit**, **DVC**, **MLflow**, **MinIO (S3)**, and **Kubernetes (Minikube)** for full lifecycle management.

---

## ⚙️ Configuration

### Environment Variables
The project uses a `.env` file for configuration. Key variables include:

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | MLflow tracking server URL |
| `MLFLOW_S3_ENDPOINT_URL` | `http://localhost:9000` | MinIO S3-compatible storage endpoint |
| `AWS_ACCESS_KEY_ID` | `admin` | MinIO access key |
| `AWS_SECRET_ACCESS_KEY` | `admin123` | MinIO secret key |
| `MLFLOW_EXPERIMENT_NAME` | `Housing_Price_Prediction` | MLflow experiment name |
| `MLFLOW_MODEL_NAME` | `HousingPriceModel` | MLflow registered model name |

---

## 🔧 Local Development

### Run the Streamlit App
```bash
streamlit run src/app.py
```

### Train the Model
```bash
python src/train.py
```
This will:
- Load and preprocess the housing dataset
- Train a Linear Regression model
- Log metrics, parameters, and artifacts to MLflow
- Register the model in MLflow Model Registry
- Save the model as `model.pkl`

### Run Tests
```bash
pytest tests/
```

### Build & Run with Docker
```bash
docker build -t house-price-app .
docker run -p 8501:8501 house-price-app
```

---

## 📦 Data Versioning with DVC

```bash
dvc init
dvc add data/housing.csv
git add data/housing.csv.dvc .gitignore
git commit -m "Add housing.csv to DVC"
```

---

## ⚙️ MLflow Tracking Server Setup

Kill any process using port 5000:
```bash
sudo fuser -k 5000/tcp
```

Start MLflow:
```bash
mlflow server   --backend-store-uri sqlite:///mlflow.db   --default-artifact-root s3://mlflow-artifacts   --host 0.0.0.0   --port 5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

---

## ☁️ Local S3 Storage via MinIO

MinIO is an open-source S3-compatible object storage system.  
It enables local MLflow artifact storage without AWS.

### Docker Compose Setup
```bash
docker compose up -d
docker compose down
```

### MinIO Configuration
```bash
mc alias set local http://localhost:9000 admin admin123
mc mb local/mlflow-artifacts
mc ls local
```

MinIO Dashboard → `http://localhost:9001`  
Credentials → `admin / admin123`

---

## ☸️ Kubernetes Deployment with Minikube



### Install Dependencies
```bash
sudo apt update
sudo apt install -y curl apt-transport-https conntrack virtualbox virtualbox-ext-pack
```

### Install Minikube
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

### Install kubectl
```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install kubectl /usr/local/bin/
```

### Start Minikube
# first create a new user
adduser [username]
su - [username]
sudo usermod -aG docker $USER && newgrp docker

```bash
minikube start --driver=docker --memory=2000m --apiserver-ips=0.0.0.0 --apiserver-port=8443
```

---

## 🧱 Build and Deploy Inside Minikube

### 1. Build Docker Image
```bash
eval $(minikube docker-env)
docker build -t house-price-app:latest .
```

### 2. Deploy Kubernetes Manifest
```bash
kubectl apply -f k8s-deployment.yaml
```

### 3. Check Pods
```bash
kubectl get pods
```

### 4. Access Streamlit Service
```bash
minikube service house-price-service
```

If running on a remote server:
```bash
kubectl port-forward service/house-price-service 8501:80
# Then open http://localhost:8501
```

---

## 🧹 Cleanup
Delete all Minikube resources:
```bash
minikube delete --profile=minikube
```

---

## 📁 Project Components

| Component     | Tool/Service      | Purpose |
|----------------|-------------------|----------|
| Web Interface  | Streamlit         | Interactive ML model UI |
| Model Tracking | MLflow + MinIO    | Track experiments & store artifacts |
| Data Versioning| DVC               | Track datasets & model inputs |
| Deployment     | Docker + Kubernetes | Scalable app hosting |

---

## ✅ Summary
This setup demonstrates a full **MLOps lifecycle**:  
data → model → tracking → packaging → orchestration — all reproducible on local infrastructure.



---

## ✨ Implemented Features

This project includes the following MLOps components:

### 🔬 Experiment Tracking
- **MLflow Integration**: Full experiment tracking with metrics, parameters, and artifacts
  - Configured via `.env` file for easy environment management
  - Automatic model logging with `mlflow.sklearn.autolog()`
  - Model registry for versioning (`HousingPriceModel`)
  - Dataset versioning tracked via DVC and Git commit hashes

### �� Data & Model Versioning
- **DVC Pipeline**: Configured in `dvc.yaml` with training stage
- **MinIO S3 Storage**: Local S3-compatible storage for MLflow artifacts
- **Git Integration**: Version control for code and DVC metadata

### 🔄 CI/CD Pipeline
- **GitHub Actions Workflow**: Automated testing and deployment
  - Linting with `flake8`
  - Unit tests with `pytest`
  - Docker image build and push
  - Kubernetes deployment automation

### 🧪 Testing
- **Unit Tests**: Comprehensive test coverage for prediction logic
- **Integration Tests**: Model training validation
- Located in `tests/` directory

### 🚀 Deployment
- **Docker**: Containerized Streamlit application
- **Kubernetes**: Production-ready deployment with 3 replicas
- **Load Balancer**: Service configuration for external access

### 🎯 Web Interface
- **Streamlit App**: Interactive UI for house price predictions
- Real-time predictions using trained ML model
- User-friendly input forms for all features

---

## 🔮 Future Enhancements

Potential improvements for production readiness:

- **Monitoring**: Add Prometheus/Grafana for metrics and observability
- **API Layer**: Separate REST API (FastAPI) for programmatic access
- **Advanced K8s**: Ingress configuration, persistent volumes, Helm charts
- **Extended Testing**: Training pipeline tests, MLflow integration tests
- **Documentation**: Architecture diagrams, API documentation
