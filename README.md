# 🧠 MLOps End-to-End Demo — House Price Prediction

## Overview
This project demonstrates essential **MLOps concepts** using a simple ML pipeline for housing price prediction.  
It integrates **Streamlit**, **DVC**, **MLflow**, **MinIO (S3)**, and **Kubernetes (Minikube)** for full lifecycle management.

---

## 🔧 Local Development

### Run the Streamlit App
```bash
streamlit run app.py
```

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


TODO: 
Missing critical components preventing full MLOps workflow:

1. **MLflow Tracking Integration**

   * Server setup script or Docker service (currently only manual command).
   * `mlflow.set_tracking_uri()` or `.env` configuration in code.
   * No experiment registration or logging structure (`mlflow.log_params`, `mlflow.log_metrics`).

2. **Model Registry / Artifact Management**

   * MinIO config exists, but not used programmatically.
   * Missing code to log model artifacts to MLflow (`mlflow.sklearn.log_model`).

3. **CI/CD Pipeline**

   * No automated test runner (GitHub Actions or GitLab CI).
   * No image build–push–deploy pipeline.

4. **Kubernetes Integration**

   * Working manifest but no Ingress or persistent storage.
   * No Helm chart or namespace configuration.

5. **Prediction API / Streamlit separation**

   * API for inference (`FastAPI`) replaced by Streamlit UI.
   * Need REST endpoint for programmatic access (model monitoring, A/B tests).

6. **Unit and Integration Tests**

   * Tests exist but cover only API validation.
   * No training pipeline or MLflow logging tests.

7. **Environment Definition**

   * Missing `.env` and `.env.example` for credentials (MinIO, MLflow, DB).
   * No `Makefile` for standardizing workflow (`make train`, `make deploy`).

8. **Documentation**

   * README lacks architecture diagram, environment variable list, and deployment flow diagram.

9. **Monitoring / Logging**

   * No Prometheus/Grafana integration or even basic Streamlit logs persisted.

10. **Reproducibility**

* No `dvc.yaml` defining training pipeline stages (`prepare`, `train`, `evaluate`).
* No versioning for datasets or models beyond a single `.dvc` file.

Your setup runs but isn’t production-ready — missing automation, configuration abstraction, and experiment traceability.

