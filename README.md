# 🏥 Chronic Disease Predictor

> An end-to-end machine learning system that predicts chronic disease indicators using CDC data, deployed on Azure with automated CI/CD.

[![Build Status](https://github.com/ackben0226/chronic-disease-predictor-mlops/actions/workflows/deploy.yml/badge.svg)](https://github.com/ackben0226/chronic-disease-predictor-mlops/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

---

## 📖 What is this?

This is a **production-grade ML system** that:

1. **Ingests** CDC Chronic Disease Indicators data (400K+ rows)
2. **Validates** data quality (schema, ranges, completeness)
3. **Cleans** missing values, duplicates, and outliers
4. **Engineers** 58 features for machine learning
5. **Trains** a Random Forest model (R² = 0.63)
6. **Serves** predictions via a secure REST API
7. **Deploys** automatically to Azure via CI/CD
8. **Monitors** performance and detects data drift

### 🎯 Why This Matters

Healthcare organizations need fast, accurate predictions for:
- **Resource allocation** — Direct funding to high-risk regions
- **Intervention planning** — Identify at-risk populations
- **Policy decisions** — Data-driven public health strategies

This system provides predictions in **~134ms**, enabling real-time decision-making at scale.

---

## ✨ Features

### 📊 Data Pipeline
- ✅ Schema validation with per-metric-type bounds
- ✅ Smart imputation (median by DataValueType)
- ✅ IQR-based outlier detection
- ✅ 58 engineered features (demographics, geography, temporal)

### 🧠 Machine Learning
- ✅ Random Forest Regressor (R² = 0.63)
- ✅ 3-fold cross-validation
- ✅ Hyperparameter tuning (RandomizedSearchCV)
- ✅ Feature importance analysis
- ✅ Confidence intervals

### 🔌 API
- ✅ FastAPI with auto-generated Swagger UI
- ✅ Single + batch predictions
- ✅ API key authentication (X-API-Key header)
- ✅ Pydantic validation
- ✅ Health and readiness checks

### 🚀 DevOps
- ✅ Docker containerization
- ✅ Azure Container Registry (ACR)
- ✅ Azure Container Instances (ACI)
- ✅ GitHub Actions CI/CD
- ✅ Azure Key Vault for secrets

### 📈 Monitoring
- ✅ Application Insights integration
- ✅ Custom data drift detection
- ✅ Structured logging
- ✅ Prediction tracking

---

## 🏗️ Architecture
```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM ARCHITECTURE                                │
└──────────────────────────────────────────────────────────────────────────────┘

                                     CLIENT
                                        │
                                        │ HTTPS + X-API-Key
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                                AZURE CLOUD                                   │
│                                                                              │
│   ┌──────────────────┐              ┌──────────────────┐                    │
│   │ Azure Container  │              │ Azure Key Vault  │                    │
│   │ Registry (ACR)   │              │   (API Keys)     │                    │
│   └────────┬─────────┘              └────────┬─────────┘                    │
│            │ Docker Image                    │ Secrets                       │
│            └──────────────────┬──────────────┘                               │
│                               ▼                                              │
│              ┌─────────────────────────────────────┐                         │
│              │ Azure Container Instance (ACI)      │                         │
│              │                                     │                         │
│              │  ┌───────────────────────────────┐  │                         │
│              │  │ FastAPI Application           │  │                         │
│              │  │ + Random Forest Model         │  │                         │
│              │  │                               │  │                         │
│              │  │ • /predict                    │  │                         │
│              │  │ • /health                     │  │                         │
│              │  │ • /monitoring                 │  │                         │
│              │  └───────────────────────────────┘  │                         │
│              └──────────────────┬──────────────────┘                         │
│                                 │                                            │
│                                 ▼                                            │
│              ┌─────────────────────────────────────┐                         │
│              │ Azure Application Insights          │                         │
│              │ Metrics • Logs • Monitoring • Alerts│                         │
│              └─────────────────────────────────────┘                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

                                   DEVELOPER
                                       │
                                       │ git push
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            GITHUB ACTIONS CI/CD                              │
│                                                                              │
│      1. Run Tests  ─────▶  2. Build Image  ─────▶  3. Deploy to ACI         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

```
![System Architecture](docs/images/mermaid-diagram.png)
---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Language** | Python 3.10 | Core language |
| **Data** | pandas, numpy | Data manipulation |
| **ML** | scikit-learn | Random Forest model |
| **API** | FastAPI, Pydantic | REST endpoints |
| **Server** | Uvicorn | ASGI server |
| **Container** | Docker | Packaging |
| **Registry** | Azure ACR | Image storage |
| **Compute** | Azure ACI | Container hosting |
| **Secrets** | Azure Key Vault | Secure secrets |
| **Monitoring** | Application Insights | Observability |
| **CI/CD** | GitHub Actions | Automation |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker Desktop
- Git
- (Optional) Azure account for deployment

---

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ackben0226/chronic-disease-predictor-mlops.git
cd chronic-disease-predictor-mlops

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download data
# Place Chronic_Disease_Indicators.csv in data/raw/

# 5. Train the model
python main.py

# 6. Run the API locally
python -m deployment.app.main

# 7. Test in browser
# Open http://localhost:8000/docs
```
---

### Docker
```bash
# Build
docker build -t health-predictor:v1.0 .

# Run
docker run -p 8000:8000 health-predictor:v1.0

# Test
curl http://localhost:8000/health
```
---
### 📖 Usage
```bash
Making a Prediction
bash
curl -X POST http://YOUR_IP:8000/predict \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "YearStart": 2020,
    "Latitude": 36.1162,
    "Longitude": -119.6816,
    "Demo_Overall": 1,
    "Region_West": 1,
    "Topic_Diabetes": 1
  }'
```
---
### Response:
```json
{
  "prediction": 44.88,
  "prediction_rounded": 44.88,
  "confidence_lower": -100.09,
  "confidence_upper": 189.85,
  "model_version": "1.0.0",
  "model_type": "RandomForestRegressor",
  "processing_time_ms": 134.65,
  "timestamp": "2026-09-12T17:12:28.872227"
}
```
---
### Python Client
```python
import requests

response = requests.post(
    "http://YOUR_IP:8000/predict",
    headers={"X-API-Key": "your-key"},
    json={
        "YearStart": 2020,
        "Latitude": 36.1162,
        "Longitude": -119.6816,
        "Demo_Overall": 1,
        "Region_West": 1,
        "Topic_Diabetes": 1
    }
)
print(response.json())
```
---

### 📁 Project Structure
```text
chronic-disease-predictor-mlops/
│
├── 📁 src/                          # Core ML pipeline
│   ├── ingestion/                   # Data loading & validation
│   ├── preprocessing/               # Cleaning & features
│   └── models/                      # Model training
│
├── 📁 deployment/                   # API & deployment
│   └── app/                         # FastAPI application
│       ├── main.py                  # App entry point
│       ├── routes.py                # Endpoints
│       ├── schemas.py               # Request/response models
│       ├── security.py              # API key auth
│       ├── dependencies.py          # Model loading
│       └── config.py                # Configuration
│
├── 📁 monitoring/                   # Monitoring tools
│   ├── drift_detector.py            # Data drift detection
│   └── alerts.py                    # Alerting system
│
├── 📁 tests/                        # Test suite
│   ├── test_drift_detector.py
│   ├── test_data_loading.py
│   └── ...
│
├── 📁 docs/                         # Documentation
│   ├── architecture.md
│   ├── setup.md
│   ├── api.md
│   └── ...
│
├── 📁 models/                       # Saved models (gitignored)
├── 📁 data/                         # Data files (gitignored)
├── 📁 logs/                         # Logs (gitignored)
│
├── 📄 main.py                       # Training pipeline
├── 📄 Dockerfile                    # Container definition
├── 📄 requirements.txt              # Dependencies
├── 📄 README.md                     # This file
├── 📄 CHANGELOG.md                  # Version history
└── 📄 LICENSE                       # MIT License
```
---
### 🔌 API Documentation
__Endpoints__
|Method	|Endpoint	|Description	|Auth Required|
|-------|---------|-------------|-------------|
|GET	|/health	|Basic health check	|❌|
|GET	|/info	|Model information|	❌|
|GET	|/features	|List of 58 features	|❌|
|POST	|/predict	|Single prediction|	✅|
|POST	|/predict_batch|	Batch predictions|	✅|
|GET	|/monitoring/health|	Detailed health|	❌|
|GET	|/monitoring/drift-report	|Drift status	|❌|
|GET	|/health/liveness|	Liveness probe	|❌|
|GET	|/health/readiness|	Readiness probe	|❌|

### Interactive API Docs
Visit: http://127.0.0.1:8000/docs for Swagger UI

### 🧪 Testing
```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov=deployment --cov=monitoring

# Specific test
pytest tests/test_drift_detector.py -v

# Skip slow tests
pytest tests/ -m "not slow"
```
---
### 🚢 Deployment
__Automatic (CI/CD)__
```bash
# Just push to main — GitHub Actions handles everything
git push origin main
```
---
__Pipeline stages:__

- ✅ Run tests (30s)
- ✅ Build Docker image (2 min)
- ✅ Push to Azure ACR (1 min)
- ✅ Deploy to Azure ACI (2 min)
- ✅ Verify with health check (1 min)

---
__Total time:__ ~5-8 minutes

### Manual Deployment
```bash
# 1. Build Docker image
docker build -t health-predictor:v1.0 .

# 2. Push to Azure Container Registry
az acr login --name mlhealthacr
docker tag health-predictor:v1.0 mlhealthacr.azurecr.io/health-predictor:v1.0
docker push mlhealthacr.azurecr.io/health-predictor:v1.0

# 3. Deploy to Azure Container Instances
az container create \
  --resource-group ml-health-predictor \
  --name health-predictor-api \
  --image mlhealthacr.azurecr.io/health-predictor:v1.0 \
  --dns-name-label health-predictor \
  --ports 8000 \
  --cpu 2 \
  --memory 4
```
---

### 📊 Monitoring
Application Insights
Monitor your deployed API at Azure Portal

Key metrics tracked:
- __Requests per minute — API usage patterns__
- __Response time — Performance tracking__
- __Error rate — Health indicator__
- __CPU/Memory — Resource usage__

### Custom Monitoring
- Drift Report: ```GET /monitoring/drift-report```
- Prediction Logs: ```logs/predictions.log```
- Alert Logs: ```logs/alerts.log```
---
### 📈 Model Performance
|Metric|	Value|	Interpretation|
|----|----|----|
|R²	|0.63	|Explains 63% of variance|
|RMSE	|692.38	|Average prediction error|
|MAE	|161.81	|Mean absolute error|
|Features	|58	|Input dimensions|
|Training rows	|282,728	|Training samples|


### 🤝 Contributing
Contributions are welcome! Please see [CONTRIBUTING.md](contributing.md) for guidelines.

### Development Setup
```bash
# 1. Fork the repository
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/chronic-disease-predictor-mlops.git

# 3. Create a feature branch
git checkout -b feature/amazing-feature

# 4. Make changes and test
pytest tests/ -v

# 5. Commit and push
git add .
git commit -m "Add amazing feature"
git push origin feature/amazing-feature

# 6. Open a Pull Request
```
---
### 📄 License
This project is licensed under the MIT License — see LICENSE file for details.

### 👤 Author
Benjamin Ackah

GitHub: [@ackben0226](https://github.com/ackben0226)

Project: [chronic-disease-predictor-mlops](https://github.com/ackben0226/chronic-disease-predictor-mlops)

### 🙏 Acknowledgments
- CDC for the Chronic Disease Indicators dataset
- Microsoft Azure for cloud infrastructure
- FastAPI for the excellent web framework
- scikit-learn for the ML library

### 📚 Additional Documentation
- [Architecture](docs/architecture.md)
- [Setup Guide](docs/setup.md)
- [API Reference](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Data Documentation](docs/data.md)
- [Model Documentation](docs/model.md)
- [Monitoring](docs/monitoring.md)
- [Troubleshooting](docs/troubleshooting.md)

<div align="center">
⭐ If this project helped you, please give it a star! ⭐

Made with ❤️ using Python, FastAPI, and Azure

</div> ```
