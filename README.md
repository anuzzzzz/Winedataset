 Wine Classification MLOps Pipeline

A complete production-ready MLOps pipeline for wine cultivar classification using the UCI Wine Dataset, demonstrating ethical AI practices, scalability, and comprehensive monitoring.

## 📋 Project Overview

This project implements an end-to-end MLOps workflow for classifying wine cultivars based on chemical analysis. The solution includes model training, containerization, Kubernetes deployment, fairness analysis, and explainability features.

### Key Features
- **100% Model Accuracy** using Random Forest classifier
- **Kubernetes Auto-scaling** with Horizontal Pod Autoscaler
- **Fairness Analysis** ensuring unbiased predictions across demographic groups
- **Model Explainability** with SHAP analysis focused on Cultivar 2
- **Robustness Testing** against data poisoning attacks
- **Production Monitoring** with OpenTelemetry instrumentation

## 🏗️ Architecture
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  Model Training │───▶│   MLflow        │
│  (Wine Dataset) │    │  & Validation   │    │  Tracking       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
│
▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Docker        │◀───│  Flask API      │───▶│  OpenTelemetry  │
│ Container       │    │  Application    │    │  Monitoring     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
│
▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      HPA        │◀───│   Kubernetes    │───▶│  Load Balancer  │
│   (Auto-scale)  │    │   Deployment    │    │    Service      │
└─────────────────┘    └─────────────────┘    └─────────────────┘

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker
- Kubernetes (kind/minikube/GKE)
- kubectl

### Installation
```bash
# Clone repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
pip install -r requirements.txt

# Start MLflow server
./start_mlflow.sh

# Train models
python src/train_model.py
Deployment
bash# Build Docker image
docker build -t wine-classifier:latest .

# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n wine-classifier
📊 Model Performance
ModelAccuracyPrecisionRecallF1-ScoreRandom Forest100%100%100%100%Logistic Regression97.22%97%97%97%
🔍 Analysis Results
Data Poisoning Robustness

Noise Attacks: No performance degradation up to 50% poison level
Label Flipping: Maintains performance until 50% (drops to 80.56%)
Outlier Injection: Robust against extreme values

Fairness Analysis

Demographic Parity: Minimal difference between location groups
Equalized Odds: Fair treatment across all demographic segments
Bias Detection: No significant location-based bias identified

SHAP Explainability (Cultivar 2 Focus)
Top 5 Most Important Features:

OD280/OD315 ratio (0.1217)
Color intensity (0.1191)
Hue (0.0965)
Flavanoids (0.0884)
Ash content (0.0470)

📁 Project Structure
├── src/
│   ├── train_model.py              # Model training and MLflow logging
│   ├── app.py                      # Flask API application
│   ├── data_poisoning_analysis.py  # Robustness testing
│   ├── fairness_analysis.py        # Bias and fairness assessment
│   └── shap_analysis.py           # Model explainability
├── k8s/
│   ├── deployment.yaml            # Kubernetes deployment
│   ├── service.yaml              # LoadBalancer service
│   └── hpa.yaml                  # Horizontal Pod Autoscaler
├── models/                       # Trained model artifacts
├── data/                        # Dataset files
├── reports/                     # Analysis reports and visualizations
├── .github/workflows/           # CI/CD pipeline
├── Dockerfile                   # Container configuration
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
🔧 API Endpoints

GET /health - Health check
GET /info - API information
POST /predict - Single wine classification
POST /predict/batch - Batch wine classification

Example Usage
bashcurl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "alcohol": 13.0,
    "malic_acid": 2.0,
    "ash": 2.5,
    "alcalinity_of_ash": 20.0,
    "magnesium": 100.0,
    "total_phenols": 2.5,
    "flavanoids": 2.0,
    "nonflavanoid_phenols": 0.3,
    "proanthocyanins": 1.5,
    "color_intensity": 5.0,
    "hue": 1.0,
    "od280/od315_of_diluted_wines": 2.5,
    "proline": 800.0,
    "location": 0
  }'
📈 Monitoring & Observability

MLflow: Experiment tracking and model registry
OpenTelemetry: Distributed tracing
Kubernetes Metrics: Resource utilization monitoring
Health Checks: Liveness and readiness probes

🛡️ Security & Ethics

Fairness Testing: Regular bias assessment across demographic groups
Data Validation: Input sanitization and anomaly detection
Model Interpretability: SHAP explanations for transparency
Robustness Testing: Defense against adversarial attacks

📊 Performance Metrics

Response Time: < 1 second average
Throughput: Handles concurrent requests efficiently
Scalability: Auto-scales from 2-10 pods based on load
Availability: High availability with multi-replica deployment

🔄 CI/CD Pipeline
The GitHub Actions workflow automatically:

Runs unit tests and validation
Builds and pushes Docker images
Deploys to Kubernetes cluster
Generates CML reports for model performance

📚 Key Technologies

ML Framework: scikit-learn, MLflow
API: Flask, Gunicorn
Containerization: Docker
Orchestration: Kubernetes
Monitoring: OpenTelemetry, Prometheus
Analysis: Evidently, Fairlearn, SHAP
Testing: Locust, pytest

🎯 Exam Requirements Fulfilled

✅ Model Development & Experiment Tracking
✅ Containerization & Continuous Deployment
✅ Scalability & Observability
✅ Data Integrity & Robustness
✅ Fairness & Explainability

📄 Reports Generated

Data poisoning robustness analysis
Fairness assessment with demographic parity metrics
SHAP explainability report focused on Wine Cultivar 2
Load testing performance analysis
Comprehensive project documentation

🚀 Production Readiness
This pipeline is production-ready with:

Automated scaling based on demand
Comprehensive monitoring and alerting
Fair and explainable AI decisions
Robust security and validation
Complete documentation and testing

📞 Contact
For questions about this implementation or MLOps best practices, please refer to the comprehensive documentation in the reports/ directory.
