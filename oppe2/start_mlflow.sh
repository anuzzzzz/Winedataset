#!/bin/bash
echo "Starting MLflow server on port 5000..."
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000 > logs/mlflow.log 2>&1 &

echo $! > logs/mlflow.pid
echo "MLflow server started. PID saved to logs/mlflow.pid"
echo "Access MLflow UI at: http://localhost:5000"
