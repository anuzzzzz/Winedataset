#!/bin/bash

case "$1" in
    start)
        echo "Starting wine-classifier container..."
        docker run -d -p 5002:5001 --name wine-classifier wine-classifier:latest
        sleep 5
        curl -s http://localhost:5002/health && echo "✅ Container started successfully"
        ;;
    stop)
        echo "Stopping wine-classifier containers..."
        docker stop wine-classifier-test wine-classifier-test2 wine-classifier 2>/dev/null || true
        docker rm wine-classifier-test wine-classifier-test2 wine-classifier 2>/dev/null || true
        echo "✅ Containers stopped and removed"
        ;;
    status)
        echo "=== Container Status ==="
        docker ps | grep wine-classifier || echo "No wine-classifier containers running"
        echo ""
        if curl -s http://localhost:5002/health > /dev/null; then
            echo "✅ API is accessible on port 5002"
        else
            echo "❌ API is not accessible on port 5002"
        fi
        ;;
    logs)
        echo "=== Container Logs ==="
        docker logs wine-classifier-test2 2>/dev/null || docker logs wine-classifier 2>/dev/null || echo "No containers found"
        ;;
    test)
        echo "=== Quick API Test ==="
        curl -s -X POST http://localhost:5002/predict \
          -H "Content-Type: application/json" \
          -d '{
            "alcohol": 13.0, "malic_acid": 2.0, "ash": 2.5,
            "alcalinity_of_ash": 20.0, "magnesium": 100.0,
            "total_phenols": 2.5, "flavanoids": 2.0,
            "nonflavanoid_phenols": 0.3, "proanthocyanins": 1.5,
            "color_intensity": 5.0, "hue": 1.0,
            "od280/od315_of_diluted_wines": 2.5, "proline": 800.0,
            "location": 0
          }' | python -m json.tool
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs|test}"
        ;;
esac
