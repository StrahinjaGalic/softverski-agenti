#!/bin/bash
# Docker startup helper script

echo "🐳 Starting Federated HVAC System..."
echo "📊 Use 'docker-compose logs -f' to monitor progress"
echo "🎯 Demo will start automatically once all services are ready"

# Start all services
docker-compose up --build