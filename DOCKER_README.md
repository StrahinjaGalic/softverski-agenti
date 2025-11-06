# 🐳 Docker Setup Guide

## Quick Start

```bash
# Clone repository
git clone <repository-url>
cd softverski-agenti

# Start the complete system (only command needed!)
docker-compose up --build
```

That's it! The federated HVAC demo will start automatically.

## What Happens

1. **Container Startup** (30-60s): All services initialize
2. **Federation Phase** (2-3min): 3 rounds of distributed learning
3. **Real-time Control** (1-2min): HVAC commands based on sensor consensus
4. **Results Generation**: Automatic logs and visualizations

## Monitoring

```bash
# Watch all services
docker-compose logs -f

# Watch specific services
docker-compose logs -f demo          # Demo orchestrator
docker-compose logs -f coordinator   # Federation hub
docker-compose logs -f sensors       # ML training
docker-compose logs -f infrastructure # HVAC + Logging

# See live system status
docker-compose ps
```

## Results

All results are automatically saved to `./logs/`:

- `system_log.json` - Detailed metrics and events
- `demo_summary.txt` - Human-readable summary
- `charts/` - Visualization graphs (if generated)

## Visualization

```bash
# Generate charts (while system is running or after)
python visualization.py

# Or run inside container
docker-compose exec demo python visualization.py
```

## Management

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild and restart
docker-compose up --build --force-recreate

# Scale services (advanced)
docker-compose up --scale sensors=2
```

## Container Architecture

- **coordinator**: Central hub for federation and real-time control
- **infrastructure**: Logger + DeviceController services
- **sensors**: All 5 sensor actors (living_room, bedroom, kitchen, office, bathroom)  
- **demo**: Orchestrator that replaces demo.py functionality

## Troubleshooting

```bash
# Check service health
docker-compose ps

# Debug specific service
docker-compose logs coordinator

# Access container shell
docker-compose exec coordinator bash

# Check network connectivity
docker-compose exec coordinator ping sensors
```

## Development

```bash
# Make code changes, then test
docker-compose up --build

# For faster iteration (no rebuild)
docker-compose restart demo
```

## Configuration

Edit `config/system_config.json` to customize:
- Federation rounds
- Local epochs
- Control parameters
- Sensor configurations

Changes take effect on next `docker-compose up --build`.