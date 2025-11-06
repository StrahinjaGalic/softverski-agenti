"""
Sensors Runner - Starts all 5 SensorActor instances in a single container.
Each sensor represents a different room with its own ML model and data.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from actors.sensor_actor import SensorActor
from utils.config_loader import load_config_from_json


async def wait_for_coordinator():
    """Wait for coordinator service to be ready."""
    import socket
    
    coordinator_host = os.getenv('COORDINATOR_HOST', 'coordinator')
    coordinator_port = int(os.getenv('COORDINATOR_PORT', '8000'))
    
    print(f"⏳ Waiting for coordinator at {coordinator_host}:{coordinator_port}...")
    
    for attempt in range(30):  # 30 attempts, 2s each = 60s max
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((coordinator_host, coordinator_port))
            sock.close()
            if result == 0:
                print(f"✅ Coordinator ready at {coordinator_host}:{coordinator_port}")
                return True
        except Exception:
            pass
        
        if attempt < 29:
            print(f"   ⏳ Coordinator not ready, retrying... ({attempt + 1}/30)")
            await asyncio.sleep(2)
    
    print(f"❌ Coordinator not available after 60s")
    return False


async def main():
    """Start all 5 SensorActor instances."""
    print("📡 Starting Sensor Services...")
    
    # Load configuration
    config_path = "/app/config/system_config.json"
    if os.path.exists(config_path):
        config = load_config_from_json(config_path)
    else:
        # Fallback for development
        from utils.config import SystemConfig
        config = SystemConfig()
    
    print(f"✅ Configuration loaded from {config_path}")
    
    # Wait for coordinator
    if not await wait_for_coordinator():
        print("❌ Cannot start sensors without coordinator")
        return
    
    # Get coordinator connection info from environment
    coordinator_host = os.getenv('COORDINATOR_HOST', 'coordinator')
    coordinator_port = int(os.getenv('COORDINATOR_PORT', '8000'))
    
    print(f"🔗 Coordinator: {coordinator_host}:{coordinator_port}")
    
    # Create all 5 sensors
    sensors = []
    for i, sensor_config in enumerate(config.sensors):
        sensor = SensorActor(
            sensor_config, 
            coordinator_host=coordinator_host,
            coordinator_port=coordinator_port
        )
        sensors.append(sensor)
        print(f"🟢 SensorActor {sensor_config.sensor_id} ({sensor_config.location}) initialized on port {8010 + i}")
    
    print(f"📡 Starting {len(sensors)} sensor services...")
    
    # Start all sensors
    try:
        await asyncio.gather(*[sensor.start() for sensor in sensors])
    except KeyboardInterrupt:
        print("\n🛑 Sensor services shutting down...")
        # Stop all sensors gracefully
        await asyncio.gather(*[sensor.stop() for sensor in sensors], return_exceptions=True)
    except Exception as e:
        print(f"❌ Error in sensor services: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Sensor services stopped")