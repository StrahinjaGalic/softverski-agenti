"""
Coordinator Runner - Starts the CoordinatorActor service.
This is the central hub for federated learning and real-time aggregation.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from actors.coordinator_actor import CoordinatorActor
from utils.config_loader import load_config_from_json


async def wait_for_dependencies():
    """Wait for infrastructure services to be ready."""
    import socket
    import time
    
    logger_host = os.getenv('LOGGER_HOST', 'infrastructure')
    device_host = os.getenv('DEVICE_HOST', 'infrastructure')
    
    print("⏳ Waiting for infrastructure services...")
    
    # Wait for logger service
    for attempt in range(30):  # 30 attempts, 2s each = 60s max
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((logger_host, 8002))
            sock.close()
            if result == 0:
                print(f"✅ Logger service ready at {logger_host}:8002")
                break
        except Exception:
            pass
        
        if attempt < 29:
            print(f"   ⏳ Logger not ready, retrying... ({attempt + 1}/30)")
            await asyncio.sleep(2)
    else:
        print(f"❌ Logger service not available after 60s")
    
    # Wait for device service  
    for attempt in range(30):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((device_host, 8001))
            sock.close()
            if result == 0:
                print(f"✅ Device service ready at {device_host}:8001")
                break
        except Exception:
            pass
        
        if attempt < 29:
            print(f"   ⏳ Device not ready, retrying... ({attempt + 1}/30)")
            await asyncio.sleep(2)
    else:
        print(f"❌ Device service not available after 60s")


async def main():
    """Start CoordinatorActor service."""
    print("🎯 Starting Coordinator Service...")
    
    # Load configuration
    config_path = "/app/config/system_config.json"
    if os.path.exists(config_path):
        config = load_config_from_json(config_path)
    else:
        # Fallback for development
        from utils.config import SystemConfig
        config = SystemConfig()
    
    # Update network config for Docker
    config.network.device_controller_host = os.getenv('DEVICE_HOST', 'infrastructure')
    config.network.logger_host = os.getenv('LOGGER_HOST', 'infrastructure')
    
    print(f"✅ Configuration loaded from {config_path}")
    print(f"🔗 Logger host: {config.network.logger_host}")
    print(f"🔗 Device host: {config.network.device_controller_host}")
    
    # Wait for dependencies
    await wait_for_dependencies()
    
    # Create coordinator
    coordinator = CoordinatorActor(config)
    print("🟢 CoordinatorActor initialized on port 8000")
    
    # Start coordinator service
    try:
        await coordinator.start()
    except KeyboardInterrupt:
        print("\n🛑 Coordinator service shutting down...")
        await coordinator.stop()
    except Exception as e:
        print(f"❌ Error in coordinator service: {e}")
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
        print("\n👋 Coordinator service stopped")