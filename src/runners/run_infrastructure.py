"""
Infrastructure Runner - Starts LoggerActor and DeviceControllerActor in the same container.
This combines the two support services for efficiency.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from actors.logger_actor import LoggerActor
from actors.device_controller_actor import DeviceControllerActor
from utils.config_loader import load_config_from_json


async def wait_for_startup():
    """Give services time to initialize properly."""
    await asyncio.sleep(2)


async def main():
    """Start Logger and DeviceController services."""
    print("🏗️  Starting Infrastructure Services...")
    
    # Load configuration
    config_path = "/app/config/system_config.json"
    if os.path.exists(config_path):
        config = load_config_from_json(config_path)
    else:
        # Fallback for development
        from utils.config import SystemConfig
        config = SystemConfig()
    
    print(f"✅ Configuration loaded from {config_path}")
    
    # Create actors
    logger_actor = LoggerActor(log_file="/app/logs/system_log.json")
    device_actor = DeviceControllerActor(
        port=8001,
        logger_host="localhost",  # Same container
        logger_port=8002,
        mode_threshold=config.control.mode_threshold,
        min_on_time_seconds=config.control.min_on_time,
        min_off_time_seconds=config.control.min_off_time
    )
    
    print("🟢 LoggerActor initialized on port 8002")
    print("🟢 DeviceControllerActor initialized on port 8001")
    
    # Start both services
    try:
        await asyncio.gather(
            logger_actor.start(),
            device_actor.start()
        )
    except KeyboardInterrupt:
        print("\n🛑 Infrastructure services shutting down...")
        await logger_actor.stop()
        await device_actor.stop()
    except Exception as e:
        print(f"❌ Error in infrastructure services: {e}")
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
        print("\n👋 Infrastructure services stopped")