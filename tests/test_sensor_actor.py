"""
Test script za SensorActor - osnovne funkcionalnosti.
"""
import asyncio
import sys
import os

# Dodaj src u path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set PYTHONPATH za relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.actors.sensor_actor import SensorActor
from src.utils.config import config


async def test_sensor_actor():
    """Test osnovnih funkcionalnosti SensorActor-a."""
    print("🧪 Testing SensorActor...")
    
    # Kreiraj sensor konfiguraciju
    sensor_config = config.sensors[0]  # sensor_01, living_room
    
    # Kreiraj SensorActor
    sensor = SensorActor(sensor_config)
    
    print(f"✅ SensorActor created: {sensor.actor_id}")
    print(f"   Location: {sensor_config.location}")
    print(f"   Port: {sensor.port}")
    print(f"   Temperature range: {sensor_config.temp_range}")
    print(f"   Luminosity range: {sensor_config.luminosity_range}")
    
    # Test generisanje početnih podataka
    await sensor._generate_initial_training_data()
    
    print(f"✅ Training data generated:")
    print(f"   Samples: {len(sensor.training_data['temperatures'])}")
    print(f"   Current T: {sensor.current_conditions['temperature']:.1f}°C")
    print(f"   Current L: {sensor.current_conditions['luminosity']:.0f}lx")
    
    # Test lokalno treniranje
    try:
        mse = await sensor._train_local_model()
        print(f"✅ Local training completed:")
        print(f"   MSE: {mse:.4f}")
        print(f"   Model weights: {sensor.model.weights}")
        print(f"   Model bias: {sensor.model.bias:.2f}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Test predikcija
    prediction = sensor.model.predict(25.0, 400.0)
    print(f"✅ Prediction test (T=25°C, L=400lx): {prediction:.2f}°C")
    
    # Test simulacija uslova
    original_temp = sensor.current_conditions['temperature']
    sensor._update_simulated_conditions()
    new_temp = sensor.current_conditions['temperature']
    
    print(f"✅ Sensor simulation:")
    print(f"   Original T: {original_temp:.1f}°C")
    print(f"   Updated T: {new_temp:.1f}°C")
    print(f"   Change: {abs(new_temp - original_temp):.2f}°C")
    
    # Test status
    status = sensor.get_status()
    print(f"✅ Sensor status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print("🎉 All SensorActor tests passed!")


if __name__ == "__main__":
    asyncio.run(test_sensor_actor())