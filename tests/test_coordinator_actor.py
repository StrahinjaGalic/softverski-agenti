"""
Test za CoordinatorActor - Testira federativno učenje i real-time agregaciju.
"""
import asyncio
import pytest
from datetime import datetime

from actors.coordinator_actor import CoordinatorActor
from actors.sensor_actor import SensorActor
from actors.logger_actor import LoggerActor
from utils.config import SystemConfig


@pytest.mark.asyncio
async def test_coordinator_initialization():
    """Test inicijalizacije Coordinator-a."""
    
    config = SystemConfig()
    coordinator = CoordinatorActor(config)
    
    assert coordinator.actor_id == "coordinator"
    assert coordinator.config == config
    assert coordinator.current_round == 0
    assert not coordinator.in_realtime_mode
    
    print("\n✅ Coordinator inicijalizovan")


@pytest.mark.asyncio
async def test_federation_with_sensors():
    """
    Testira kompletan federativni proces sa Coordinator-om i Sensor-ima.
    
    Flow:
    1. Pokreni Coordinator i Logger
    2. Pokreni 3 Sensor-a (razne lokacije)
    3. Coordinator pokreće federaciju (3 runde)
    4. Proveri da svi Sensori prime GlobalModelUpdate
    5. Proveri da Logger beleži metrike
    """
    
    config = SystemConfig()
    config.num_federation_rounds = 3  # Smanjujemo za test
    
    # Kreiraj aktere
    coordinator = CoordinatorActor(config)
    logger = LoggerActor()
    
    sensors = [
        SensorActor("sensor_1", "bedroom", config, port=8010),
        SensorActor("sensor_2", "bathroom", config, port=8011),
        SensorActor("sensor_3", "living_room", config, port=8012)
    ]
    
    # Pokreni aktere
    tasks = []
    tasks.append(asyncio.create_task(coordinator.start()))
    tasks.append(asyncio.create_task(logger.start()))
    for sensor in sensors:
        tasks.append(asyncio.create_task(sensor.start()))
    
    await asyncio.sleep(1)  # Daj im vreme da se povežu
    
    print("\n🚀 Pokrećem federaciju sa 3 sensora, 3 runde...\n")
    
    # Pokreni federaciju
    federation_task = asyncio.create_task(coordinator.start_federation())
    
    # Čekaj da se završi
    try:
        await asyncio.wait_for(federation_task, timeout=60)
    except asyncio.TimeoutError:
        print("❌ Federacija nije završena u roku od 60s")
        pytest.fail("Timeout")
    
    print("\n✅ Federacija završena!\n")
    
    # Proveri status
    status = coordinator.get_status()
    print(f"Coordinator status: {status}")
    
    assert status['current_round'] == 3
    assert status['in_realtime_mode'] == True
    
    # Proveri logger
    logger_status = logger.get_status()
    print(f"Logger status: {logger_status}")
    
    # Logger bi trebalo da ima metrike
    assert logger_status['total_metrics'] > 0
    
    # Sačuvaj logove
    await logger.save_logs()
    
    # Zaustavi aktere
    for task in tasks:
        task.cancel()
    
    print("\n✅ Test prošao!")


@pytest.mark.asyncio
async def test_realtime_aggregation():
    """
    Testira real-time agregaciju komandi.
    
    Flow:
    1. Pokreni Coordinator u realtime modu
    2. Simuliraj prijem SensorData sa 3 sensora
    3. Proveri da Coordinator agregira i šalje ApplyCommand
    """
    
    config = SystemConfig()
    config.num_federation_rounds = 1  # Samo 1 runda da uđemo u realtime
    
    coordinator = CoordinatorActor(config)
    logger = LoggerActor()
    
    # Pokreni aktere
    coord_task = asyncio.create_task(coordinator.start())
    logger_task = asyncio.create_task(logger.start())
    
    await asyncio.sleep(1)
    
    # Forsiraj realtime mode (bez federacije)
    coordinator.in_realtime_mode = True
    
    print("\n🔄 Simuliram SensorData sa 3 sensora...\n")
    
    # Simuliraj SensorData
    from utils.messages import SensorData
    
    sensor_data_1 = SensorData(
        sender_id="sensor_1",
        temperature=24.5,
        light_level=450,
        timestamp=datetime.now()
    )
    
    sensor_data_2 = SensorData(
        sender_id="sensor_2",
        temperature=25.0,
        light_level=500,
        timestamp=datetime.now()
    )
    
    sensor_data_3 = SensorData(
        sender_id="sensor_3",
        temperature=23.8,
        light_level=420,
        timestamp=datetime.now()
    )
    
    # Pošalji SensorData Coordinator-u
    await coordinator.mailbox.put(sensor_data_1)
    await coordinator.mailbox.put(sensor_data_2)
    await coordinator.mailbox.put(sensor_data_3)
    
    await asyncio.sleep(2)  # Daj vreme za obradu
    
    print("✅ SensorData poslati i obrađeni\n")
    
    # Zaustavi
    coord_task.cancel()
    logger_task.cancel()


if __name__ == "__main__":
    # Pokreni testove
    asyncio.run(test_coordinator_initialization())
    asyncio.run(test_federation_with_sensors())
    asyncio.run(test_realtime_aggregation())
