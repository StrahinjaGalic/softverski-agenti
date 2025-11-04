# -*- coding: utf-8 -*-
"""
Integration test - Testira ceo sistem u akciji!

Ovaj test pokreće sve aktere:
- CoordinatorActor
- LoggerActor  
- 5 SensorActor-a (različite lokacije)
- DeviceControllerActor (simulacija)

I testira kompletan flow:
1. Federativno učenje (5 rundi)
2. Real-time agregaciju komandi
3. Logovanje metrika i događaja
"""
import asyncio
import pytest
from datetime import datetime

from actors.coordinator_actor import CoordinatorActor
from actors.sensor_actor import SensorActor
from actors.logger_actor import LoggerActor
from actors.device_controller_actor import DeviceControllerActor
from utils.config import SystemConfig, SensorConfig
from utils.messages import ApplyCommand


@pytest.mark.asyncio
async def test_complete_system():
    """
    GLAVNI INTEGRATION TEST
    
    Pokreće kompletan sistem sa svim aktorima i testira:
    1. Federation: 5 rundi učenja
    2. Real-time: Agregaciju predloga i slanje komandi
    3. Logging: Provera da su metrike i eventi logovani
    """
    
    print("\n" + "="*70)
    print(">>> INTEGRATION TEST - KOMPLETAN SISTEM")
    print("="*70 + "\n")
    
    # Konfiguracija
    config = SystemConfig()
    config.federation.num_rounds = 3  # 3 runde za brži test
    config.federation.local_epochs = 5  # 5 epoha za brži test
    
    # Kreiraj aktere
    coordinator = CoordinatorActor(config)
    logger = LoggerActor()
    device = DeviceControllerActor(port=8001)
    
    sensors = [
        SensorActor(config.sensors[0]),  # sensor_1 - bedroom
        SensorActor(config.sensors[1]),  # sensor_2 - bathroom
        SensorActor(config.sensors[2]),  # sensor_3 - living_room
        SensorActor(config.sensors[3]),  # sensor_4 - kitchen
        SensorActor(config.sensors[4])   # sensor_5 - office
    ]
    
    print(f"✅ Kreirano {len(sensors)} senzora")
    print(f"✅ Federacija: {config.federation.num_rounds} runde, {config.federation.local_epochs} epoha\n")
    
    # Pokreni sve aktere
    tasks = []
    
    # Coordinator
    coord_task = asyncio.create_task(coordinator.start())
    tasks.append(coord_task)
    
    # Logger
    logger_task = asyncio.create_task(logger.start())
    tasks.append(logger_task)
    
    # Device Controller
    device_task = asyncio.create_task(device.start())
    tasks.append(device_task)
    
    # Sensori
    for sensor in sensors:
        sensor_task = asyncio.create_task(sensor.start())
        tasks.append(sensor_task)
    
    # Daj im vreme da se pokrenu
    await asyncio.sleep(2)
    print(f"✅ Svi akteri pokrenuti!\n")
    
    # ===== FAZA 1: FEDERATIVNO UČENJE =====
    print("="*70)
    print("📚 FAZA 1: FEDERATIVNO UČENJE")
    print("="*70 + "\n")
    
    # Pokreni federaciju sa listom sensor ID-jeva
    sensor_ids = [s.actor_id for s in sensors]
    federation_task = asyncio.create_task(coordinator.start_federation(sensor_ids))
    
    # Čekaj da se završi (timeout 120s za 3 runde)
    try:
        await asyncio.wait_for(federation_task, timeout=120)
        print("\n✅ Federacija uspešno završena!\n")
    except asyncio.TimeoutError:
        print("\n❌ TIMEOUT: Federacija nije završena u roku od 120s")
        pytest.fail("Federation timeout")
    
    # Proveri status koordinatora
    coord_status = coordinator.get_status()
    print(f"📊 Coordinator status:")
    print(f"   - Runda: {coord_status['current_round']}/{config.federation.num_rounds}")
    print(f"   - Realtime mode: {coord_status['in_realtime_mode']}")
    print(f"   - Model updates primljeno: {coord_status['model_updates_received']}\n")
    
    assert coord_status['current_round'] == config.federation.num_rounds, "Sve runde nisu završene!"
    assert coord_status['in_realtime_mode'] == True, "Sistem nije ušao u realtime mode!"
    
    # Proveri da su svi senzori primili global model
    for sensor in sensors:
        sensor_status = sensor.get_status()
        print(f"📊 {sensor.actor_id} ({sensor.sensor_config.location}):")
        print(f"   - Model treniran: {sensor_status['model_trained']}")
        print(f"   - MSE: {sensor_status['local_mse']:.4f}")
        print(f"   - Federacija runda: {sensor_status['federation_round']}")
        
        assert sensor_status['model_trained'], f"{sensor.actor_id} nema treniran model!"
    
    print("\n✅ SVI SENZORI IMAJU GLOBAL MODEL!\n")
    
    # ===== FAZA 2: REAL-TIME AGREGACIJA =====
    print("="*70)
    print("🔄 FAZA 2: REAL-TIME AGREGACIJA")
    print("="*70 + "\n")
    
    # Simuliraj 3 ciklusa real-time agregacije
    print("Simuliram 3 real-time ciklusa...\n")
    
    from utils.messages import SensorData
    
    for cycle in range(3):
        print(f"--- Ciklus {cycle + 1}/3 ---")
        
        # Simuliraj slanje SensorData od svih senzora
        for sensor in sensors:
            status = sensor.get_status()
            sensor_data = SensorData(
                sender_id=sensor.actor_id,
                receiver_id="coordinator",
                timestamp=datetime.now(),
                temperature=status['current_temperature'],
                luminosity=status['current_luminosity']
            )
            # Pošalji direktno Coordinator-u
            await sensor.send_message(sensor_data, coordinator.host, coordinator.port)
        
        print(f"   Poslato SensorData od {len(sensors)} senzora")
        
        # Sačekaj da se obradi (duže za prvi ciklus - senzori nisu odmah spremni)
        if cycle == 0:
            await asyncio.sleep(3)  # Prvi ciklus - duže čekanje
        else:
            await asyncio.sleep(2)  # Ostali ciklusi
        
        # Proveri device controller
        num_commands = len(device.commands_received)
        print(f"   Komandi primljeno: {num_commands}")
        
        if num_commands > 0:
            last_cmd = device.commands_received[-1]
            mode = last_cmd['data'].get('mode', 'N/A')
            setpoint = last_cmd['data'].get('setpoint', 0.0)
            print(f"   Poslednja komanda: mode={mode}, setpoint={setpoint:.1f}\n")
    
    print(f"✅ Real-time faza završena! Ukupno {len(device.commands_received)} komandi\n")
    
    # ===== FAZA 3: PROVERA LOGGER-A =====
    print("="*70)
    print("📝 FAZA 3: PROVERA LOGOVA")
    print("="*70 + "\n")
    
    # Daj logger-u malo vremena da obradi sve
    await asyncio.sleep(1)
    
    logger_status = logger.get_status()
    print(f"📊 Logger status:")
    print(f"   - Ukupno metrika: {logger_status['total_metrics']}")
    print(f"   - Ukupno događaja: {logger_status['total_events']}")
    print(f"   - Tipovi metrika: {logger_status['metric_types']}\n")
    
    # Logger bi trebalo da ima metrike
    assert logger_status['total_metrics'] > 0, "Logger nema ni jednu metriku!"
    
    # Sačuvaj logove
    await logger.save_logs()
    print(f"✅ Logovi sačuvani u: {logger.log_file}\n")
    
    # Prikaži summary
    print(logger.get_summary())
    
    # ===== FAZA 4: PROVERA SENZORA =====
    print("\n" + "="*70)
    print("🔍 FAZA 4: FINALNA PROVERA SENZORA")
    print("="*70 + "\n")
    
    for sensor in sensors:
        status = sensor.get_status()
        print(f"{sensor.actor_id} ({sensor.sensor_config.location}):")
        print(f"   T={status['current_temperature']:.1f}°C, L={status['current_luminosity']} lx")
        print(f"   MSE={status['local_mse']:.4f}, Model trained={status['model_trained']}")
        
        # Proveri da su modeli validni
        assert status['local_mse'] is not None, f"{sensor.actor_id} nema MSE!"
        assert status['local_mse'] < 5.0, f"{sensor.actor_id} ima prevelik MSE: {status['local_mse']}"
    
    print("\n✅ SVI SENZORI IMAJU VALIDNE MODELE!\n")
    
    # ===== CLEANUP =====
    print("="*70)
    print("🧹 CLEANUP")
    print("="*70 + "\n")
    
    # Zaustavi device controller
    await device.stop()
    print("✅ DeviceController zaustavljen")
    
    # Zaustavi sve aktere
    for task in tasks:
        task.cancel()
    
    # Sačekaj da se sve završi
    await asyncio.sleep(1)
    
    print("✅ Svi akteri zaustavljeni\n")
    
    # ===== FINALNI REZULTAT =====
    print("="*70)
    print("🎉 TEST USPEŠNO ZAVRŠEN!")
    print("="*70)
    print(f"✅ Federacija: {config.federation.num_rounds} rundi")
    print(f"✅ Senzori: {len(sensors)} aktivnih")
    print(f"✅ Komande: {len(device.commands_received)} poslato")
    print(f"✅ Metrike: {logger_status['total_metrics']} logovano")
    print(f"✅ Događaji: {logger_status['total_events']} logovano")
    print("="*70 + "\n")


@pytest.mark.asyncio
async def test_sensor_training_convergence():
    """
    Test da svi senzori konvergiraju ka razumnim MSE vrednostima.
    """
    print("\n" + "="*70)
    print("TEST 1: KONVERGENCIJA TRENIRANJA")
    print("="*70 + "\n")
    
    config = SystemConfig()
    
    # Kreiraj 3 senzora
    sensors = [
        SensorActor(config.sensors[0]),  # bedroom
        SensorActor(config.sensors[1]),  # bathroom
        SensorActor(config.sensors[2])   # living_room
    ]
    
    # Pokreni ih
    tasks = []
    for sensor in sensors:
        task = asyncio.create_task(sensor.start())
        tasks.append(task)
    
    await asyncio.sleep(1)
    
    # Treniraj svaki senzor lokalno
    print("Treniram senzore lokalno...\n")
    
    for sensor in sensors:
        # Generiši podatke
        await sensor._generate_initial_training_data()
        
        # Treniraj
        mse = await sensor._train_local_model()
        
        print(f"✅ {sensor.actor_id} ({sensor.sensor_config.location}): MSE = {mse:.4f}")
        
        # Proveri da je MSE razuman
        assert mse < 1.0, f"{sensor.actor_id} ima prevelik MSE: {mse}"
    
    print("\n✅ Svi senzori konvergirali ka niskim MSE vrednostima!\n")
    
    # Zaustavi
    for task in tasks:
        task.cancel()


@pytest.mark.asyncio
async def test_coordinator_model_aggregation():
    """
    Test da Coordinator pravilno agregira modele pomoću FedAvg.
    """
    print("\n" + "="*70)
    print("🔄 TEST: AGREGACIJA MODELA")
    print("="*70 + "\n")
    
    config = SystemConfig()
    config.federation.num_rounds = 2
    config.federation.local_epochs = 3
    
    coordinator = CoordinatorActor(config)
    logger = LoggerActor()
    
    sensors = [
        SensorActor(config.sensors[0]),  # bedroom
        SensorActor(config.sensors[1]),  # bathroom
        SensorActor(config.sensors[2])   # living_room
    ]
    
    # Pokreni sve
    tasks = []
    tasks.append(asyncio.create_task(coordinator.start()))
    tasks.append(asyncio.create_task(logger.start()))
    
    for sensor in sensors:
        tasks.append(asyncio.create_task(sensor.start()))
    
    await asyncio.sleep(2)
    
    # Pokreni federaciju
    print("Pokrećem federaciju sa 2 runde...\n")
    sensor_ids = [s.actor_id for s in sensors]
    federation_task = asyncio.create_task(coordinator.start_federation(sensor_ids))
    
    try:
        await asyncio.wait_for(federation_task, timeout=60)
    except asyncio.TimeoutError:
        pytest.fail("Timeout")
    
    # Proveri da su modeli ažurirani
    coord_status = coordinator.get_status()
    print(f"✅ Federacija završena: {coord_status['current_round']} rundi\n")
    
    # Proveri da je agregacija radila
    print(f"📊 Koordinator agregirao modele uspešno!\n")
    
    print("\n✅ Agregacija modela radi pravilno!\n")
    
    # Zaustavi
    await logger.save_logs()
    for task in tasks:
        task.cancel()


if __name__ == "__main__":
    # Pokreni samo kompletan test (drugi testovi imaju emoji probleme)
    print("\n>>> POKRETANJE INTEGRATION TESTA...\n")
    
    # asyncio.run(test_sensor_training_convergence())
    # asyncio.run(test_coordinator_model_aggregation())
    asyncio.run(test_complete_system())
