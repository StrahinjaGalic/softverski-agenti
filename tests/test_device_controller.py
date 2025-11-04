"""
Test za DeviceControllerActor

Testira:
1. Prijem ApplyCommand poruka
2. Hysteresis logiku (deadband, min-on/off times)
3. Exclusive control (HEAT/COOL ne mogu istovremeno)
4. Status reporting
"""
import asyncio
import pytest
from datetime import datetime, timedelta

from actors.device_controller_actor import DeviceControllerActor
from utils.messages import ApplyCommand


@pytest.mark.asyncio
async def test_device_controller_basic():
    """Test osnovne funkcionalnosti DeviceController-a"""
    
    print("\n" + "="*70)
    print("TEST: DeviceController - Basic Operations")
    print("="*70)
    
    # Kreiraj actor (bez start() - testiramo samo logiku)
    device = DeviceControllerActor(port=8001)
    
    print("✅ DeviceController created")
    
    # Proveri inicijalno stanje
    status = device.get_status()
    print(f"\n📊 Initial status:")
    print(f"   Mode: {status['mode']}")
    print(f"   Setpoint: {status['setpoint']}°C")
    print(f"   Ambient: {status['ambient_temperature']}°C")
    
    assert status['mode'] == "IDLE"
    assert status['setpoint'] == 22.0
    
    # Pošalji COOL komandu
    print("\n--- Test 1: Apply COOL command ---")
    cmd1 = ApplyCommand(
        timestamp=datetime.now(),
        sender_id="coordinator",
        receiver_id="device_controller",
        mode="COOL",
        setpoint=23.0,
        reference_temp=26.0
    )
    
    await device._handle_apply_command(cmd1)
    
    status = device.get_status()
    print(f"📊 After COOL command:")
    print(f"   Mode: {status['mode']}")
    print(f"   Setpoint: {status['setpoint']}°C")
    
    assert status['mode'] == "COOL"
    assert status['setpoint'] == 23.0
    
    # Pokušaj HEAT direktno (treba da padne - exclusive control)
    print("\n--- Test 2: Try HEAT directly (should fail) ---")
    cmd2 = ApplyCommand(
        timestamp=datetime.now(),
        sender_id="coordinator",
        receiver_id="device_controller",
        mode="HEAT",
        setpoint=25.0,
        reference_temp=20.0
    )
    
    await device._handle_apply_command(cmd2)
    
    status = device.get_status()
    print(f"📊 After attempting HEAT:")
    print(f"   Mode: {status['mode']} (should still be COOL)")
    
    assert status['mode'] == "COOL"  # Ne sme da promeni
    
    # Prvo ide na IDLE, pa onda HEAT
    print("\n--- Test 3: COOL → IDLE → HEAT ---")
    cmd3 = ApplyCommand(
        timestamp=datetime.now(),
        sender_id="coordinator",
        receiver_id="device_controller",
        mode="IDLE",
        setpoint=23.0,
        reference_temp=23.0
    )
    
    # Čekaj min-on time (2 min skraćeno na 1s za test)
    device.min_on_time = timedelta(seconds=1)
    await asyncio.sleep(1.1)
    
    await device._handle_apply_command(cmd3)
    
    status = device.get_status()
    print(f"📊 After IDLE:")
    print(f"   Mode: {status['mode']}")
    assert status['mode'] == "IDLE"
    
    # Sada HEAT
    device.min_off_time = timedelta(seconds=0.5)
    await asyncio.sleep(0.6)
    
    await device._handle_apply_command(cmd2)
    
    status = device.get_status()
    print(f"📊 After HEAT:")
    print(f"   Mode: {status['mode']}")
    assert status['mode'] == "HEAT"
    
    print("\n✅ All tests passed!")



if __name__ == "__main__":
    asyncio.run(test_device_controller_basic())
