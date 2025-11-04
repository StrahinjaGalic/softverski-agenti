"""
DEMO - Federativno učenje i Real-time agregacija HVAC sistema

Ovaj script pokreće kompletan sistem:
1. Coordinator - hub za federaciju i agregaciju
2. Logger - čuva metrike i događaje
3. DeviceController - upravlja HVAC uređajem sa hysteresis logikom
4. 5 SensorActor-a - distribuirani senzori u različitim prostorijama

Faze:
- FAZA 1: Federativno učenje (3 runde)
- FAZA 2: Real-time agregacija komandi
- FAZA 3: Prikaz rezultata i metrika
"""
import asyncio
import sys
import time
from datetime import datetime
from typing import List

# Add src to path
sys.path.insert(0, 'd:/4.godina/Agenti/Projekat/softverski-agenti/src')

from actors.coordinator_actor import CoordinatorActor
from actors.sensor_actor import SensorActor
from actors.logger_actor import LoggerActor
from actors.device_controller_actor import DeviceControllerActor
from utils.config import SystemConfig
from utils.config_loader import load_config_from_json
from utils.messages import SensorData


class DemoOrchestrator:
    """Orkestrator za demo prezentaciju sistema."""
    
    def __init__(self):
        # Učitaj config iz JSON-a!
        self.config = load_config_from_json("config/system_config.json")
        self.config.federation.num_rounds = 3
        self.config.federation.local_epochs = 5
        
        # Akteri
        self.coordinator = None
        self.logger = None
        self.device = None
        self.sensors = []
        self.tasks = []
        
    def print_header(self, text: str, char: str = "="):
        """Formatovan header."""
        width = 70
        print("\n" + char * width)
        print(f"{text:^{width}}")
        print(char * width + "\n")
    
    def print_progress(self, current: int, total: int, label: str = "Progress"):
        """Progress bar."""
        bar_length = 40
        filled = int(bar_length * current / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        percent = 100 * current / total
        print(f"\r{label}: |{bar}| {percent:.0f}% ({current}/{total})", end="", flush=True)
        if current == total:
            print()  # Nova linija na kraju
    
    async def start_actors(self):
        """Pokreni sve aktere."""
        self.print_header("POKRETANJE SISTEMA", "=")
        
        print("[*]  Inicijalizacija aktera...")
        self.coordinator = CoordinatorActor(self.config)
        self.logger = LoggerActor()
        self.device = DeviceControllerActor(
            port=8001,
            mode_threshold=self.config.control.mode_threshold,  # Prosleđuj iz konfiga!
            min_on_time_seconds=self.config.control.min_on_time,  # Prosleđuj iz konfiga!
            min_off_time_seconds=self.config.control.min_off_time  # Prosleđuj iz konfiga!
        )
        
        self.sensors = [
            SensorActor(self.config.sensors[0]),  # living_room
            SensorActor(self.config.sensors[1]),  # bedroom
            SensorActor(self.config.sensors[2]),  # kitchen
            SensorActor(self.config.sensors[3]),  # office
            SensorActor(self.config.sensors[4])   # bathroom
        ]
        
        print(f"[+] Kreirano:")
        print(f"  - 1x CoordinatorActor")
        print(f"  - 1x LoggerActor")
        print(f"  - 1x DeviceControllerActor")
        print(f"  - {len(self.sensors)}x SensorActor")
        
        print("\n[>>] Pokretanje aktera...")
        
        # Start tasks
        self.tasks.append(asyncio.create_task(self.coordinator.start()))
        self.tasks.append(asyncio.create_task(self.logger.start()))
        self.tasks.append(asyncio.create_task(self.device.start()))
        
        for sensor in self.sensors:
            self.tasks.append(asyncio.create_task(sensor.start()))
        
        await asyncio.sleep(2)  # Daj im vreme da se pokrenu
        print("[+] Svi akteri aktivni!\n")
    
    async def run_federation(self):
        """Pokreni federativno učenje."""
        self.print_header("FAZA 1: FEDERATIVNO UČENJE", "=")
        
        print("Parametri:")
        print(f"  - Broj rundi: {self.config.federation.num_rounds}")
        print(f"  - Epohe po rundi: {self.config.federation.local_epochs}")
        print(f"  - Broj senzora: {len(self.sensors)}")
        print(f"  - Algoritam: FedAvg (Federated Averaging)\n")
        
        print("Treniranje u toku...\n")
        
        # Pokreni federaciju
        sensor_ids = [s.sensor_config.sensor_id for s in self.sensors]
        await self.coordinator.start_federation(sensor_ids)
        
        start_time = time.time()
        
        # Čekaj da se federacija završi
        timeout = 120
        elapsed = 0
        while elapsed < timeout:
            coord_status = self.coordinator.get_status()
            
            if coord_status['in_realtime_mode']:
                elapsed_time = time.time() - start_time
                print(f"\n[+] Federacija završena za {elapsed_time:.1f}s")
                break
            
            # Progress bar
            current_round = coord_status['current_round']
            self.print_progress(
                current_round, 
                self.config.federation.num_rounds,
                "Federacija"
            )
            
            await asyncio.sleep(1)
            elapsed += 1
        
        # Prikaz rezultata
        print("\nRezultati federacije:")
        for i, sensor in enumerate(self.sensors, 1):
            status = sensor.get_status()
            location = status['location']
            local_mse = status.get('local_mse', 0.0)
            print(f"  {i}. {location:12s} - MSE: {local_mse:.4f}")
        
        print(f"\n[+] Svi senzori imaju globalni model!\n")
    
    async def run_realtime(self):
        """Pokreni real-time agregaciju."""
        self.print_header("FAZA 2: REAL-TIME AGREGACIJA", "=")
        
        print("Simulacija real-time ciklusa...\n")
        
        num_cycles = 5
        for cycle in range(num_cycles):
            print(f"--- Ciklus {cycle + 1}/{num_cycles} ---")
            
            # Prikupi trenutne podatke
            temps = []
            for sensor in self.sensors:
                status = sensor.get_status()
                temps.append(status['current_temperature'])
                
                # Pošalji SensorData
                sensor_data = SensorData(
                    sender_id=sensor.actor_id,
                    receiver_id="coordinator",
                    timestamp=datetime.now(),
                    temperature=status['current_temperature'],
                    luminosity=status['current_luminosity']
                )
                await sensor.send_message(sensor_data, self.coordinator.host, self.coordinator.port)
            
            avg_temp = sum(temps) / len(temps)
            print(f"  Prosečna temperatura: {avg_temp:.1f}°C")
            
            # Čekaj agregaciju
            await asyncio.sleep(2 if cycle == 0 else 1.5)
            
            # Proveri komande
            num_commands = len(self.device.commands_received)
            device_status = self.device.get_status()
            mode = device_status['mode']
            setpoint = device_status['setpoint']
            
            print(f"  DeviceController: {mode} @ {setpoint:.1f}°C")
            print(f"  Komandi poslato: {num_commands}\n")
        
        total_commands = len(self.device.commands_received)
        print(f"[+] Real-time faza završena!")
        print(f"  Ukupno komandi: {total_commands}\n")
    
    async def show_results(self):
        """Prikaz finalnih rezultata."""
        self.print_header("FAZA 3: FINALNI REZULTATI", "=")
        
        # Logger stats
        logger_status = self.logger.get_status()
        print("Logger statistika:")
        print(f"  - Metrike: {logger_status['total_metrics']}")
        print(f"  - Događaji: {logger_status['total_events']}")
        print(f"  - Log fajl: {self.logger.log_file}\n")
        
        # Device stats
        device_status = self.device.get_status()
        print("DeviceController:")
        print(f"  - Trenutni mod: {device_status['mode']}")
        print(f"  - Setpoint: {device_status['setpoint']}°C")
        print(f"  - Ambient: {device_status['ambient_temperature']}°C")
        print(f"  - Komandi primljeno: {len(self.device.commands_received)}\n")
        
        # Sensor stats
        print("Senzori:")
        for sensor in self.sensors:
            status = sensor.get_status()
            local_mse = status.get('local_mse', 0.0)
            print(f"  - {status['location']:12s}: T={status['current_temperature']:.1f}°C, "
                  f"L={status['current_luminosity']:.0f}lx, MSE={local_mse:.4f}")
        
        print("\n[+] Svi akteri funkcionalni!\n")
    
    async def cleanup(self):
        """Zaustavi sve aktere."""
        self.print_header("CLEANUP", "=")
        
        print("Zaustavljanje aktera...")
        
        await self.device.stop()
        
        for task in self.tasks:
            task.cancel()
        
        await asyncio.sleep(1)
        print("[+] Svi akteri zaustavljeni\n")
    
    async def run(self):
        """Glavni flow demo-a."""
        try:
            print("\n" + "=" * 70)
            print("DEMO: Federativno učenje i Real-time HVAC kontrola".center(70))
            print("=" * 70)
            
            await self.start_actors()
            await self.run_federation()
            await self.run_realtime()
            await self.show_results()
            
            self.print_header("DEMO ZAVRŠEN USPEŠNO!", "=")
            
        except KeyboardInterrupt:
            print("\n\nDemo prekinut (Ctrl+C)")
        except Exception as e:
            print(f"\n\nGreška: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.cleanup()


async def main():
    """Entry point."""
    demo = DemoOrchestrator()
    await demo.run()


if __name__ == "__main__":
    asyncio.run(main())
