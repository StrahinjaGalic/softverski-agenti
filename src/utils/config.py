"""
Konfiguracija sistema - parametri za federativno učenje i kontrolu uređaja.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class FederationConfig:
    """Parametri federativnog učenja"""
    num_rounds: int = 5
    local_epochs: int = 10
    learning_rate: float = 0.01
    min_samples_per_sensor: int = 10
    convergence_threshold: float = 0.001


@dataclass
class ControlConfig:
    """Parametri kontrole uređaja"""
    # Histereza
    deadband: float = 0.5      # ±0.5°C deadband
    mode_threshold: float = 1.0 # ±1.0°C za prelazak režima
    
    # Min-on/off vremena
    min_on_time: int = 240     # 4 minuta u sekundama
    min_off_time: int = 240    # 4 minuta u sekundama
    
    # Opsezi rada
    min_setpoint: float = 16.0  # minimalna temperatura
    max_setpoint: float = 30.0  # maksimalna temperatura


@dataclass
class SensorConfig:
    """Konfiguracija senzora"""
    sensor_id: str
    location: str
    temp_range: tuple = (18.0, 32.0)     # opseg temperatura (°C)
    luminosity_range: tuple = (100, 800)  # opseg osvetljenosti (lx)
    noise_std: float = 0.5               # standardna devijacija šuma


@dataclass
class NetworkConfig:
    """Mrežna konfiguracija"""
    coordinator_host: str = "localhost"
    coordinator_port: int = 8000
    sensor_ports: List[int] = None
    device_controller_port: int = 8001
    logger_port: int = 8002
    
    def __post_init__(self):
        if self.sensor_ports is None:
            # Portovi za 5 senzora
            self.sensor_ports = [8010, 8011, 8012, 8013, 8014]


@dataclass
class SystemConfig:
    """Glavna konfiguracija sistema"""
    federation: FederationConfig = FederationConfig()
    control: ControlConfig = ControlConfig()
    network: NetworkConfig = NetworkConfig()
    
    # Lista konfiguracija senzora
    sensors: List[SensorConfig] = None
    
    # Simulacija
    simulation_duration: int = 300  # 5 minuta u sekundama
    data_collection_interval: int = 1  # svake sekunde
    
    def __post_init__(self):
        if self.sensors is None:
            self.sensors = [
                SensorConfig("sensor_01", "living_room", (20, 28), (200, 600)),
                SensorConfig("sensor_02", "bedroom", (18, 26), (100, 400)),
                SensorConfig("sensor_03", "kitchen", (22, 30), (300, 700)),
                SensorConfig("sensor_04", "office", (19, 27), (400, 800)),
                SensorConfig("sensor_05", "bathroom", (21, 29), (150, 500)),
            ]


# Globalna instanca konfiguracije
config = SystemConfig()