"""
Definicije poruka koje se razmenjuju između aktora u sistemu.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import numpy as np


@dataclass
class Message:
    """Bazna klasa za sve poruke"""
    timestamp: datetime
    sender_id: str
    receiver_id: str


@dataclass  
class StartTraining(Message):
    """Poruka za početak treniranja lokalnog modela"""
    pass


@dataclass
class ModelUpdate(Message):
    """Poruka sa lokalnim parametrima modela"""
    weights: np.ndarray  # w1, w2 za T i L
    bias: float          # b
    num_samples: int     # broj uzoraka za ponderisanje
    mse: float          # lokalna greška


@dataclass
class GlobalModelUpdate(Message):
    """Poruka sa globalnim parametrima modela"""
    global_weights: np.ndarray
    global_bias: float
    round_number: int


@dataclass
class PredictRequest(Message):
    """Zahtev za predikciju"""
    temperature: float   # T_i (°C)
    luminosity: float    # L_i (lx)


@dataclass
class PredictResponse(Message):
    """Odgovor sa predikcijom"""
    predicted_command: float  # Y_cmd (°C)
    temperature: float
    luminosity: float


@dataclass
class CollectProposals(Message):
    """Poruka za sakupljanje predloga od svih senzora"""
    pass


@dataclass
class ProposalResponse(Message):
    """Odgovor sa predlogom komande"""
    proposal: float      # n_i - predlog senzora
    temperature: float   # lokalna temperatura
    confidence: float    # pouzdanost predloga


@dataclass
class ApplyCommand(Message):
    """Komanda za primenu na uređaju"""
    mode: str           # HEAT/COOL/IDLE
    setpoint: float     # ciljna temperatura (°C)
    reference_temp: float # medijan temperatura T_med


@dataclass
class LogMetrics(Message):
    """Poruka za logovanje metrika"""
    metric_type: str    # MSE, accuracy, itd.
    value: float
    round_number: Optional[int] = None


@dataclass
class LogEvent(Message):
    """Poruka za logovanje događaja"""
    event_type: str     # mode_change, training_complete, itd.
    description: str
    data: Optional[dict] = None


@dataclass
class SensorData(Message):
    """Podaci sa senzora"""
    temperature: float
    luminosity: float
    actual_command: Optional[float] = None  # za treniranje