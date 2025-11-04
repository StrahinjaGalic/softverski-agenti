"""
DeviceControllerActor - Kontroler uređaja (klima/grejanje)

Simulira upravljanje HVAC sistemom sa:
- Exclusive control (HEAT XOR COOL, nikad oba)
- Hysteresis (deadband, min-on/off times)
- Temperature-based decision logic
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass

from actors import BaseActor
from utils.messages import ApplyCommand, LogMetrics


@dataclass
class DeviceState:
    """Stanje uređaja"""
    current_mode: str = "IDLE"  # HEAT, COOL, IDLE
    target_setpoint: float = 22.0
    last_mode_change: Optional[datetime] = None
    heat_on_time: Optional[datetime] = None
    cool_on_time: Optional[datetime] = None
    

class DeviceControllerActor(BaseActor):
    """
    DeviceController upravlja HVAC sistemom.
    
    Logika:
    - Prima ApplyCommand sa modom i setpoint-om
    - Implementira hysteresis:
        * Deadband: Ne menja mode ako je T unutar ±0.5°C od setpoint-a
        * Min-on time: Mora biti upaljen minimum 2 minuta
        * Min-off time: Mora biti isključen minimum 1 minut
    - Ekskluzivna kontrola: Nikad istovremeno HEAT i COOL
    """
    
    def __init__(
        self,
        actor_id: str = "device_controller",
        logger_host: str = "localhost",
        logger_port: int = 8002,
        mode_threshold: float = 0.7,  # Iz system_config.json
        min_on_time_seconds: int = 2,  # Iz system_config.json
        min_off_time_seconds: int = 2,  # Iz system_config.json
        **kwargs
    ):
        super().__init__(actor_id=actor_id, **kwargs)
        self.logger_host = logger_host
        self.logger_port = logger_port
        
        # Device state
        self.state = DeviceState()
        
        # Hysteresis parameters - koristi iz konfiga!
        self.deadband = mode_threshold  # Ne hardkoduj, koristi iz konfiga
        self.min_on_time = timedelta(seconds=min_on_time_seconds)
        self.min_off_time = timedelta(seconds=min_off_time_seconds)
        
        # Ambient temperatura - biće update-ovana iz senzora
        self.ambient_temperature = None  # Ne hardkoduj, računaj iz senzora
        
        # Tracking za testiranje
        self.commands_received = []  # Lista primljenih komandi
        
        self.logger.info(f"DeviceControllerActor initialized: {actor_id}")
    
    async def handle_message(self, message: Any):
        """Obrađuje primljene poruke."""
        if isinstance(message, ApplyCommand):
            await self._handle_apply_command(message)
        else:
            self.logger.warning(f"Unknown message type: {type(message)}")
    
    async def _handle_apply_command(self, message: ApplyCommand):
        """
        Obrađuje ApplyCommand poruku.
        
        Logika:
        1. Proveri da li je moguće promeniti mode (hysteresis)
        2. Primeni komandu ako je validna
        3. Loguj metrike
        """
        requested_mode = message.mode
        requested_setpoint = message.setpoint
        
        self.logger.info(
            f"📥 Received ApplyCommand: mode={requested_mode}, "
            f"setpoint={requested_setpoint:.1f}°C, ref_temp={message.reference_temp:.1f}°C"
        )
        
        # Dodaj u listu primljenih (za testiranje)
        self.commands_received.append({
            'timestamp': message.timestamp.isoformat(),
            'data': {
                'mode': message.mode,
                'setpoint': message.setpoint,
                'reference_temp': message.reference_temp
            }
        })
        
        # Update ambient temperature iz reference_temp (medijan senzora)
        self.ambient_temperature = message.reference_temp
        
        # Proveri histarezis
        can_change, reason = self._can_change_mode(requested_mode, requested_setpoint)
        
        if not can_change:
            self.logger.warning(f"⚠️  Cannot change mode: {reason}")
            return
        
        # Primeni novu komandu
        old_mode = self.state.current_mode
        self.state.current_mode = requested_mode
        self.state.target_setpoint = requested_setpoint
        self.state.last_mode_change = datetime.now()
        
        # Ažuriraj on/off times
        if requested_mode == "HEAT":
            self.state.heat_on_time = datetime.now()
            self.state.cool_on_time = None
        elif requested_mode == "COOL":
            self.state.cool_on_time = datetime.now()
            self.state.heat_on_time = None
        else:  # IDLE
            self.state.heat_on_time = None
            self.state.cool_on_time = None
        
        self.logger.info(
            f"✅ Mode changed: {old_mode} → {requested_mode} @ {requested_setpoint:.1f}°C"
        )
        
        # Loguj metriku
        await self._log_command_metric(message, old_mode)
    
    def _can_change_mode(self, new_mode: str, new_setpoint: float) -> tuple[bool, str]:
        """
        Provera da li je dozvoljena promena moda (hysteresis logic).
        
        Returns:
            (can_change: bool, reason: str)
        """
        current_mode = self.state.current_mode
        now = datetime.now()
        
        # Ako je isti mode, uvek dozvoljeno
        if new_mode == current_mode:
            return True, "Same mode"
        
        # Deadband check: Da li je temperatura u deadband-u?
        if self.ambient_temperature is not None:
            temp_diff = abs(self.ambient_temperature - new_setpoint)
            if temp_diff < self.deadband and new_mode == "IDLE":
                # Temperatura je u deadband-u, dozvoljavamo IDLE
                return True, "Within deadband, switching to IDLE"
        
        # Min-on time check: Mora biti upaljen minimum min_on_time
        if current_mode == "HEAT" and self.state.heat_on_time:
            time_on = now - self.state.heat_on_time
            if time_on < self.min_on_time:
                return False, f"HEAT on for only {time_on.total_seconds():.1f}s < {self.min_on_time.total_seconds()}s"
        
        if current_mode == "COOL" and self.state.cool_on_time:
            time_on = now - self.state.cool_on_time
            if time_on < self.min_on_time:
                return False, f"COOL on for only {time_on.total_seconds():.1f}s < {self.min_on_time.total_seconds()}s"
        
        # Min-off time check: Mora biti isključen minimum min_off_time
        if current_mode == "IDLE" and self.state.last_mode_change:
            time_off = now - self.state.last_mode_change
            if time_off < self.min_off_time:
                return False, f"IDLE for only {time_off.total_seconds():.1f}s < {self.min_off_time.total_seconds()}s"
        
        # Exclusive control: Nikad HEAT i COOL istovremeno
        if current_mode == "HEAT" and new_mode == "COOL":
            # Mora prvo kroz IDLE
            return False, "Cannot switch directly from HEAT to COOL"
        
        if current_mode == "COOL" and new_mode == "HEAT":
            # Mora prvo kroz IDLE
            return False, "Cannot switch directly from COOL to HEAT"
        
        return True, "Valid mode change"
    
    async def _log_command_metric(self, command: ApplyCommand, old_mode: str):
        """Šalje metrike LoggerActor-u."""
        # Value = temperaturna razlika (ambient - setpoint)
        temp_diff = abs(self.ambient_temperature - command.setpoint) if self.ambient_temperature is not None else 0.0
        
        metric = LogMetrics(
            timestamp=datetime.now(),
            sender_id=self.actor_id,
            receiver_id="logger",
            metric_type="device_command",
            value=temp_diff,
            data={
                "old_mode": old_mode,
                "new_mode": command.mode,
                "setpoint": command.setpoint,
                "reference_temp": command.reference_temp,
                "ambient_temp": self.ambient_temperature
            }
        )
        
        await self.send_message(metric, self.logger_host, self.logger_port)
    
    def get_status(self) -> Dict[str, Any]:
        """Vraća trenutno stanje uređaja."""
        return {
            "mode": self.state.current_mode,
            "setpoint": self.state.target_setpoint,
            "ambient_temperature": self.ambient_temperature,
            "last_mode_change": self.state.last_mode_change.isoformat() if self.state.last_mode_change else None,
            "heat_on_since": self.state.heat_on_time.isoformat() if self.state.heat_on_time else None,
            "cool_on_since": self.state.cool_on_time.isoformat() if self.state.cool_on_time else None,
        }
