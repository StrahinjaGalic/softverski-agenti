"""
CoordinatorActor - Centralni hub za federativno učenje i real-time agregaciju.

CoordinatorActor je odgovoran za:
1. Pokretanje rundi federativnog učenja (StartTraining broadcast)
2. Prikupljanje ModelUpdate poruka od svih senzora
3. Agregaciju modela pomoću FedAvg algoritma
4. Distribuciju GlobalModelUpdate svim senzorima
5. Real-time agregaciju predloga od senzora (medijan)
6. Slanje komandi DeviceController-u
"""
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging

from actors import BaseActor
from federation.fedavg import FedAvg
from utils.messages import (
    Message, StartTraining, ModelUpdate, GlobalModelUpdate,
    CollectProposals, ProposalResponse, ApplyCommand, 
    SensorData, LogMetrics, LogEvent
)
from utils.config import SystemConfig


class CoordinatorActor(BaseActor):
    """
    Koordinator za federativno učenje i real-time kontrolu.
    
    Lifecycle:
    1. Federation phase: Pokreće num_rounds rundi treniranja
    2. Real-time phase: Agregira predloge i šalje komande
    """
    
    def __init__(self, config: SystemConfig):
        """
        Args:
            config: SystemConfig objekat sa svim parametrima
        """
        super().__init__("coordinator", "localhost", 8000)
        
        # Konfiguracija
        self.config = config
        
        # Federativno učenje
        self.fedavg = FedAvg()
        self.num_sensors = len(config.sensors)  # Broj senzora iz liste
        self.current_round = 0
        self.max_rounds = config.federation.num_rounds
        
        # Prikupljanje model update-a
        self.model_updates: Dict[str, ModelUpdate] = {}  # sensor_id -> ModelUpdate
        self.waiting_for_updates = False
        
        # Real-time faza
        self.sensor_proposals: Dict[str, ProposalResponse] = {}  # sensor_id -> ProposalResponse
        self.sensor_data: Dict[str, SensorData] = {}  # sensor_id -> SensorData
        self.in_realtime_mode = False
        
        # Device controller i Logger portovi iz config-a
        self.device_controller_host = "localhost"
        self.device_controller_port = config.network.device_controller_port
        
        self.logger_host = "localhost"
        self.logger_port = config.network.logger_port
        
        # Lista sensor ID-jeva (pratimo ko je aktivan)
        self.sensor_ids: List[str] = []
        
        self.logger.info("CoordinatorActor initialized")
    
    async def start(self):
        """Pokreće koordinatora i automatski startuje federativno učenje."""
        self.logger.info("Starting CoordinatorActor")
        
        # Pokreni base actor (server + message loop)
        await super().start()
    
    async def start_federation(self, sensor_ids: List[str]):
        """
        Pokreće proces federativnog učenja.
        
        Args:
            sensor_ids: Lista ID-jeva svih senzora u sistemu
        """
        self.sensor_ids = sensor_ids
        self.logger.info(f"Starting federation with {len(sensor_ids)} sensors")
        
        # Pokreni sve runde
        for round_num in range(1, self.max_rounds + 1):
            self.current_round = round_num
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"FEDERATION ROUND {round_num}/{self.max_rounds}")
            self.logger.info(f"{'='*60}")
            
            # Izvši jednu rundu
            await self._run_federation_round(round_num)
            
            # Kratka pauza između rundi
            await asyncio.sleep(1)
        
        # Posle svih rundi, pređi u real-time mod
        self.in_realtime_mode = True
        self.logger.info("\n🎉 Federation training completed! Switching to real-time mode.")
        
        # Loguj završetak
        await self._log_event("federation_complete", 
                             f"Completed {self.max_rounds} rounds",
                             {"final_round": self.max_rounds})
    
    async def _run_federation_round(self, round_num: int):
        """
        Izvršava jednu rundu federativnog učenja.
        
        Flow:
        1. Broadcast StartTraining svim senzorima
        2. Čekaj ModelUpdate od svih senzora
        3. Agreguj sa FedAvg
        4. Broadcast GlobalModelUpdate svim senzorima
        """
        # KORAK 1: Pošalji StartTraining
        self.model_updates.clear()
        self.waiting_for_updates = True
        
        await self._broadcast_start_training(round_num)
        
        # KORAK 2: Čekaj sve model update-e (sa timeout-om)
        timeout = 30  # 30 sekundi max
        await self._wait_for_all_model_updates(timeout)
        
        # KORAK 3: Agreguj modele
        expected_sensors = len(self.sensor_ids)
        if len(self.model_updates) == expected_sensors:
            global_weights, global_bias = self.fedavg.aggregate_models(
                list(self.model_updates.values())
            )
            
            # Log agregaciju
            stats = self.fedavg.get_aggregation_stats()
            self.logger.info(f"✅ Aggregation complete:")
            self.logger.info(f"   Global weights: {global_weights}")
            self.logger.info(f"   Global bias: {global_bias:.2f}")
            self.logger.info(f"   Participants: {stats['num_participants']}")
            self.logger.info(f"   Total samples: {stats['total_samples']}")
            
            # Loguj metrike
            await self._log_metrics("aggregation", {
                "round": round_num,
                "weights": global_weights.tolist(),
                "bias": float(global_bias),
                "num_participants": stats['num_participants']
            })
            
            # KORAK 4: Broadcast globalni model
            await self._broadcast_global_model(global_weights, global_bias, round_num)
        else:
            self.logger.error(f"❌ Timeout! Only {len(self.model_updates)}/{expected_sensors} sensors responded")
        
        self.waiting_for_updates = False
    
    async def _broadcast_start_training(self, round_num: int):
        """Šalje StartTraining poruku svim senzorima."""
        self.logger.info(f"📤 Broadcasting StartTraining to {len(self.sensor_ids)} sensors...")
        
        for sensor_id in self.sensor_ids:
            # Ekstraktuj port iz sensor_id (sensor_01 -> port 8010)
            sensor_port = self._get_sensor_port(sensor_id)
            
            message = StartTraining(
                timestamp=datetime.now(),
                sender_id=self.actor_id,
                receiver_id=sensor_id
            )
            
            await self.send_message(message, "localhost", sensor_port)
            self.logger.debug(f"   → Sent to {sensor_id}")
    
    async def _wait_for_all_model_updates(self, timeout: int):
        """
        Čeka da svi senzori pošalju ModelUpdate.
        
        Args:
            timeout: Maksimalno vreme čekanja u sekundama
        """
        expected_sensors = len(self.sensor_ids)  # Broj aktivnih senzora
        self.logger.info(f"⏳ Waiting for ModelUpdate from {expected_sensors} sensors...")
        
        start_time = asyncio.get_event_loop().time()
        
        while len(self.model_updates) < expected_sensors:
            # Proveri timeout
            if asyncio.get_event_loop().time() - start_time > timeout:
                self.logger.warning(f"⚠️ Timeout reached!")
                break
            
            # Čekaj malo pre sledećeg check-a
            await asyncio.sleep(0.5)
            
            # Debug info
            received = len(self.model_updates)
            expected = len(self.sensor_ids)
            if received > 0:
                self.logger.debug(f"   Received: {received}/{expected}")
    
    async def _broadcast_global_model(self, weights: np.ndarray, bias: float, round_num: int):
        """Šalje GlobalModelUpdate svim senzorima."""
        self.logger.info(f"📤 Broadcasting GlobalModelUpdate (round {round_num})...")
        
        for sensor_id in self.sensor_ids:
            sensor_port = self._get_sensor_port(sensor_id)
            
            message = GlobalModelUpdate(
                timestamp=datetime.now(),
                sender_id=self.actor_id,
                receiver_id=sensor_id,
                global_weights=weights,
                global_bias=bias,
                round_number=round_num
            )
            
            await self.send_message(message, "localhost", sensor_port)
            self.logger.debug(f"   → Sent to {sensor_id}")
    
    async def handle_message(self, message: Message):
        """
        Obrađuje primljene poruke.
        
        Message types:
        - ModelUpdate: Od senzora tokom federacije
        - SensorData: Od senzora u real-time modu
        - ProposalResponse: Odgovor na CollectProposals
        """
        if isinstance(message, ModelUpdate):
            await self._handle_model_update(message)
        elif isinstance(message, SensorData):
            await self._handle_sensor_data(message)
        elif isinstance(message, ProposalResponse):
            await self._handle_proposal_response(message)
        else:
            self.logger.warning(f"Unknown message type: {type(message)}")
    
    async def _handle_model_update(self, message: ModelUpdate):
        """Prikuplja ModelUpdate od senzora tokom federacije."""
        self.logger.info(f"📥 Received ModelUpdate from {message.sender_id}")
        self.logger.info(f"   Weights: {message.weights}")
        self.logger.info(f"   Bias: {message.bias:.2f}")
        self.logger.info(f"   MSE: {message.mse:.4f}")
        self.logger.info(f"   Samples: {message.num_samples}")
        
        # Sačuvaj update
        self.model_updates[message.sender_id] = message
        
        self.logger.info(f"   Progress: {len(self.model_updates)}/{len(self.sensor_ids)}")
    
    async def _handle_sensor_data(self, message: SensorData):
        """
        Obrađuje SensorData u real-time modu.
        Kada prikupi sve, pokreće agregaciju predloga.
        """
        if not self.in_realtime_mode:
            return  # Ignorišemo ako nismo u real-time modu
        
        self.logger.debug(f"📥 SensorData from {message.sender_id}: T={message.temperature:.1f}°C, L={message.luminosity:.0f}lx")
        
        # Sačuvaj podatke
        self.sensor_data[message.sender_id] = message
        
        # Ako imamo od svih, pokreni agregaciju
        expected_sensors = len(self.sensor_ids)
        if len(self.sensor_data) >= expected_sensors:
            await self._aggregate_and_apply_command()
            self.sensor_data.clear()
    
    async def _handle_proposal_response(self, message: ProposalResponse):
        """Prikuplja ProposalResponse od senzora."""
        self.logger.debug(f"📥 Proposal from {message.sender_id}: {message.proposal:.2f}°C")
        
        self.sensor_proposals[message.sender_id] = message
    
    async def _aggregate_and_apply_command(self):
        """
        Agregira podatke i predloge od svih senzora i šalje komandu DeviceController-u.
        
        Logika:
        1. Izračunaj T_med (medijan temperatura)
        2. Izračunaj Y_agg (medijan predloga)
        3. Odredi mode (HEAT/COOL/IDLE) na osnovu razlike
        4. Pošalji ApplyCommand
        """
        # Ekstraktuj temperature i predloge
        temperatures = [data.temperature for data in self.sensor_data.values()]
        
        # Prvo tražimo predloge - šaljemo CollectProposals
        await self._collect_proposals()
        
        # Čekaj malo za odgovore
        await asyncio.sleep(1)
        
        expected_sensors = len(self.sensor_ids)
        if len(self.sensor_proposals) < expected_sensors:
            self.logger.warning(f"Not enough proposals: {len(self.sensor_proposals)}/{expected_sensors}")
            return
        
        proposals = [prop.proposal for prop in self.sensor_proposals.values()]
        
        # Medijani
        T_med = float(np.median(temperatures))
        Y_agg = float(np.median(proposals))
        
        # Odredi mode
        diff = T_med - Y_agg
        mode_threshold = self.config.control.mode_threshold
        
        if diff > mode_threshold:
            mode = "COOL"  # Previše toplo
            setpoint = Y_agg
        elif diff < -mode_threshold:
            mode = "HEAT"  # Previše hladno
            setpoint = Y_agg
        else:
            mode = "IDLE"  # U deadband-u
            setpoint = Y_agg
        
        self.logger.info(f"🎯 Aggregation: T_med={T_med:.2f}°C, Y_agg={Y_agg:.2f}°C → {mode} @ {setpoint:.2f}°C")
        
        # Pošalji komandu
        await self._send_apply_command(mode, setpoint, T_med)
        
        # Clear proposals za sledeći ciklus
        self.sensor_proposals.clear()
    
    async def _collect_proposals(self):
        """Šalje CollectProposals svim senzorima."""
        self.logger.debug("📤 Collecting proposals from sensors...")
        
        for sensor_id in self.sensor_ids:
            sensor_port = self._get_sensor_port(sensor_id)
            
            message = CollectProposals(
                timestamp=datetime.now(),
                sender_id=self.actor_id,
                receiver_id=sensor_id
            )
            
            await self.send_message(message, "localhost", sensor_port)
    
    async def _send_apply_command(self, mode: str, setpoint: float, reference_temp: float):
        """Šalje ApplyCommand DeviceController-u."""
        message = ApplyCommand(
            timestamp=datetime.now(),
            sender_id=self.actor_id,
            receiver_id="device_controller",
            mode=mode,
            setpoint=setpoint,
            reference_temp=reference_temp
        )
        
        await self.send_message(message, self.device_controller_host, self.device_controller_port)
        self.logger.info(f"📤 Sent ApplyCommand to DeviceController: {mode} @ {setpoint:.2f}°C")
    
    async def _log_metrics(self, metric_type: str, data: dict):
        """Šalje metrike LoggerActor-u."""
        message = LogMetrics(
            timestamp=datetime.now(),
            sender_id=self.actor_id,
            receiver_id="logger",
            metric_type=metric_type,
            value=0.0,  # Možemo dodati specifičnu vrednost ako treba
            round_number=self.current_round if not self.in_realtime_mode else None
        )
        message.data = data  # Dodaj extra podatke (nisu u spec, ali korisno)
        
        await self.send_message(message, self.logger_host, self.logger_port)
    
    async def _log_event(self, event_type: str, description: str, data: Optional[dict] = None):
        """Šalje događaj LoggerActor-u."""
        message = LogEvent(
            timestamp=datetime.now(),
            sender_id=self.actor_id,
            receiver_id="logger",
            event_type=event_type,
            description=description,
            data=data
        )
        
        await self.send_message(message, self.logger_host, self.logger_port)
    
    def _get_sensor_port(self, sensor_id: str) -> int:
        """Mapira sensor ID na port."""
        try:
            sensor_num = int(sensor_id.split('_')[1])
            return self.config.network.sensor_ports[sensor_num - 1]
        except (IndexError, ValueError):
            return self.config.network.sensor_ports[0]
    
    def get_status(self) -> Dict:
        """Vraća trenutni status koordinatora."""
        return {
            'actor_id': self.actor_id,
            'current_round': self.current_round,
            'max_rounds': self.max_rounds,
            'in_realtime_mode': self.in_realtime_mode,
            'num_sensors': self.num_sensors,
            'model_updates_received': len(self.model_updates),
            'global_model': {
                'weights': self.fedavg.global_weights.tolist() if self.fedavg.global_weights is not None else None,
                'bias': float(self.fedavg.global_bias) if self.fedavg.global_bias is not None else None
            }
        }
