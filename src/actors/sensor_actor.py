"""
SensorActor - Implementira lokalno treniranje modela i simulaciju senzora.

SensorActor je odgovoran za:
1. Lokalno treniranje HVACModel na svojim podacima
2. Simulaciju realnih senzorskih podataka (T, L)
3. Slanje lokalnih model parametara u federativnom učenju
4. Real-time predloge komandi na osnovu trenutnih uslova
"""
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging

from actors import BaseActor
from federation.model import HVACModel
from simulation.data_generator import DataGenerator
from utils.messages import (
    Message, StartTraining, ModelUpdate, GlobalModelUpdate,
    PredictRequest, PredictResponse, ProposalResponse, CollectProposals, SensorData
)
from utils.config import config, SensorConfig


class SensorActor(BaseActor):
    """
    Aktor koji simulira senzor sa lokalnim treniranjem modela.
    
    Atributi:
        sensor_config: konfiguracija senzora (ID, lokacija, opsezi)
        model: HVACModel instance za lokalno treniranje
        training_data: lokalni podaci za treniranje (T, L, Y_cmd)
        current_conditions: trenutna temperatura i osvetljenost
        coordinator_info: host i port Coordinator-a za komunikaciju
    """
    
    def __init__(self, sensor_config: SensorConfig, 
                 coordinator_host: str = "localhost", coordinator_port: int = 8000):
        """
        Inicijalizuje SensorActor.
        
        Args:
            sensor_config: konfiguracija senzora (ID, lokacija, opsezi)
            coordinator_host: IP adresa Coordinator-a
            coordinator_port: port Coordinator-a
        """
        # Pokreni BaseActor sa sensor ID i portom
        sensor_port = self._get_sensor_port(sensor_config.sensor_id)
        super().__init__(sensor_config.sensor_id, "localhost", sensor_port)
        
        # Sensor konfiguracija
        self.sensor_config = sensor_config
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        
        # ML komponente
        self.model = HVACModel()
        self.training_data = {
            'temperatures': [],
            'luminosities': [], 
            'commands': []
        }
        
        # Trenutni uslovi (simulirani senzori)
        self.current_conditions = {
            'temperature': 25.0,  # °C
            'luminosity': 400.0,  # lx
            'last_update': datetime.now()
        }
        
        # Stanje aktora
        self.is_training_mode = False
        self.federation_round = 0
        self.local_mse = float('inf')
        
        # Data generator za simulaciju
        self.data_generator = DataGenerator()
        
        # Scenario tracking - za hardkodovane cikluse iz config-a
        self.current_cycle = 0
        self.scenario_config = self._load_scenario_config()
        
        # Task-ovi za periodične operacije
        self.sensor_simulation_task = None
        self.proposal_task = None
        
        self.logger.info(f"SensorActor {self.sensor_config.sensor_id} initialized on port {sensor_port}")
    
    def _load_scenario_config(self):
        """Učitava scenario konfiguraciju iz system_config.json."""
        try:
            import json
            from pathlib import Path
            
            config_path = Path(__file__).parent.parent.parent / "config" / "system_config.json"
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
                scenarios = full_config.get("scenarios", {})
                
                if scenarios.get("enabled", False):
                    self.logger.info(f"Scenario mode ENABLED - using hardcoded cycles from config")
                    return scenarios
                else:
                    self.logger.info(f"Scenario mode DISABLED - using dynamic simulation")
                    return None
        except Exception as e:
            self.logger.warning(f"Failed to load scenario config: {e}. Using dynamic simulation.")
            return None
    
    def _get_sensor_port(self, sensor_id: str) -> int:
        """
        Mapira sensor ID na port broj.
        
        Args:
            sensor_id: ID senzora (npr. "sensor_01")
            
        Returns:
            Port broj za ovaj senzor
        """
        # Ekstraktuj broj iz sensor_id (sensor_01 -> 1)
        try:
            sensor_num = int(sensor_id.split('_')[1])
            return config.network.sensor_ports[sensor_num - 1]
        except (IndexError, ValueError):
            # Fallback na prvi dostupan port
            return config.network.sensor_ports[0]
    
    async def start(self):
        """Proširuje BaseActor.start() sa sensor-specifičnim task-ovima."""
        self.logger.info("Starting SensorActor with simulation tasks")
        
        # Generiši početne training podatke
        await self._generate_initial_training_data()
        
        # Pokreni sensor simulaciju
        self.sensor_simulation_task = asyncio.create_task(self._sensor_simulation_loop())
        
        # Pokreni proposal task (čeka na federation završetak)
        # self.proposal_task će biti kreiran nakon što se završi federativno učenje
        
        # Pokreni base actor
        await super().start()
    
    async def stop(self):
        """Proširuje BaseActor.stop() sa cleanup task-ova."""
        self.logger.info("Stopping SensorActor")
        
        # Prekini task-ove
        if self.sensor_simulation_task:
            self.sensor_simulation_task.cancel()
        if self.proposal_task:
            self.proposal_task.cancel()
        
        await super().stop()
    
    async def _generate_initial_training_data(self):
        """
        Generiše početne lokalne podatke za treniranje modela.
        Svaki senzor ima različite karakteristike na osnovu lokacije.
        """
        self.logger.info(f"Generating training data for {self.sensor_config.location}")
        
        # Generiši povijene podatke (7 dana sa 10min intervalima = 1008 uzoraka)
        # Svaki senzor će imati različite podatke jer imaju različite temp_range i luminosity_range
        df = self.data_generator.generate_sensor_data(
            sensor_id=self.sensor_config.sensor_id,
            location=self.sensor_config.location,
            temp_range=self.sensor_config.temp_range,
            luminosity_range=self.sensor_config.luminosity_range,
            duration_hours=168,  # 7 dana
            interval_minutes=10,
            noise_std=self.sensor_config.noise_std
        )
        
        # Izdvoji podatke za treniranje
        self.training_data['temperatures'] = df['temperature'].values
        self.training_data['luminosities'] = df['luminosity'].values
        self.training_data['commands'] = df['y_cmd'].values
        
        # Postavi početne uslove RANDOM za realističnu simulaciju
        temp_min, temp_max = self.sensor_config.temp_range
        lum_min, lum_max = self.sensor_config.luminosity_range
        
        # Potpuno random početne vrednosti - nema hardcoding-a!
        # Sistem treba da radi sa bilo kojim validnim početnim uslovima
        initial_temp = np.random.uniform(temp_min, temp_max)
        initial_lum = np.random.uniform(lum_min, lum_max)
        
        self.current_conditions.update({
            'temperature': float(initial_temp),
            'luminosity': float(initial_lum),
            'last_update': datetime.now()
        })
        
        self.logger.info(f"Generated {len(df)} training samples. "
                        f"Initial: T={initial_temp:.1f}°C, L={initial_lum:.0f}lx")
    
    async def _sensor_simulation_loop(self):
        """
        Periodično ažurira simulirane senzorske podatke.
        Simulira realne varijacije temperature i osvetljenosti.
        """
        try:
            while self.is_running:
                await asyncio.sleep(config.simulation_duration)  # svake sekunde
                
                # Ažuriraj simulirane uslove
                self._update_simulated_conditions()
                
                # Log trenutne uslove
                self.logger.debug(f"Simulated conditions: "
                                f"T={self.current_conditions['temperature']:.1f}°C, "
                                f"L={self.current_conditions['luminosity']:.0f}lx")
                
        except asyncio.CancelledError:
            self.logger.info("Sensor simulation stopped")
            raise
        except Exception as e:
            self.logger.error(f"Error in sensor simulation: {e}")
    
    def _update_simulated_conditions(self):
        """
        Ažurira simulirane temperature sa FIZIČKI realističnom dinamikom.
        
        Dva režima rada:
        1. SCENARIO MODE (ako enabled u config-u): Koristi hardkodovane vrednosti za svaki ciklus
        2. DYNAMIC MODE: Simulira ekstremne događaje i drift
        """
        # SCENARIO MODE: Koristi hardkodovane vrednosti iz config-a
        if self.scenario_config and self.scenario_config.get("enabled", False):
            cycles = self.scenario_config.get("cycles", [])
            
            # Ako imamo definisan ciklus, primeni te vrednosti
            if 0 <= self.current_cycle < len(cycles):
                cycle_data = cycles[self.current_cycle]
                location_key = self.sensor_config.location.lower().replace(" ", "_")
                
                if location_key in cycle_data:
                    sensor_data = cycle_data[location_key]
                    self.current_conditions['temperature'] = sensor_data.get('temperature', self.current_conditions['temperature'])
                    self.current_conditions['luminosity'] = sensor_data.get('luminosity', self.current_conditions['luminosity'])
                    self.current_conditions['last_update'] = datetime.now()
                    return
        
        # DYNAMIC MODE: Originalna simulacija sa ekstremima
        # Vreme od poslednjeg ažuriranja
        now = datetime.now()
        time_diff = (now - self.current_conditions['last_update']).total_seconds()
        
        curr_temp = self.current_conditions['temperature']
        curr_lum = self.current_conditions['luminosity']
        
        # Temperatura se menja zbog:
        # 1. Spoljašnjih faktora (ambient drift) - lagana tendencija ka nekoj "prirodnoj" temperaturi
        # 2. EKSTREMNIH dogadjaja (kuvanje, otvoreni prozor, sunce)
        # 3. Šuma (merenja, varijacije)
        
        # Training mode: Statična simulacija (samo šum)
        if self.is_training_mode:
            temp_noise = np.random.normal(0, self.sensor_config.noise_std * 0.1)
            lum_noise = np.random.normal(0, self.sensor_config.noise_std * 2)
            
            new_temp = curr_temp + temp_noise
            new_lum = curr_lum + lum_noise
        
        # Real-time mode: Dinamička simulacija sa EKSTREMIMA
        else:
            temp_min, temp_max = self.sensor_config.temp_range
            
            # === EKSTREMNI SCENARIJI (nasumično se dogadjaju) ===
            # Svaki senzor može doživeti ekstremne uslove
            extreme_event_chance = 0.15  # 15% šanse za ekstrem po update-u
            extreme_modifier = 0.0
            
            if np.random.random() < extreme_event_chance:
                # Kuhinja: kuvanje (35°C)
                if "kitchen" in self.sensor_config.location.lower() or "03" in self.sensor_config.sensor_id:
                    extreme_modifier = np.random.uniform(3, 8) * time_diff  # Brzo zagrevanje
                    curr_lum = min(curr_lum + 100, self.sensor_config.luminosity_range[1])
                
                # Dnevna soba / Spavaća: otvoreni prozor zimi (16-17°C)
                elif "living" in self.sensor_config.location.lower() or "bedroom" in self.sensor_config.location.lower():
                    extreme_modifier = np.random.uniform(-5, -3) * time_diff  # Brzo hladjenje
                
                # Kupatilo: vrući tuš (28-30°C)
                elif "bathroom" in self.sensor_config.location.lower() or "05" in self.sensor_config.sensor_id:
                    extreme_modifier = np.random.uniform(2, 5) * time_diff
                    curr_lum = max(curr_lum - 50, self.sensor_config.luminosity_range[0])
                
                # Kancelarija: sunce kroz prozor (25-27°C)
                elif "office" in self.sensor_config.location.lower() or "04" in self.sensor_config.sensor_id:
                    extreme_modifier = np.random.uniform(1, 3) * time_diff
                    curr_lum = min(curr_lum + 150, self.sensor_config.luminosity_range[1])
            
            # === NORMALNA DINAMIKA (drift ka baznoj temp) ===
            # Odredi baznu temperaturu na osnovu lokacije
            if "01" in self.sensor_config.sensor_id or "02" in self.sensor_config.sensor_id:
                base_temp = temp_min + (temp_max - temp_min) * 0.35
            elif "04" in self.sensor_config.sensor_id:
                base_temp = (temp_min + temp_max) / 2
            else:
                base_temp = temp_min + (temp_max - temp_min) * 0.65
            
            # Drift ka baznoj temp (sporiji od ekstrema)
            drift_rate = 0.10  # Sporiji drift da omogući ekstreme
            drift = (base_temp - curr_temp) * drift_rate * time_diff
            
            # Dodaj varijacije (weather effects)
            temp_variation = np.random.normal(0, 0.8) * time_diff  # VELIKE varijacije
            temp_noise = np.random.normal(0, self.sensor_config.noise_std * 0.3)
            
            # Osvetljenost: dnevna varijacija + šum
            lum_step = np.random.uniform(-15, 15) * time_diff  # Veće promene
            lum_noise = np.random.normal(0, self.sensor_config.noise_std * 5)
            
            # FINALNA TEMPERATURA = drift + ekstrem + varijacije + šum
            new_temp = curr_temp + drift + extreme_modifier + temp_variation + temp_noise
            new_lum = curr_lum + lum_step + lum_noise
        
        # Ograniči na dozvoljene opsege
        self.current_conditions['temperature'] = np.clip(
            new_temp, self.sensor_config.temp_range[0], self.sensor_config.temp_range[1]
        )
        self.current_conditions['luminosity'] = np.clip(
            new_lum, self.sensor_config.luminosity_range[0], self.sensor_config.luminosity_range[1]
        )
        self.current_conditions['last_update'] = now
    
    async def handle_message(self, message: Message):
        """
        Obrađuje primljene poruke.
        
        Args:
            message: primljena poruka
        """
        if isinstance(message, StartTraining):
            await self._handle_start_training(message)
        elif isinstance(message, GlobalModelUpdate):
            await self._handle_global_model_update(message)
        elif isinstance(message, CollectProposals):
            await self._handle_collect_proposals(message)
        elif isinstance(message, PredictRequest):
            await self._handle_predict_request(message)
        else:
            self.logger.warning(f"Unknown message type: {type(message)}")
    
    async def _handle_start_training(self, message: StartTraining):
        """Obrađuje StartTraining poruku - pokreće lokalno treniranje."""
        self.logger.info(f"Starting local training for federation round {self.federation_round + 1}")
        
        try:
            # Treniraj lokalni model
            mse = await self._train_local_model()
            
            # Pošalji model update Coordinator-u
            await self._send_model_update(mse)
            
            self.federation_round += 1
            self.is_training_mode = True
            
        except Exception as e:
            self.logger.error(f"Error during local training: {e}")
    
    async def _train_local_model(self) -> float:
        """
        Trenira lokalni model na svojим podacima.
        
        Returns:
            MSE greška na training podacima
        """
        self.logger.info("Training local model")
        
        # Konvertuj podatke u numpy nizove
        temperatures = np.array(self.training_data['temperatures'])
        luminosities = np.array(self.training_data['luminosities'])
        commands = np.array(self.training_data['commands'])
        
        # Treniraj model
        mse = self.model.train(
            temperatures, luminosities, commands,
            epochs=config.federation.local_epochs
        )
        
        self.local_mse = mse
        self.logger.info(f"Local training completed. MSE: {mse:.4f}")
        
        return mse
    
    async def _send_model_update(self, mse: float):
        """
        Šalje ModelUpdate poruku Coordinator-u sa lokalnim parametrima.
        
        Args:
            mse: lokalna MSE greška
        """
        weights, bias, num_samples = self.model.get_parameters()
        
        # Kreiraj ModelUpdate poruku
        model_update = ModelUpdate(
            timestamp=datetime.now(),
            sender_id=self.actor_id,
            receiver_id="coordinator",
            weights=weights,
            bias=bias,
            num_samples=len(self.training_data['temperatures']),
            mse=mse
        )
        
        # Pošalji Coordinator-u
        await self.send_message(model_update, self.coordinator_host, self.coordinator_port)
        
        self.logger.info(f"Sent ModelUpdate: weights={weights}, bias={bias:.2f}, "
                        f"samples={num_samples}, mse={mse:.4f}")
    
    async def _handle_global_model_update(self, message: GlobalModelUpdate):
        """
        Obrađuje GlobalModelUpdate - ažurira lokalni model sa globalnim parametrima.
        
        Args:
            message: GlobalModelUpdate poruka sa globalnim parametrima
        """
        self.logger.info(f"Received global model update for round {message.round_number}")
        
        # Ažuriraj lokalni model
        self.model.set_parameters(message.global_weights, message.global_bias)
        
        self.logger.info(f"Updated local model: weights={message.global_weights}, "
                        f"bias={message.global_bias:.2f}")
        
        # Ako je ovo poslednja runda, pokreni real-time proposal task
        if message.round_number >= config.federation.num_rounds:
            self.logger.info("Federation training completed. Starting real-time proposals.")
            self.is_training_mode = False
            # Resetuj ciklus na 0 na početku real-time mode-a
            self.current_cycle = 0
            self.proposal_task = asyncio.create_task(self._proposal_loop())
    
    async def _handle_collect_proposals(self, message: CollectProposals):
        """
        Obrađuje CollectProposals - šalje svoj predlog komande.
        Takođe inkrementira ciklus za scenario mode.
        
        Args:
            message: CollectProposals poruka
        """
        # Generiši predlog na osnovu trenutnih uslova
        proposal = self.model.predict(
            self.current_conditions['temperature'],
            self.current_conditions['luminosity']
        )
        
        # Računaj confidence na osnovu lokalnej MSE
        confidence = 1.0 / (1.0 + self.local_mse) if self.local_mse < float('inf') else 0.5
        
        # INKREMENTUJ CIKLUS za sledeći poziv (scenario mode)
        if self.scenario_config and self.scenario_config.get("enabled", False):
            cycles = self.scenario_config.get("cycles", [])
            if cycles:
                self.current_cycle = (self.current_cycle + 1) % len(cycles)
                self.logger.info(f"📍 Next cycle: {self.current_cycle + 1}/{len(cycles)}")
        
        # Kreiraj i pošalji odgovor
        response = ProposalResponse(
            timestamp=datetime.now(),
            sender_id=self.actor_id,
            receiver_id=message.sender_id,
            proposal=proposal,
            temperature=self.current_conditions['temperature'],
            confidence=confidence
        )
        
        await self.send_message(response, self.coordinator_host, self.coordinator_port)
        
        self.logger.debug(f"Sent proposal: {proposal:.2f}°C (confidence: {confidence:.2f})")
    
    async def _handle_predict_request(self, message: PredictRequest):
        """
        Obrađuje PredictRequest - vraća predikciju za date uslove.
        
        Args:
            message: PredictRequest poruka
        """
        prediction = self.model.predict(message.temperature, message.luminosity)
        
        response = PredictResponse(
            timestamp=datetime.now(),
            sender_id=self.actor_id,
            receiver_id=message.sender_id,
            predicted_command=prediction,
            temperature=message.temperature,
            luminosity=message.luminosity
        )
        
        await self.send_message(response, self.coordinator_host, self.coordinator_port)
    
    async def _proposal_loop(self):
        """
        Periodično šalje predloge Coordinator-u (real-time faza).
        """
        try:
            while self.is_running and not self.is_training_mode:
                await asyncio.sleep(config.data_collection_interval)  # svake sekunde
                
                # Pošalji trenutne uslove kao SensorData
                sensor_data = SensorData(
                    timestamp=datetime.now(),
                    sender_id=self.actor_id,
                    receiver_id="coordinator",
                    temperature=self.current_conditions['temperature'],
                    luminosity=self.current_conditions['luminosity']
                )
                
                await self.send_message(sensor_data, self.coordinator_host, self.coordinator_port)
                
        except asyncio.CancelledError:
            self.logger.info("Proposal loop stopped")
            raise
        except Exception as e:
            self.logger.error(f"Error in proposal loop: {e}")
    
    def get_status(self) -> Dict:
        """
        Vraća trenutni status senzora.
        
        Returns:
            Dict sa statusom senzora
        """
        return {
            'sensor_id': self.sensor_config.sensor_id,
            'location': self.sensor_config.location,
            'current_temperature': self.current_conditions['temperature'],
            'current_luminosity': self.current_conditions['luminosity'],
            'is_training': self.is_training_mode,
            'federation_round': self.federation_round,
            'local_mse': self.local_mse,
            'model_trained': self.model.is_trained,
            'num_training_samples': len(self.training_data['temperatures'])
        }