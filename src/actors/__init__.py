"""
Bazne klase za aktore u sistemu.
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime
import json
import logging

from utils.messages import Message


class BaseActor(ABC):
    """
    Bazna klasa za sve aktore u sistemu.
    """
    
    def __init__(self, actor_id: str, host: str = "localhost", port: int = 8000):
        self.actor_id = actor_id
        self.host = host
        self.port = port
        self.is_running = False
        self.message_queue = asyncio.Queue()
        self.connections = {}  # peer_id -> (reader, writer)
        self.logger = logging.getLogger(f"Actor.{actor_id}")
        
        # Setup logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'%(asctime)s - {actor_id} - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    async def start(self):
        """Pokreće aktora."""
        self.is_running = True
        self.logger.info(f"Starting actor on {self.host}:{self.port}")
        
        # Pokreni server za primanje poruka
        server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )
        
        # Pokreni message loop
        message_task = asyncio.create_task(self._message_loop())
        
        try:
            await asyncio.gather(
                server.serve_forever(),
                message_task
            )
        except asyncio.CancelledError:
            self.logger.info("Actor stopped")
        finally:
            server.close()
            await server.wait_closed()
    
    async def stop(self):
        """Zaustavlja aktora."""
        self.is_running = False
        
        # Zatvori sve konekcije
        for peer_id, (reader, writer) in self.connections.items():
            writer.close()
            await writer.wait_closed()
        
        self.connections.clear()
        self.logger.info("Actor stopped")
    
    async def send_message(self, message: Message, target_host: str, target_port: int):
        """
        Šalje poruku drugom aktoru.
        
        Args:
            message: poruka za slanje
            target_host: IP adresa ciljnog aktora
            target_port: port ciljnog aktora
        """
        try:
            # Otvori konekciju
            reader, writer = await asyncio.open_connection(target_host, target_port)
            
            # Serijalizuj poruku
            message_data = self._serialize_message(message)
            message_bytes = message_data.encode('utf-8')
            
            # Pošalji dužinu pa poruku
            writer.write(len(message_bytes).to_bytes(4, 'big'))
            writer.write(message_bytes)
            await writer.drain()
            
            # Zatvori konekciju
            writer.close()
            await writer.wait_closed()
            
            self.logger.debug(f"Sent message to {target_host}:{target_port}")
            
        except Exception as e:
            self.logger.error(f"Failed to send message to {target_host}:{target_port}: {e}")
    
    async def _handle_client(self, reader, writer):
        """Rukuje dolaznim konekcijama."""
        try:
            # Pročitaj dužinu poruke
            length_bytes = await reader.read(4)
            if not length_bytes:
                return
            
            message_length = int.from_bytes(length_bytes, 'big')
            
            # Pročitaj poruku
            message_data = await reader.read(message_length)
            message_str = message_data.decode('utf-8')
            
            # Deserijalizuj poruku
            message = self._deserialize_message(message_str)
            
            # Dodaj u queue
            await self.message_queue.put(message)
            
        except Exception as e:
            self.logger.error(f"Error handling client: {e}")
        finally:
            writer.close()
    
    async def _message_loop(self):
        """Glavna petlja za obradu poruka."""
        while self.is_running:
            try:
                # Čekaj poruku sa timeout-om
                message = await asyncio.wait_for(
                    self.message_queue.get(), timeout=1.0
                )
                
                # Obradi poruku
                await self.handle_message(message)
                
            except asyncio.TimeoutError:
                # Timeout je ok, samo nastavi
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
    
    @abstractmethod
    async def handle_message(self, message: Message):
        """
        Obrađuje primljenu poruku. Mora biti implementirano u nasleđenim klasama.
        
        Args:
            message: primljena poruka
        """
        pass
    
    def _serialize_message(self, message: Message) -> str:
        """Serijalizuje poruku u JSON string."""
        # Kreiraaj rečnik sa tipom i podacima poruke
        message_dict = {
            'type': message.__class__.__name__,
            'data': message.__dict__.copy()
        }
        
        # Konvertuj datetime objekte u ISO string
        for key, value in message_dict['data'].items():
            if isinstance(value, datetime):
                message_dict['data'][key] = value.isoformat()
            elif hasattr(value, 'tolist'):  # numpy array
                message_dict['data'][key] = value.tolist()
        
        return json.dumps(message_dict)
    
    def _deserialize_message(self, message_str: str) -> Message:
        """Deserijalizuje JSON string u poruku."""
        from utils.messages import (
            StartTraining, ModelUpdate, GlobalModelUpdate,
            PredictRequest, PredictResponse, CollectProposals,
            ProposalResponse, ApplyCommand, LogMetrics, LogEvent, SensorData
        )
        
        # Mapa tipova poruka
        MESSAGE_TYPES = {
            'StartTraining': StartTraining,
            'ModelUpdate': ModelUpdate,
            'GlobalModelUpdate': GlobalModelUpdate,
            'PredictRequest': PredictRequest,
            'PredictResponse': PredictResponse,
            'CollectProposals': CollectProposals,
            'ProposalResponse': ProposalResponse,
            'ApplyCommand': ApplyCommand,
            'LogMetrics': LogMetrics,
            'LogEvent': LogEvent,
            'SensorData': SensorData,
        }
        
        message_dict = json.loads(message_str)
        message_type = MESSAGE_TYPES[message_dict['type']]
        data = message_dict['data']
        
        # Konvertuj datetime stringove nazad u datetime objekte
        for key, value in data.items():
            if key == 'timestamp' and isinstance(value, str):
                data[key] = datetime.fromisoformat(value)
            elif key in ['weights', 'global_weights'] and isinstance(value, list):
                import numpy as np
                data[key] = np.array(value)
        
        return message_type(**data)