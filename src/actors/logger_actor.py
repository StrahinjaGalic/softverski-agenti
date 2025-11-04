"""
LoggerActor - Beleži metrike, događaje i omogućava vizualizaciju.

LoggerActor je odgovoran za:
1. Prijem LogMetrics poruka (MSE, runde, komande, itd.)
2. Prijem LogEvent poruka (mode_change, training_complete, itd.)
3. Perzistenciju logova u JSON
4. Mogućnost vizualizacije (grafici)
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, List
import logging

from actors import BaseActor
from utils.messages import Message, LogMetrics, LogEvent


class LoggerActor(BaseActor):
    """
    Aktor za centralizovano logovanje metrika i događaja.
    """
    
    def __init__(self, log_file: str = "logs/system_log.json"):
        """
        Args:
            log_file: Putanja do fajla za perzistenciju logova
        """
        super().__init__("logger", "localhost", 8002)
        
        self.log_file = log_file
        
        # Kolekcije metrika i događaja
        self.metrics: Dict[str, List[dict]] = {
            'aggregation': [],
            'mse': [],
            'commands': [],
            'temperatures': [],
            'modes': []
        }
        
        self.events: List[dict] = []
        
        self.logger.info("LoggerActor initialized")
    
    async def handle_message(self, message: Message):
        """
        Obrađuje primljene poruke.
        
        Message types:
        - LogMetrics: Metrike sa vrednostima
        - LogEvent: Događaji u sistemu
        """
        if isinstance(message, LogMetrics):
            await self._handle_log_metrics(message)
        elif isinstance(message, LogEvent):
            await self._handle_log_event(message)
        else:
            self.logger.warning(f"Unknown message type: {type(message)}")
    
    async def _handle_log_metrics(self, message: LogMetrics):
        """
        Beleži metriku.
        
        Metrika sadrži:
        - metric_type: Tip metrike (MSE, accuracy, itd.)
        - value: Vrednost metrike
        - round_number: Opcionalno, runda federacije
        - data: Dodatni podaci (dict)
        """
        entry = {
            'timestamp': message.timestamp.isoformat(),
            'sender': message.sender_id,
            'type': message.metric_type,
            'value': message.value,
            'round': message.round_number,
            'data': getattr(message, 'data', None)
        }
        
        # Dodaj u odgovarajuću kolekciju
        if message.metric_type in self.metrics:
            self.metrics[message.metric_type].append(entry)
        else:
            # Kreiraj novu kolekciju ako ne postoji
            self.metrics[message.metric_type] = [entry]
        
        self.logger.debug(f"📊 Logged metric: {message.metric_type} = {message.value}")
    
    async def _handle_log_event(self, message: LogEvent):
        """
        Beleži događaj.
        
        Događaj sadrži:
        - event_type: Tip događaja (mode_change, training_complete, itd.)
        - description: Opis događaja
        - data: Dodatni podaci (dict)
        """
        entry = {
            'timestamp': message.timestamp.isoformat(),
            'sender': message.sender_id,
            'event_type': message.event_type,
            'description': message.description,
            'data': message.data
        }
        
        self.events.append(entry)
        
        self.logger.info(f"📝 Event: {message.event_type} - {message.description}")
    
    async def save_logs(self):
        """Čuva sve logove u JSON fajl."""
        import os
        
        # Kreiraj direktorijum ako ne postoji
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        log_data = {
            'metrics': self.metrics,
            'events': self.events,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.logger.info(f"💾 Logs saved to {self.log_file}")
    
    def get_status(self) -> Dict:
        """Vraća trenutni status logger-a."""
        total_metrics = sum(len(v) for v in self.metrics.values())
        
        return {
            'actor_id': self.actor_id,
            'total_metrics': total_metrics,
            'total_events': len(self.events),
            'metric_types': list(self.metrics.keys()),
            'log_file': self.log_file
        }
    
    def get_summary(self) -> str:
        """Vraća tekstualni summary logova."""
        lines = []
        lines.append("="*60)
        lines.append("LOGGER SUMMARY")
        lines.append("="*60)
        
        # Metrike
        lines.append(f"\n📊 METRICS:")
        for metric_type, entries in self.metrics.items():
            lines.append(f"   {metric_type}: {len(entries)} entries")
        
        # Događaji
        lines.append(f"\n📝 EVENTS: {len(self.events)} total")
        for event in self.events[-5:]:  # Poslednjih 5
            lines.append(f"   [{event['timestamp']}] {event['event_type']}: {event['description']}")
        
        return "\n".join(lines)
