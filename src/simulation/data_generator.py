"""
Generisanje test podataka za simulaciju HVAC sistema.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple
import json


class DataGenerator:
    """
    Generiše simulacione podatke za HVAC sistem.
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed
    
    def generate_sensor_data(self, sensor_id: str, location: str,
                           temp_range: Tuple[float, float] = (18, 32),
                           luminosity_range: Tuple[float, float] = (100, 800),
                           duration_hours: int = 24,
                           interval_minutes: int = 5,
                           noise_std: float = 0.5) -> pd.DataFrame:
        """
        Generiše vremenske serije podataka za jedan senzor.
        
        Args:
            sensor_id: ID senzora
            location: lokacija senzora
            temp_range: opseg temperatura (min, max) °C
            luminosity_range: opseg osvetljenosti (min, max) lx
            duration_hours: trajanje simulacije u satima
            interval_minutes: interval merenja u minutima
            noise_std: standardna devijacija šuma
            
        Returns:
            DataFrame sa kolonama: timestamp, temperature, luminosity, y_cmd
        """
        # Generiši vremenske oznake
        start_time = datetime.now()
        num_points = int(duration_hours * 60 / interval_minutes)
        timestamps = [start_time + timedelta(minutes=i * interval_minutes) 
                     for i in range(num_points)]
        
        # Generiši baznu temperaturu (sinusoidala za dnevni ciklus)
        hours = np.array([(ts.hour + ts.minute/60) for ts in timestamps])
        temp_base = (temp_range[0] + temp_range[1]) / 2
        temp_amplitude = (temp_range[1] - temp_range[0]) / 4
        
        # Dnevni ciklus temperature (min ujutru, max popodne)
        temperatures = temp_base + temp_amplitude * np.sin(2 * np.pi * (hours - 6) / 24)
        
        # Generiši osvetljenost (korelisana sa vremenom dana)
        lum_base = (luminosity_range[0] + luminosity_range[1]) / 2
        lum_amplitude = (luminosity_range[1] - luminosity_range[0]) / 3
        
        # Osvetljenost zavisi od vremena (veća danju)
        luminosities = lum_base + lum_amplitude * np.maximum(0, np.sin(2 * np.pi * (hours - 6) / 24))
        
        # Dodaj šum
        temperatures += np.random.normal(0, noise_std, len(temperatures))
        luminosities += np.random.normal(0, noise_std * 10, len(luminosities))
        
        # Ograniči na dozvoljene opsege
        temperatures = np.clip(temperatures, temp_range[0], temp_range[1])
        luminosities = np.clip(luminosities, luminosity_range[0], luminosity_range[1])
        
        # Generiš ideal komande (simulira obrnuto proporcionalno ponašanje)
        # Viša temperatura -> niža komanda (hlađenje)
        # Viša osvetljenost -> niža komanda (sunce greje prostor)
        y_cmd = self._generate_ideal_commands(temperatures, luminosities, location)
        
        # Kreiraj DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'sensor_id': sensor_id,
            'location': location,
            'temperature': temperatures,
            'luminosity': luminosities,
            'y_cmd': y_cmd
        })
        
        return df
    
    def _generate_ideal_commands(self, temperatures: np.ndarray, 
                               luminosities: np.ndarray, 
                               location: str) -> np.ndarray:
        """
        Generiše idealne komande na osnovu temperature i osvetljenosti.
        """
        # Bazni setpoint zavisi od lokacije
        location_bias = {
            'living_room': 23.0,
            'bedroom': 22.0, 
            'kitchen': 24.0,
            'office': 23.5,
            'bathroom': 25.0
        }
        
        base_temp = location_bias.get(location, 23.0)
        
        # Linearni model: Y_cmd = base - k1*T - k2*L + noise
        k1 = 0.3  # koeficijent za temperaturu
        k2 = 0.002  # koeficijent za osvetljenost
        
        # Normalizuj temperature i osvetljenost
        temp_norm = (temperatures - 25.0) / 10.0
        lum_norm = (luminosities - 400.0) / 300.0
        
        commands = base_temp - k1 * temp_norm - k2 * lum_norm
        
        # Dodaj mali šum
        commands += np.random.normal(0, 0.2, len(commands))
        
        # Ograniči na razuman opseg
        return np.clip(commands, 16.0, 30.0)
    
    def generate_training_dataset(self, num_samples: int = 100) -> pd.DataFrame:
        """
        Generiše sintetički dataset za treniranje.
        
        Args:
            num_samples: broj uzoraka
            
        Returns:
            DataFrame sa training podacima
        """
        # Generiši nasumične temperature i osvetljenosti
        temperatures = np.random.uniform(18, 32, num_samples)
        luminosities = np.random.uniform(100, 800, num_samples)
        
        # Generiš komande na osnovu realističnog modela
        y_cmd = self._generate_realistic_commands(temperatures, luminosities)
        
        # Kreiraj DataFrame
        df = pd.DataFrame({
            'temperature': temperatures,
            'luminosity': luminosities,
            'y_cmd': y_cmd
        })
        
        return df
    
    def _generate_realistic_commands(self, temperatures: np.ndarray,
                                   luminosities: np.ndarray) -> np.ndarray:
        """
        Generiš realistične komande na osnovu fizičkih principa.
        """
        # Kompleksniji model koji simulira pravo ponašanje
        base_setpoint = 23.0
        
        # Temperature factor: viša temp -> niži setpoint (hlađenje)
        temp_factor = (30.0 - temperatures) / 12.0
        
        # Luminosity factor: više svetla -> niži setpoint (sunce greje)
        lum_factor = (600.0 - luminosities) / 500.0
        
        # Kombinuj faktore
        commands = base_setpoint + 0.7 * temp_factor + 0.3 * lum_factor
        
        # Dodaj realistični šum
        commands += np.random.normal(0, 0.5, len(commands))
        
        return np.clip(commands, 16.0, 30.0)
    
    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """Čuva dataset u JSON fajl."""
        # Konvertuj timestamp u string za JSON serijalizaciju
        df_copy = df.copy()
        if 'timestamp' in df_copy.columns:
            df_copy['timestamp'] = df_copy['timestamp'].dt.isoformat()
        
        df_copy.to_json(filepath, orient='records', indent=2)
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """Učitava dataset iz JSON fajla."""
        df = pd.read_json(filepath, orient='records')
        
        # Konvertuj timestamp nazad u datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df