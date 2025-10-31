"""
Linearni regresioni model za predikciju komande uređaja.
Model: Y_cmd = w1 * T + w2 * L + b
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from typing import Tuple, Optional
import pickle
import json


class HVACModel:
    """
    Linearni model za predikciju komande HVAC uređaja na osnovu temperature i osvetljenosti.
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
        self.weights = None  # [w1, w2] za T i L
        self.bias = None     # b
        
    def train(self, temperatures: np.ndarray, luminosities: np.ndarray, 
              commands: np.ndarray, epochs: int = 10) -> float:
        """
        Trenira model na lokalnim podacima.
        
        Args:
            temperatures: temperatura (°C)
            luminosities: osvetljenost (lx)  
            commands: komande uređaja (°C)
            epochs: broj epoha treniranja
            
        Returns:
            MSE greška na trenirnim podacima
        """
        # Kombinuj features
        X = np.column_stack([temperatures, luminosities])
        y = commands
        
        # Treniranje
        self.model.fit(X, y)
        self.is_trained = True
        
        # Izdvoji parametre
        self.weights = self.model.coef_
        self.bias = self.model.intercept_
        
        # Računaj MSE
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        
        return mse
    
    def predict(self, temperature: float, luminosity: float) -> float:
        """
        Predikcija komande na osnovu trenutnih uslova.
        
        Args:
            temperature: temperatura (°C)
            luminosity: osvetljenost (lx)
            
        Returns:
            Predviđena komanda (°C)
        """
        if not self.is_trained:
            # Ako model nije treniran, koristi jednostavnu heuristiku
            return self._heuristic_prediction(temperature, luminosity)
        
        X = np.array([[temperature, luminosity]])
        prediction = self.model.predict(X)[0]
        
        # Ograniči na razuman opseg
        return np.clip(prediction, 16.0, 30.0)
    
    def _heuristic_prediction(self, temperature: float, luminosity: float) -> float:
        """
        Jednostavna heuristika kada model nije treniran.
        Viša temperatura -> niža komanda (hlađenje)
        Viša osvetljenost -> niža komanda (sunce greje)
        """
        base_temp = 23.0  # ciljna temperatura
        temp_factor = (30.0 - temperature) / 10.0  # normalizovano
        luminosity_factor = (800 - luminosity) / 1000.0  # normalizovano
        
        command = base_temp + temp_factor + luminosity_factor
        return np.clip(command, 16.0, 30.0)
    
    def get_parameters(self) -> Tuple[np.ndarray, float, int]:
        """
        Vraća parametre modela za federativno učenje.
        
        Returns:
            (weights, bias, num_samples)
        """
        if not self.is_trained:
            # Vraća nasumične parametre ako nije treniran
            return np.array([0.0, 0.0]), 23.0, 0
        
        return self.weights.copy(), float(self.bias), getattr(self, '_num_samples', 1)
    
    def set_parameters(self, weights: np.ndarray, bias: float):
        """
        Postavlja parametre modela (za federativno učenje).
        
        Args:
            weights: [w1, w2] težine za T i L
            bias: b konstanta
        """
        self.weights = weights.copy()
        self.bias = bias
        
        # Ručno postavi parametre sklearn modela
        self.model.coef_ = self.weights
        self.model.intercept_ = self.bias
        self.is_trained = True
    
    def save_model(self, filepath: str):
        """Čuva model u fajl."""
        model_data = {
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': self.bias,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath: str):
        """Učitava model iz fajla."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        if model_data['weights'] is not None:
            self.weights = np.array(model_data['weights'])
            self.bias = model_data['bias']
            self.is_trained = model_data['is_trained']
            
            if self.is_trained:
                self.model.coef_ = self.weights
                self.model.intercept_ = self.bias