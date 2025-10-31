"""
FedAvg (Federated Averaging) algoritam za agregaciju lokalnih modela.
"""
import numpy as np
from typing import List, Tuple
from ..utils.messages import ModelUpdate


class FedAvg:
    """
    Implementacija Federated Averaging algoritma.
    Agreguje lokalne parametre modela u globalni model ponderisano brojem uzoraka.
    """
    
    def __init__(self):
        self.global_weights = None
        self.global_bias = None
        self.round_number = 0
        self.history = {
            'global_weights': [],
            'global_bias': [],
            'num_participants': [],
            'total_samples': []
        }
    
    def aggregate_models(self, model_updates: List[ModelUpdate]) -> Tuple[np.ndarray, float]:
        """
        Agreguje lokalne model update-e u globalni model.
        
        Args:
            model_updates: Lista ModelUpdate poruka od senzora
            
        Returns:
            (global_weights, global_bias)
        """
        if not model_updates:
            raise ValueError("Nema model update-a za agregaciju")
        
        # Izdvoji podatke
        local_weights = []
        local_biases = []
        sample_counts = []
        
        for update in model_updates:
            local_weights.append(update.weights)
            local_biases.append(update.bias)
            sample_counts.append(update.num_samples)
        
        # Konvertuj u numpy nizove
        local_weights = np.array(local_weights)  # shape: (num_clients, num_features)
        local_biases = np.array(local_biases)    # shape: (num_clients,)
        sample_counts = np.array(sample_counts)  # shape: (num_clients,)
        
        # Računaj ponderisane proseke
        total_samples = np.sum(sample_counts)
        weights_normalized = sample_counts / total_samples
        
        # Agreguj težine - ponderisan prosek po kolonama
        self.global_weights = np.average(local_weights, weights=weights_normalized, axis=0)
        
        # Agreguj bias
        self.global_bias = np.average(local_biases, weights=weights_normalized)
        
        # Ažuriraj istoriju
        self.round_number += 1
        self.history['global_weights'].append(self.global_weights.copy())
        self.history['global_bias'].append(float(self.global_bias))
        self.history['num_participants'].append(len(model_updates))
        self.history['total_samples'].append(int(total_samples))
        
        return self.global_weights.copy(), float(self.global_bias)
    
    def get_global_model(self) -> Tuple[np.ndarray, float]:
        """
        Vraća trenutni globalni model.
        
        Returns:
            (global_weights, global_bias)
        """
        if self.global_weights is None:
            # Inicijalne vrednosti ako nema globalnog modela
            return np.array([0.0, 0.0]), 23.0
        
        return self.global_weights.copy(), float(self.global_bias)
    
    def check_convergence(self, threshold: float = 0.001) -> bool:
        """
        Proverava konvergenciju na osnovu promene globalnih parametara.
        
        Args:
            threshold: prag za konvergenciju
            
        Returns:
            True ako je model konvergirao
        """
        if len(self.history['global_weights']) < 2:
            return False
        
        # Poređaj poslednje dve iteracije
        prev_weights = self.history['global_weights'][-2]
        curr_weights = self.history['global_weights'][-1]
        prev_bias = self.history['global_bias'][-2]
        curr_bias = self.history['global_bias'][-1]
        
        # Računaj promene
        weights_change = np.linalg.norm(curr_weights - prev_weights)
        bias_change = abs(curr_bias - prev_bias)
        
        return weights_change < threshold and bias_change < threshold
    
    def get_aggregation_stats(self) -> dict:
        """
        Vraća statistike agregacije.
        
        Returns:
            Rečnik sa statistikama poslednje runde
        """
        if self.round_number == 0:
            return {}
        
        return {
            'round_number': self.round_number,
            'global_weights': self.global_weights.tolist(),
            'global_bias': float(self.global_bias),
            'num_participants': self.history['num_participants'][-1],
            'total_samples': self.history['total_samples'][-1],
            'converged': self.check_convergence()
        }
    
    def reset(self):
        """Resetuje agregator za novu sesiju."""
        self.global_weights = None
        self.global_bias = None
        self.round_number = 0
        self.history = {
            'global_weights': [],
            'global_bias': [],
            'num_participants': [],
            'total_samples': []
        }