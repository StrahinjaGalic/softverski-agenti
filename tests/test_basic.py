"""
Osnovni test fajl za proveru funkcionalnosti sistema.
"""
import pytest
import asyncio
import numpy as np
from datetime import datetime

# Import components to test
from src.federation.model import HVACModel
from src.federation.fedavg import FedAvg
from src.utils.messages import ModelUpdate
from src.simulation.data_generator import DataGenerator


class TestHVACModel:
    """Testovi za HVAC model."""
    
    def test_model_initialization(self):
        """Test kreiranje modela."""
        model = HVACModel()
        assert not model.is_trained
        assert model.weights is None
        assert model.bias is None
    
    def test_heuristic_prediction(self):
        """Test heurističke predikcije."""
        model = HVACModel()
        
        # Test sa tipičnim vrednostima
        prediction = model.predict(25.0, 400.0)
        assert 16.0 <= prediction <= 30.0
        
        # Test sa visokom temperaturom (treba manju komandu)
        high_temp_pred = model.predict(30.0, 400.0)
        low_temp_pred = model.predict(20.0, 400.0)
        assert high_temp_pred < low_temp_pred
    
    def test_model_training(self):
        """Test treniranja modela."""
        model = HVACModel()
        
        # Generiši test podatke
        temperatures = np.array([20, 22, 24, 26, 28])
        luminosities = np.array([200, 300, 400, 500, 600])
        commands = np.array([26, 24, 22, 20, 18])
        
        # Treniraj model
        mse = model.train(temperatures, luminosities, commands)
        
        assert model.is_trained
        assert model.weights is not None
        assert model.bias is not None
        assert mse >= 0
    
    def test_parameter_operations(self):
        """Test get/set parametara."""
        model = HVACModel()
        
        # Test set parametara
        weights = np.array([0.5, -0.001])
        bias = 23.0
        model.set_parameters(weights, bias)
        
        # Test get parametara
        get_weights, get_bias, samples = model.get_parameters()
        np.testing.assert_array_equal(get_weights, weights)
        assert get_bias == bias


class TestFedAvg:
    """Testovi za FedAvg algoritam."""
    
    def test_aggregation_basic(self):
        """Test osnovne agregacije."""
        fedavg = FedAvg()
        
        # Kreiraaj test model update-e
        updates = [
            ModelUpdate(
                timestamp=datetime.now(),
                sender_id="sensor1",
                receiver_id="coordinator",
                weights=np.array([1.0, 0.5]),
                bias=20.0,
                num_samples=10,
                mse=0.5
            ),
            ModelUpdate(
                timestamp=datetime.now(),
                sender_id="sensor2", 
                receiver_id="coordinator",
                weights=np.array([2.0, 1.0]),
                bias=25.0,
                num_samples=20,
                mse=0.3
            )
        ]
        
        # Agreguraj
        global_weights, global_bias = fedavg.aggregate_models(updates)
        
        # Proveri rezultate (ponderisano po broju uzoraka)
        expected_weights = (10*np.array([1.0, 0.5]) + 20*np.array([2.0, 1.0])) / 30
        expected_bias = (10*20.0 + 20*25.0) / 30
        
        np.testing.assert_array_almost_equal(global_weights, expected_weights)
        assert abs(global_bias - expected_bias) < 1e-6
    
    def test_convergence_check(self):
        """Test provere konvergencije."""
        fedavg = FedAvg()
        
        # Prva agregacija
        updates1 = [ModelUpdate(
            timestamp=datetime.now(),
            sender_id="sensor1",
            receiver_id="coordinator", 
            weights=np.array([1.0, 0.5]),
            bias=20.0,
            num_samples=10,
            mse=0.5
        )]
        fedavg.aggregate_models(updates1)
        
        # Druga agregacija (slična)
        updates2 = [ModelUpdate(
            timestamp=datetime.now(),
            sender_id="sensor1",
            receiver_id="coordinator",
            weights=np.array([1.001, 0.501]),  # minimalna promena
            bias=20.001,
            num_samples=10,
            mse=0.49
        )]
        fedavg.aggregate_models(updates2)
        
        # Trebalo bi da konvergira
        assert fedavg.check_convergence(threshold=0.01)


class TestDataGenerator:
    """Testovi za generisanje podataka."""
    
    def test_training_dataset_generation(self):
        """Test generisanja training dataseta."""
        generator = DataGenerator(seed=42)
        
        df = generator.generate_training_dataset(num_samples=50)
        
        assert len(df) == 50
        assert 'temperature' in df.columns
        assert 'luminosity' in df.columns  
        assert 'y_cmd' in df.columns
        
        # Proveri opsege
        assert df['temperature'].min() >= 18
        assert df['temperature'].max() <= 32
        assert df['luminosity'].min() >= 100
        assert df['luminosity'].max() <= 800
        assert df['y_cmd'].min() >= 16
        assert df['y_cmd'].max() <= 30
    
    def test_sensor_data_generation(self):
        """Test generisanja senzorskih podataka."""
        generator = DataGenerator(seed=42)
        
        df = generator.generate_sensor_data(
            sensor_id="test_sensor",
            location="test_room",
            duration_hours=1,  # kratko za test
            interval_minutes=10
        )
        
        assert len(df) == 6  # 1 sat sa 10min intervalima
        assert df['sensor_id'].iloc[0] == "test_sensor"
        assert df['location'].iloc[0] == "test_room"
        
        # Proveri da su timestamp-ovi u rastućem redosledu
        timestamps = df['timestamp'].values
        assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))


# Test runner
if __name__ == "__main__":
    pytest.main([__file__])