"""
Loader za učitavanje konfiguracije iz JSON fajla.
"""
import json
import os
from .config import SystemConfig


def load_config_from_json(json_path: str = "config/system_config.json") -> SystemConfig:
    """
    Učitava konfiguraciju iz JSON fajla i overrides default vrednosti.
    
    Args:
        json_path: Putanja do JSON fajla sa konfiguracijom
        
    Returns:
        SystemConfig instanca sa učitanim vrednostima iz JSON-a
    """
    # Default config
    cfg = SystemConfig()
    
    if not os.path.exists(json_path):
        print(f"[WARNING] Config file not found: {json_path}, using defaults")
        return cfg  # Vrati default ako nema JSON-a
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Override control parametara
    if 'control' in data:
        control = data['control']
        if 'mode_threshold_celsius' in control:
            cfg.control.mode_threshold = control['mode_threshold_celsius']
        if 'min_on_time_seconds' in control:
            cfg.control.min_on_time = control['min_on_time_seconds']
        if 'min_off_time_seconds' in control:
            cfg.control.min_off_time = control['min_off_time_seconds']
        print(
            f"[CONFIG] Loaded control params: mode_threshold={cfg.control.mode_threshold}, "
            f"min_on_time={cfg.control.min_on_time}s, min_off_time={cfg.control.min_off_time}s"
        )
    
    return cfg
