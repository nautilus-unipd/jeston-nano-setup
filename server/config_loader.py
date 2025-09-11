import json
import os
from typing import Dict, Any

class ConfigLoader:
    """Base class per il caricamento della configurazione JSON"""
    _config_data = None
    _config_file = "config.json"
    
    @classmethod
    def load_config(cls) -> Dict[str, Any]:
        """Carica la configurazione dal file JSON"""
        if cls._config_data is None:
            config_path = os.path.join(os.path.dirname(__file__), cls._config_file)
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cls._config_data = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"File di configurazione {config_path} non trovato")
            except json.JSONDecodeError as e:
                raise ValueError(f"Errore nel parsing del file JSON: {e}")
        return cls._config_data
    
    @classmethod
    def get_nested_value(cls, *keys):
        """Ottiene un valore annidato dalla configurazione"""
        config = cls.load_config()
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyError(f"Chiave non trovata: {'.'.join(keys)}")
        return value