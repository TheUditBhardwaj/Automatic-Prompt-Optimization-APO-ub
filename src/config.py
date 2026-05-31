import os
import yaml
from typing import Dict, Any

class Config:
    """Configuration loader and validator for the APO system."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.data = self._load_config()
        self._validate()

    def _load_config(self) -> Dict[str, Any]:
        """Loads configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing configuration YAML: {e}")

    def _validate(self):
        """Validates that configurations are correct and splits sum to 1.0."""
        # Check required sections
        required_keys = ['model', 'dataset', 'schema', 'output_dir']
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"Missing required config section: {key}")

        # Validate splits
        dataset_cfg = self.data['dataset']
        train = dataset_cfg.get('train_ratio', 0.6)
        val = dataset_cfg.get('val_ratio', 0.2)
        test = dataset_cfg.get('test_ratio', 0.2)
        
        if not abs((train + val + test) - 1.0) < 1e-6:
            raise ValueError(
                f"Train/Val/Test ratios must sum to 1.0. Got: {train} + {val} + {test} = {train + val + test}"
            )

    @property
    def model_name(self) -> str:
        return self.data['model'].get('name', 'gemini-1.5-flash')

    @property
    def model_temperature(self) -> float:
        return float(self.data['model'].get('temperature', 0.0))

    @property
    def model_max_output_tokens(self) -> int:
        return int(self.data['model'].get('max_output_tokens', 2048))

    @property
    def raw_dir(self) -> str:
        return self.data['dataset'].get('raw_dir', 'data/raw')

    @property
    def gold_dir(self) -> str:
        return self.data['dataset'].get('gold_dir', 'data/gold')

    @property
    def train_ratio(self) -> float:
        return float(self.data['dataset'].get('train_ratio', 0.6))

    @property
    def val_ratio(self) -> float:
        return float(self.data['dataset'].get('val_ratio', 0.2))

    @property
    def test_ratio(self) -> float:
        return float(self.data['dataset'].get('test_ratio', 0.2))

    @property
    def seed(self) -> int:
        return int(self.data['dataset'].get('seed', 42))

    @property
    def schema_type(self) -> str:
        return self.data.get('schema', 'resume')

    @property
    def output_dir(self) -> str:
        return self.data.get('output_dir', 'output')

    @property
    def opt_iterations(self) -> int:
        opt = self.data.get('optimizer', {})
        return int(opt.get('iterations', 3))

    @property
    def opt_mutation_model(self) -> str:
        opt = self.data.get('optimizer', {})
        return opt.get('mutation_model', 'gemini-2.5-flash')

    @property
    def opt_mutation_temperature(self) -> float:
        opt = self.data.get('optimizer', {})
        return float(opt.get('mutation_temperature', 0.7))

    @property
    def seed_prompt(self) -> str:
        """Optional user-provided seed prompt override from config. Returns empty string if not set."""
        return self.data.get('seed_prompt', '')
