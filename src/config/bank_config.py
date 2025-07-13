import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class BankConfig:
    """Manages bank-specific configurations and patterns."""
    
    def __init__(self, base_path: Optional[Path] = None):
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent
        self.base_path = base_path
        self.banks_config_dir = base_path / "src" / "config" / "banks"
        self.bank_statements_dir = base_path / "bank_statements"
        self._bank_configs = {}
        self._load_all_bank_configs()
    
    def _load_all_bank_configs(self):
        """Load all available bank configurations."""
        if not self.banks_config_dir.exists():
            print(f"Warning: Banks config directory not found: {self.banks_config_dir}")
            return
            
        for config_file in self.banks_config_dir.glob("*.json"):
            bank_name = config_file.stem
            try:
                with open(config_file, 'r') as f:
                    self._bank_configs[bank_name] = json.load(f)
                print(f"Loaded bank configuration: {bank_name}")
            except Exception as e:
                print(f"Error loading bank config for {bank_name}: {str(e)}")
    
    def get_available_banks(self) -> list:
        """Get list of available banks."""
        return list(self._bank_configs.keys())
    
    def get_bank_config(self, bank_name: str) -> Dict[str, Any]:
        """Get configuration for a specific bank."""
        if bank_name not in self._bank_configs:
            raise ValueError(f"Bank '{bank_name}' not found in configurations. Available banks: {list(self._bank_configs.keys())}")
        return self._bank_configs[bank_name]
    
    def get_bank_pattern(self, bank_name: str, account_type: str) -> Dict[str, Any]:
        """Get regex pattern configuration for a bank and account type."""
        config = self.get_bank_config(bank_name)
        account_config = config.get('account_types', {}).get(account_type, {})
        statement_format = account_config.get('statement_format')
        
        if not statement_format:
            raise ValueError(f"No statement format defined for {bank_name}/{account_type}")
        
        patterns = config.get('patterns', {})
        if statement_format not in patterns:
            raise ValueError(f"No pattern defined for {bank_name}/{statement_format}")
        
        return patterns[statement_format]
    
    def detect_bank_from_structure(self) -> Optional[str]:
        """Detect which bank is being used based on folder structure."""
        if not self.bank_statements_dir.exists():
            print(f"Warning: Bank statements directory not found: {self.bank_statements_dir}")
            return None
        
        # Look for bank folders in bank_statements
        bank_folders = [d for d in self.bank_statements_dir.iterdir() 
                       if d.is_dir() and d.name in self._bank_configs]
        
        if len(bank_folders) == 1:
            detected_bank = bank_folders[0].name
            print(f"Detected bank: {detected_bank}")
            return detected_bank
        elif len(bank_folders) > 1:
            # Multiple banks detected - could implement logic to choose based on most recent data
            detected_bank = bank_folders[0].name  # Default to first found
            print(f"Multiple banks detected, using: {detected_bank}")
            return detected_bank
        else:
            print("No bank folders found in bank_statements directory")
            return None
    
    def get_bank_display_name(self, bank_name: str) -> str:
        """Get the display name for a bank."""
        config = self.get_bank_config(bank_name)
        return config.get('display_name', bank_name) 