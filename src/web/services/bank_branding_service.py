from typing import Dict, Any, Optional
from pathlib import Path
import json
import sys
import os

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.bank_config import BankConfig


class BankBrandingService:
    """Service for managing bank-specific branding and styling."""
    
    def __init__(self, bank_config: Optional[BankConfig] = None):
        self.bank_config = bank_config or BankConfig()
        self._branding_cache = {}
    
    def get_bank_branding(self, bank_name: str) -> Dict[str, Any]:
        """Get branding information for a specific bank."""
        if bank_name not in self._branding_cache:
            config = self.bank_config.get_bank_config(bank_name)
            branding = config.get('branding', {})
            self._branding_cache[bank_name] = branding
        
        return self._branding_cache[bank_name]
    
    def get_bank_colors(self, bank_name: str) -> Dict[str, str]:
        """Get color scheme for a specific bank."""
        branding = self.get_bank_branding(bank_name)
        return {
            'primary': branding.get('primary_color', '#007bff'),
            'secondary': branding.get('secondary_color', '#6c757d'),
            'accent': branding.get('accent_color', '#ffc107'),
            'text': branding.get('text_color', '#333333'),
            'background': branding.get('background_color', '#ffffff'),
            'success': branding.get('success_color', '#28a745'),
            'warning': branding.get('warning_color', '#ffc107'),
            'error': branding.get('error_color', '#dc3545')
        }
    
    def get_bank_logo_path(self, bank_name: str) -> str:
        """Get the logo path for a specific bank."""
        branding = self.get_bank_branding(bank_name)
        return branding.get('logo_path', '/static/images/banks/default/logo.svg')
    
    def get_bank_favicon_path(self, bank_name: str) -> str:
        """Get the favicon path for a specific bank."""
        branding = self.get_bank_branding(bank_name)
        return branding.get('favicon_path', '/static/images/banks/default/favicon.svg')
    
    def get_bank_css_variables(self, bank_name: str) -> Dict[str, str]:
        """Get CSS variables for a specific bank."""
        branding = self.get_bank_branding(bank_name)
        return branding.get('css_variables', {})
    
    def get_bank_theme(self, bank_name: str) -> Dict[str, Any]:
        """Get theme information for a specific bank."""
        branding = self.get_bank_branding(bank_name)
        return branding.get('theme', {})
    
    def get_bank_gradients(self, bank_name: str) -> Dict[str, str]:
        """Get gradient definitions for a specific bank."""
        theme = self.get_bank_theme(bank_name)
        return theme.get('gradients', {})
    
    def get_theme_class_name(self, bank_name: str) -> str:
        """Get the CSS class name for a bank's theme."""
        return f"theme-{bank_name.lower()}"
    
    def get_all_bank_branding(self) -> Dict[str, Dict[str, Any]]:
        """Get branding information for all available banks."""
        result = {}
        for bank_name in self.bank_config.get_available_banks():
            result[bank_name] = self.get_bank_branding(bank_name)
        return result
    
    def validate_logo_exists(self, bank_name: str) -> bool:
        """Check if the bank logo file exists."""
        logo_path = self.get_bank_logo_path(bank_name)
        if logo_path.startswith('/static/'):
            # Convert web path to filesystem path
            static_path = Path(__file__).parent.parent / 'static'
            logo_file = static_path / logo_path.replace('/static/', '')
            
            # Check if the exact file exists
            if logo_file.exists():
                return True
            
            # If not found, try alternative extensions
            base_path = logo_file.with_suffix('')
            for ext in ['.svg', '.png', '.jpg', '.jpeg', '.gif']:
                if (base_path.with_suffix(ext)).exists():
                    return True
            
            return False
        return False
    
    def get_bank_logo_path_with_fallback(self, bank_name: str) -> str:
        """Get the logo path with fallback to alternative formats."""
        logo_path = self.get_bank_logo_path(bank_name)
        if logo_path.startswith('/static/'):
            static_path = Path(__file__).parent.parent / 'static'
            logo_file = static_path / logo_path.replace('/static/', '')
            
            # Check if the exact file exists
            if logo_file.exists():
                return logo_path
            
            # Try alternative extensions
            base_path = logo_file.with_suffix('')
            for ext in ['.svg', '.png', '.jpg', '.jpeg', '.gif']:
                alt_file = base_path.with_suffix(ext)
                if alt_file.exists():
                    # Convert back to web path
                    relative_path = alt_file.relative_to(static_path)
                    return f"/static/{relative_path.as_posix()}"
        
        return logo_path
    
    def get_bank_display_info(self, bank_name: str) -> Dict[str, Any]:
        """Get comprehensive display information for a bank."""
        config = self.bank_config.get_bank_config(bank_name)
        branding = self.get_bank_branding(bank_name)
        colors = self.get_bank_colors(bank_name)
        theme = self.get_bank_theme(bank_name)
        
        return {
            'name': bank_name,
            'display_name': self.bank_config.get_bank_display_name(bank_name),
            'currency': config.get('currency', 'USD'),
            'currency_symbol': config.get('currency_symbol', '$'),
            'colors': colors,
            'logo_path': self.get_bank_logo_path_with_fallback(bank_name),
            'favicon_path': self.get_bank_favicon_path(bank_name),
            'theme_class': self.get_theme_class_name(bank_name),
            'theme_name': theme.get('name', f'{bank_name}_theme'),
            'theme_description': theme.get('description', ''),
            'gradients': self.get_bank_gradients(bank_name),
            'css_variables': self.get_bank_css_variables(bank_name),
            'logo_exists': self.validate_logo_exists(bank_name)
        } 