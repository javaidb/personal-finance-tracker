"""
User Settings Service for managing user preferences and settings.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class UserSettingsService:
    """Service for managing user settings and preferences."""
    
    def __init__(self, settings_file_path: Optional[Path] = None):
        """Initialize the user settings service.
        
        Args:
            settings_file_path: Path to the user settings file. If None, uses default path.
        """
        if settings_file_path is None:
            from src.config.paths import get_cached_data_file
            settings_file_path = get_cached_data_file('user_settings')
        
        self.settings_file_path = Path(settings_file_path)
        self._settings = None
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from file."""
        if self._settings is not None:
            return self._settings
        
        try:
            if self.settings_file_path.exists():
                with open(self.settings_file_path, 'r', encoding='utf-8') as f:
                    self._settings = json.load(f)
            else:
                # Create default settings
                self._settings = {
                    "user_name": "",
                    "preferences": {
                        "theme": "default",
                        "dashboard_layout": "default"
                    },
                    "last_updated": None
                }
                self._save_settings()
        except Exception as e:
            logger.error(f"Error loading user settings: {str(e)}")
            # Return default settings on error
            self._settings = {
                "user_name": "",
                "preferences": {
                    "theme": "default",
                    "dashboard_layout": "default"
                },
                "last_updated": None
            }
        
        return self._settings
    
    def _save_settings(self) -> bool:
        """Save settings to file."""
        try:
            # Ensure directory exists
            self.settings_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Update timestamp
            self._settings["last_updated"] = datetime.now().isoformat()
            
            with open(self.settings_file_path, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            logger.error(f"Error saving user settings: {str(e)}")
            return False
    
    def get_user_name(self) -> str:
        """Get the user's name."""
        settings = self._load_settings()
        return settings.get("user_name", "")
    
    def set_user_name(self, user_name: str) -> bool:
        """Set the user's name.
        
        Args:
            user_name: The user's name to save.
            
        Returns:
            True if successful, False otherwise.
        """
        settings = self._load_settings()
        settings["user_name"] = user_name.strip()
        return self._save_settings()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference.
        
        Args:
            key: The preference key.
            default: Default value if preference doesn't exist.
            
        Returns:
            The preference value or default.
        """
        settings = self._load_settings()
        return settings.get("preferences", {}).get(key, default)
    
    def set_preference(self, key: str, value: Any) -> bool:
        """Set a user preference.
        
        Args:
            key: The preference key.
            value: The preference value.
            
        Returns:
            True if successful, False otherwise.
        """
        settings = self._load_settings()
        if "preferences" not in settings:
            settings["preferences"] = {}
        settings["preferences"][key] = value
        return self._save_settings()
    
    def get_all_preferences(self) -> Dict[str, Any]:
        """Get all user preferences.
        
        Returns:
            Dictionary of all preferences.
        """
        settings = self._load_settings()
        return settings.get("preferences", {})
    
    def reset_settings(self) -> bool:
        """Reset all settings to defaults."""
        self._settings = {
            "user_name": "",
            "preferences": {
                "theme": "default",
                "dashboard_layout": "default"
            },
            "last_updated": None
        }
        return self._save_settings() 