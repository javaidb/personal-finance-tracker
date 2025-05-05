"""Routes package."""

from .main import main_bp
from .api import api_bp
from .merchants import merchants_bp

__all__ = ['main_bp', 'api_bp', 'merchants_bp'] 