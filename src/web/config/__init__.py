"""Configuration package for the web application."""
import os
from pathlib import Path

class Config:
    """Base configuration."""
    def __init__(self):
        self.root_dir = self.get_project_root()
        self.init_app()

    @staticmethod
    def get_project_root() -> Path:
        """Get the project root directory."""
        return Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    
    def init_app(self):
        """Initialize application configuration."""
        # Common configuration settings
        from src.config.paths import get_bank_statements_dir, get_cached_data_dir, get_logs_dir, get_uploads_dir
        from src.config.paths import DATABANK_PATH, MERCHANT_DB_PATH, UNCATEGORIZED_MERCHANTS_PATH
        
        self.STATEMENTS_DIR = get_bank_statements_dir()
        self.CACHED_DATA_DIR = get_cached_data_dir()
        self.LOGS_DIR = get_logs_dir()
        self.UPLOAD_FOLDER = get_uploads_dir()
        self.ALLOWED_EXTENSIONS = {'pdf', 'csv'}
        self.MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
        self.DATABANK_PATH = DATABANK_PATH
        self.MERCHANT_DB_PATH = MERCHANT_DB_PATH
        self.UNCATEGORIZED_MERCHANTS_PATH = UNCATEGORIZED_MERCHANTS_PATH

class DevelopmentConfig(Config):
    """Development configuration."""
    def init_app(self):
        super().init_app()
        self.DEBUG = True
        self.TESTING = False
        self.SECRET_KEY = 'dev'
        self.TEMPLATES_AUTO_RELOAD = True

class TestingConfig(Config):
    """Testing configuration."""
    def init_app(self):
        super().init_app()
        self.DEBUG = True
        self.TESTING = True
        self.SECRET_KEY = 'test'
        self.TEMPLATES_AUTO_RELOAD = True
        # Override paths for testing environment
        self.STATEMENTS_DIR = os.path.join(self.root_dir, 'tests', 'test_data', 'bank_statements')
        self.CACHED_DATA_DIR = os.path.join(self.root_dir, 'tests', 'test_data', 'cached_data')

class ProductionConfig(Config):
    """Production configuration."""
    def init_app(self):
        super().init_app()
        self.DEBUG = False
        self.TESTING = False
        self.SECRET_KEY = os.environ.get('SECRET_KEY', 'change-this-in-production')
        self.TEMPLATES_AUTO_RELOAD = False

config = {
    'development': DevelopmentConfig(),
    'testing': TestingConfig(),
    'production': ProductionConfig(),
    'default': DevelopmentConfig()
} 