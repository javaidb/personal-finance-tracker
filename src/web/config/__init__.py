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
        self.STATEMENTS_DIR = os.path.join(self.root_dir, 'bank_statements')
        self.CACHED_DATA_DIR = os.path.join(self.root_dir, 'cached_data')
        self.LOGS_DIR = os.path.join(self.root_dir, 'logs')
        self.UPLOAD_FOLDER = os.path.join(self.root_dir, 'uploads')
        self.ALLOWED_EXTENSIONS = {'pdf'}
        self.MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
        self.DATABANK_PATH = os.path.join(self.root_dir, 'cached_data', 'databank.json')
        self.MERCHANT_DB_PATH = os.path.join(self.root_dir, 'cached_data', 'merchant_db.json')
        self.UNCATEGORIZED_MERCHANTS_PATH = os.path.join(self.root_dir, 'cached_data', 'uncategorized_merchants.json')

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