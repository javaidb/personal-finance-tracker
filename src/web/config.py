import os
from pathlib import Path

class Config:
    """Base configuration."""
    # Get the base directory of the project
    BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    # Upload folder configuration
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'src', 'web', 'uploads')
    
    # Statements directory
    STATEMENTS_DIR = os.path.join(BASE_DIR, 'bank_statements')
    
    # Cache directory
    CACHE_DIR = os.path.join(BASE_DIR, 'cached_data')
    
    # Ensure directories exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(STATEMENTS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = False
    TESTING = True

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 