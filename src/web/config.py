import os
from pathlib import Path
from config.paths import get_project_root, get_cached_data_dir, get_uploads_dir, get_bank_statements_dir, paths

class Config:
    """Base configuration."""
    # Get the base directory of the project
    BASE_DIR = get_project_root()
    
    # Upload folder configuration
    UPLOAD_FOLDER = get_uploads_dir()
    
    # Statements directory
    STATEMENTS_DIR = get_bank_statements_dir()
    
    # Cache directory
    CACHE_DIR = get_cached_data_dir()
    
    # Ensure directories exist
    paths.ensure_directory_exists('uploads')
    paths.ensure_directory_exists('bank_statements')
    paths.ensure_directory_exists('cached_data')

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