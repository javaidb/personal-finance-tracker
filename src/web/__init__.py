"""Flask application factory."""
import os
from pathlib import Path
from flask import Flask
from .config import config
from .config.logging_config import setup_logging
from .routes import main_bp, api_bp, merchants_bp, category_bp
from .routes.merchants import init_merchant_service

def create_app(config_name='default'):
    """Create Flask application."""
    # Set up base path
    base_path = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    
    # Set up logging
    setup_logging(base_path, config_name)
    
    # Create Flask app
    app = Flask(__name__)
    
    # Load config
    app_config = config[config_name]
    app.config.from_object(app_config)
    
    # Initialize services
    init_merchant_service(base_path)
    
    # Register blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(merchants_bp, url_prefix='/merchants')
    app.register_blueprint(category_bp, url_prefix='/api')
    
    return app 