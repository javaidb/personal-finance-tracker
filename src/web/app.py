import os
import sys
from pathlib import Path

# Add parent directory to path so we can import our existing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import Flask and CORS
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from flask_cors import CORS

# Import our modules
from src.modules.statement_interpreter import StatementInterpreter
from src.modules.helper_fns import GeneralHelperFns, CategoryUpdater
from src.modules.merchant_categorizer import MerchantCategorizer

# Import configuration
from src.web.config import config

# Create Flask app using the application factory
app = Flask(__name__)

# Load configuration
app_config = config['default']
app.config.from_object(app_config)

# Enable CORS
CORS(app)

# Import and register blueprints
from src.web.routes.category_routes import category_bp
from src.web.routes.main import main_bp
from src.web.routes.api import api_bp
from src.web.routes.merchants import merchants_bp
from src.web.routes.bank_branding import bank_branding_bp, bank_branding_web_bp
from src.web.constants.categories import CATEGORY_COLORS, get_category_color

# Register blueprints
app.register_blueprint(category_bp)
app.register_blueprint(main_bp)
app.register_blueprint(api_bp)
app.register_blueprint(merchants_bp)
app.register_blueprint(bank_branding_bp)
app.register_blueprint(bank_branding_web_bp)

if __name__ == '__main__':
    app.run(debug=True, port=8000) 