from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from flask_cors import CORS
import os
import sys
import pandas as pd
import json
from pathlib import Path
import glob
import numpy as np
from .routes.category_routes import category_bp
from .routes.main import main_bp
from .routes.api import api_bp
from .routes.merchants import merchants_bp
from .routes.bank_branding import bank_branding_bp
from .constants.categories import CATEGORY_COLORS, get_category_color

# Add parent directory to path so we can import our existing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.modules.pdf_interpreter import PDFReader
from src.modules.helper_fns import GeneralHelperFns, CategoryUpdater
from src.modules.merchant_categorizer import MerchantCategorizer

# Create Flask app
app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(category_bp)
app.register_blueprint(main_bp)
app.register_blueprint(api_bp)
app.register_blueprint(merchants_bp)
app.register_blueprint(bank_branding_bp)
app.register_blueprint(bank_branding_web_bp)

if __name__ == '__main__':
    app.run(debug=True, port=8000) 