from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
import sys
import pandas as pd
import json
from pathlib import Path
import glob
import numpy as np
import colorsys
import random
from . import create_app

# Add parent directory to path so we can import our existing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.modules.pdf_interpreter import PDFReader
from src.modules.helper_fns import GeneralHelperFns, CategoryUpdater
from src.modules.merchant_categorizer import MerchantCategorizer

# Global color mapping for categories to ensure consistency across visualizations
CATEGORY_COLORS = {
    'Groceries': '#4CAF50',        # Green
    'Dining': '#FF9800',           # Orange
    'Transport': '#2196F3',        # Blue
    'Shopping': '#9C27B0',         # Purple
    'Bills': '#F44336',            # Red
    'Entertainment': '#FFD700',    # Gold
    'Travel': '#8B4513',           # Brown
    'Healthcare': '#00BCD4',       # Cyan
    'Education': '#3F51B5',        # Indigo
    'Housing': '#E91E63',          # Pink
    'Income': '#2E7D32',           # Dark Green
    'Salary': '#1B5E20',           # Darker Green
    'Investments': '#388E3C',      # Medium Green
    'Transfers': '#424242',        # Dark Grey
    'Utilities': '#F57C00',        # Dark Orange
    'Insurance': '#D32F2F',        # Dark Red
    'Subscription': '#7B1FA2',     # Dark Purple
    'Uncategorized': '#607D8B'     # Blue Grey
}

# Function to get consistent color for a category
def get_category_color(category):
    if category in CATEGORY_COLORS:
        return CATEGORY_COLORS[category]
    
    # Generate a stable color for categories not in the predefined map
    # Use the category name as a seed for consistency
    random.seed(category)
    hue = random.random()
    saturation = 0.7 + random.random() * 0.3  # High saturation
    value = 0.7 + random.random() * 0.3  # Not too dark, not too light
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value)]
    
    # Add the new color to the map for future use
    CATEGORY_COLORS[category] = f'rgb({r},{g},{b})'
    return CATEGORY_COLORS[category]

# Create the Flask app using the factory pattern
app = create_app()

if __name__ == '__main__':
    app.run(debug=True) 