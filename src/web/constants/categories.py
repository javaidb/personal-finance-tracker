"""
Constants and utilities for category management.
"""

import colorsys
import random
import json
import os
from pathlib import Path
from typing import Dict, Optional

# Path to store category colors
CATEGORY_COLORS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cached_data', 'category_colors.json')

# Minimal default color mapping for categories (only used if category_colors.json doesn't exist)
DEFAULT_CATEGORY_COLORS = {
    'Uncategorized': '#607D8B'  # Blue Grey
}

def load_category_colors() -> Dict[str, str]:
    """Load category colors from file, falling back to defaults if file doesn't exist."""
    try:
        if os.path.exists(CATEGORY_COLORS_FILE):
            with open(CATEGORY_COLORS_FILE, 'r') as f:
                stored_colors = json.load(f)
                # Only keep colors that match the format of DEFAULT_CATEGORY_COLORS
                valid_colors = {
                    k: v for k, v in stored_colors.items()
                    if v.startswith('#') and len(v) == 7
                }
                return valid_colors
    except Exception as e:
        print(f"Error loading category colors: {str(e)}")
    return DEFAULT_CATEGORY_COLORS.copy()

def save_category_colors(colors: Dict[str, str]) -> bool:
    """Save category colors to file."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(CATEGORY_COLORS_FILE), exist_ok=True)
        # Only save colors that match the format of DEFAULT_CATEGORY_COLORS
        valid_colors = {
            k: v for k, v in colors.items()
            if v.startswith('#') and len(v) == 7
        }
        with open(CATEGORY_COLORS_FILE, 'w') as f:
            json.dump(valid_colors, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving category colors: {str(e)}")
        return False

# Initialize CATEGORY_COLORS from stored values
CATEGORY_COLORS = load_category_colors()

def update_category_color(category: str, color: str) -> bool:
    """
    Update the color for a category and persist the change.
    
    Args:
        category: The category name to update
        color: The new color in hex format (#RRGGBB)
        
    Returns:
        bool: True if the update was successful, False otherwise
    """
    try:
        # Validate color format
        if not color.startswith('#') or len(color) != 7:
            return False
            
        # Update the color
        CATEGORY_COLORS[category] = color
        
        # Save to file
        return save_category_colors(CATEGORY_COLORS)
    except Exception as e:
        print(f"Error updating category color: {str(e)}")
        return False

def get_category_color(category: str) -> str:
    """
    Get a consistent color for a category. If the category doesn't exist in the predefined
    colors, generates a stable color based on the category name.
    
    Args:
        category: The category name to get a color for
        
    Returns:
        A color string in hex format (#RRGGBB)
    """
    if category in CATEGORY_COLORS:
        return CATEGORY_COLORS[category]
    
    # Generate a stable color for categories not in the predefined map
    # Use the category name as a seed for consistency
    random.seed(category)
    hue = random.random()
    saturation = 0.7 + random.random() * 0.3  # High saturation
    value = 0.7 + random.random() * 0.3  # Not too dark, not too light
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value)]
    
    # Convert to hex format
    color = f'#{r:02x}{g:02x}{b:02x}'.upper()
    
    # Add the new color to the map and save it
    CATEGORY_COLORS[category] = color
    save_category_colors(CATEGORY_COLORS)
    
    return color 