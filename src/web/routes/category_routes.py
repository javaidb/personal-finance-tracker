from flask import Blueprint, jsonify, request
import os
import json
from datetime import datetime
from pathlib import Path

category_bp = Blueprint('category_bp', __name__, url_prefix='/api')

class CategoryService:
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        self.base_path = base_path
        self.databank_path = os.path.join(base_path, 'cached_data', 'databank.json')
        self.load_databank()

    def load_databank(self):
        """Load or create databank.json"""
        try:
            if os.path.exists(self.databank_path):
                with open(self.databank_path, 'r') as f:
                    self.databank = json.load(f)
            else:
                # Create initial databank structure
                self.databank = {
                    "categories": {
                        "Groceries": {"totalMatches": 0, "patterns": []},
                        "Dining": {"totalMatches": 0, "patterns": []},
                        "Transport": {"totalMatches": 0, "patterns": []},
                        "Shopping": {"totalMatches": 0, "patterns": []},
                        "Bills": {"totalMatches": 0, "patterns": []},
                        "Entertainment": {"totalMatches": 0, "patterns": []},
                        "Income": {"totalMatches": 0, "patterns": []},
                        "Transfer": {"totalMatches": 0, "patterns": []},
                        "uncharacterized": {"totalMatches": 0, "patterns": []}
                    }
                }
                self.save_databank()
        except Exception as e:
            print(f"Error loading databank: {str(e)}")
            self.databank = {"categories": {}}

    def save_databank(self):
        """Save databank to file"""
        try:
            with open(self.databank_path, 'w') as f:
                json.dump(self.databank, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving databank: {str(e)}")
            return False

    def get_categories(self):
        """Get all categories with their details"""
        return self.databank.get('categories', {})

    def get_category(self, category_name):
        """Get details for a specific category"""
        return self.databank.get('categories', {}).get(category_name)

    def add_category(self, name, patterns):
        """Add a new category"""
        if name in self.databank.get('categories', {}):
            return False, "Category already exists"
        
        self.databank['categories'][name] = {
            "totalMatches": 0,
            "patterns": patterns
        }
        
        if self.save_databank():
            return True, "Category added successfully"
        return False, "Error saving category"

    def update_category(self, old_name, new_name, patterns):
        """Update an existing category"""
        categories = self.databank.get('categories', {})
        if old_name not in categories:
            return False, "Category not found"
        
        if old_name != new_name and new_name in categories:
            return False, "New category name already exists"
        
        # If name is changing, remove old and add new
        if old_name != new_name:
            categories[new_name] = categories.pop(old_name)
        
        categories[new_name]['patterns'] = patterns
        
        if self.save_databank():
            return True, "Category updated successfully"
        return False, "Error updating category"

    def delete_category(self, name):
        """Delete a category"""
        if name not in self.databank.get('categories', {}):
            return False, "Category not found"
        
        del self.databank['categories'][name]
        
        if self.save_databank():
            return True, "Category deleted successfully"
        return False, "Error deleting category"

# Initialize the service
category_service = CategoryService()

@category_bp.route('/categories/details', methods=['GET'])
def get_categories():
    """Get all categories with their details"""
    try:
        categories = category_service.get_categories()
        return jsonify({"categories": categories})
    except Exception as e:
        print(f"Error getting categories: {str(e)}")
        return jsonify({"error": str(e)}), 500

@category_bp.route('/categories/<category_name>', methods=['GET'])
def get_category(category_name):
    """Get details for a specific category"""
    category = category_service.get_category(category_name)
    if category is None:
        return jsonify({"error": "Category not found"}), 404
    return jsonify(category)

@category_bp.route('/categories', methods=['POST'])
def add_category():
    """Add a new category"""
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    success, message = category_service.add_category(data['name'], data.get('patterns', []))
    if success:
        return jsonify({"message": message}), 201
    return jsonify({"error": message}), 400

@category_bp.route('/categories/<category_name>', methods=['PUT'])
def update_category(category_name):
    """Update an existing category"""
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    success, message = category_service.update_category(
        category_name, 
        data['name'], 
        data.get('patterns', [])
    )
    if success:
        return jsonify({"message": message})
    return jsonify({"error": message}), 400

@category_bp.route('/categories/<category_name>', methods=['DELETE'])
def delete_category(category_name):
    """Delete a category"""
    success, message = category_service.delete_category(category_name)
    if success:
        return jsonify({"message": message})
    return jsonify({"error": message}), 404 