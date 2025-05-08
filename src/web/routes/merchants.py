from flask import Blueprint, jsonify, request, render_template, redirect, url_for, current_app
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
from ..services.merchant_service import MerchantService
from ..utils.error_handlers import handle_service_error, ServiceError

# Configure logger
logger = logging.getLogger(__name__)

merchants_bp = Blueprint('merchants', __name__, url_prefix='/merchants')
merchant_service = None

def init_merchant_service(base_path: Path) -> None:
    """Initialize the merchant service with the given base path."""
    global merchant_service
    merchant_service = MerchantService(base_path=base_path)

@merchants_bp.before_request
def before_request() -> None:
    """Ensure merchant service is initialized."""
    if merchant_service is None:
        raise ServiceError("Merchant service not initialized")

@merchants_bp.route('/')
def index():
    """Render the merchants management page."""
    try:
        global merchant_service
        if merchant_service is None:
            return render_template('error.html', 
                                message="Merchant service not initialized",
                                show_details=False)

        # Get all required data
        stats = merchant_service.get_merchant_stats()
        categories = merchant_service.get_categories()
        has_uncharacterized = merchant_service.has_uncharacterized_merchants()
        
        # Get category colors
        category_colors = {
            'Groceries': '#4CAF50',
            'Dining': '#FF9800',
            'Transport': '#2196F3',
            'Shopping': '#E91E63',
            'Bills': '#9C27B0',
            'Entertainment': '#00BCD4',
            'Activities': '#FFEB3B',
            'Online': '#795548',
            'Income': '#4CAF50',
            'Rent': '#F44336',
            'Investment': '#009688',
            'Transfer': '#607D8B',
            'Uncategorized': '#9E9E9E'
        }

        return render_template('merchants.html',
                            merchant_count=stats.get('merchant_count', 0),
                            alias_count=stats.get('alias_count', 0),
                            categories=categories,
                            has_review_data=has_uncharacterized,
                            categoryColors=category_colors)
    except Exception as e:
        logger.error(f"Error in merchants index route: {str(e)}", exc_info=True)
        return render_template('error.html',
                            message="Failed to load merchant data",
                            show_details=True,
                            error_details=str(e))

@merchants_bp.route('/api/statistics')
def get_statistics() -> Dict[str, Any]:
    """Get merchant statistics."""
    try:
        global merchant_service
        if merchant_service is None:
            return jsonify({
                "success": False,
                "error": "Merchant service not initialized"
            }), 500

        stats = merchant_service.get_merchant_stats()
        return jsonify({
            "success": True,
            "data": stats
        })
    except Exception as e:
        logger.error(f"Error getting merchant statistics: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Failed to get merchant statistics"
        }), 500

@merchants_bp.route('/api/uncharacterized')
def get_uncharacterized() -> Dict[str, Any]:
    """Get list of uncharacterized merchants."""
    try:
        global merchant_service
        if merchant_service is None:
            return jsonify({
                "success": False,
                "error": "Merchant service not initialized"
            }), 500

        merchants = merchant_service.get_uncharacterized_merchants()
        return jsonify({
            "success": True,
            "data": merchants
        })
    except Exception as e:
        logger.error(f"Error getting uncharacterized merchants: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Failed to get uncharacterized merchants"
        }), 500

@merchants_bp.route('/api/characterize', methods=['POST'])
def characterize_merchant() -> Dict[str, Any]:
    """Characterize a merchant with a category."""
    try:
        global merchant_service
        if merchant_service is None:
            return jsonify({
                "success": False,
                "error": "Merchant service not initialized"
            }), 500

        data = request.get_json()
        if not data or 'merchant' not in data or 'category' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required fields: merchant, category"
            }), 400

        merchant = data['merchant']
        category = data['category']
        pattern = data.get('pattern', merchant)  # Optional pattern, defaults to merchant name

        success = merchant_service.characterize_merchant(merchant, category, pattern)
        if success:
            return jsonify({
                "success": True,
                "message": f"Successfully characterized {merchant} as {category}"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to characterize merchant"
            }), 500
    except Exception as e:
        logger.error(f"Error characterizing merchant: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Failed to characterize merchant"
        }), 500

@merchants_bp.route('/review')
def merchants_review() -> str:
    """Render the page for reviewing uncharacterized merchants."""
    try:
        if not merchant_service.has_uncharacterized_merchants():
            return redirect(url_for('merchants.merchants_dashboard'))
        
        merchant_count = merchant_service.get_uncharacterized_count()
        categories = merchant_service.get_categories()
        
        if merchant_count == 0:
            return redirect(url_for('merchants.merchants_dashboard'))
        
        return render_template('merchants_review.html',
                             merchant_count=merchant_count,
                             categories=categories)
    except Exception as e:
        logger.error(f"Error in merchants_review: {str(e)}", exc_info=True)
        return handle_service_error(e)

@merchants_bp.route('/api/merchants')
def get_merchants() -> Tuple[Dict[str, Any], int]:
    """API endpoint to get merchants."""
    try:
        search_term = request.args.get('search', '')
        merchants = merchant_service.search_merchants(search_term) if search_term else merchant_service.get_all_merchants()
        
        # Format merchants list to match frontend expectations
        formatted_merchants = [{"name": m["name"], "category": m["category"]} for m in merchants]
        
        return jsonify({
            "success": True,
            "merchants": formatted_merchants,
            "count": len(formatted_merchants)
        }), 200
    except Exception as e:
        logger.error(f"Error in get_merchants: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@merchants_bp.route('/api/merchants', methods=['POST'])
def update_merchant() -> Tuple[Dict[str, Any], int]:
    """API endpoint to update a merchant's category."""
    try:
        data = request.get_json()
        if not data or 'merchant' not in data or 'category' not in data:
            raise ServiceError("Missing required fields: merchant and category")
        
        merchant_name = data['merchant']
        category = data['category']
        
        success = merchant_service.add_merchant(merchant_name, category)
        if not success:
            raise ServiceError(f"Failed to update merchant {merchant_name}")
        
        return jsonify({
            "success": True,
            "merchant": merchant_name,
            "category": category
        }), 200
    except Exception as e:
        logger.error(f"Error in update_merchant: {str(e)}", exc_info=True)
        return handle_service_error(e)

@merchants_bp.route('/api/aliases', methods=['POST'])
def add_alias() -> Tuple[Dict[str, Any], int]:
    """API endpoint to add a merchant alias."""
    try:
        data = request.get_json()
        if not data or 'alias' not in data or 'merchant' not in data:
            raise ServiceError("Missing required fields: alias and merchant")
        
        alias = data['alias']
        merchant = data['merchant']
        
        success = merchant_service.add_alias(alias, merchant)
        if not success:
            raise ServiceError(f"Could not add alias. Ensure '{merchant}' exists in the database.")
        
        return jsonify({
            "success": True,
            "alias": alias,
            "merchant": merchant
        }), 200
    except Exception as e:
        logger.error(f"Error in add_alias: {str(e)}", exc_info=True)
        return handle_service_error(e)

@merchants_bp.route('/api/merchants/uncharacterized')
def get_uncharacterized_merchants() -> Tuple[Dict[str, Any], int]:
    """API endpoint to get uncharacterized merchants for review."""
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        
        result = merchant_service.get_uncharacterized_merchants(page=page, limit=limit)
        result['success'] = True
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error in get_uncharacterized_merchants: {str(e)}", exc_info=True)
        return handle_service_error(e)

@merchants_bp.route('/api/merchants/uncharacterized', methods=['POST'])
def categorize_uncharacterized_merchant() -> Tuple[Dict[str, Any], int]:
    """API endpoint to categorize an uncharacterized merchant."""
    try:
        data = request.get_json()
        if not data or 'merchant' not in data or 'category' not in data:
            raise ServiceError("Missing required fields: merchant and category")
        
        merchant_name = data['merchant']
        category = data['category']
        
        success = merchant_service.categorize_uncharacterized_merchant(merchant_name, category)
        if not success:
            raise ServiceError(f"Failed to categorize merchant {merchant_name}")
        
        return jsonify({
            "success": True,
            "merchant": merchant_name,
            "category": category
        }), 200
    except Exception as e:
        logger.error(f"Error in categorize_uncharacterized_merchant: {str(e)}", exc_info=True)
        return handle_service_error(e)

@merchants_bp.route('/api/merchants/uncharacterized/count')
def get_uncharacterized_count() -> Tuple[Dict[str, Any], int]:
    """API endpoint to get the count of uncharacterized merchants."""
    try:
        count = merchant_service.get_uncharacterized_count()
        return jsonify({
            "success": True,
            "count": count
        }), 200
    except Exception as e:
        logger.error(f"Error in get_uncharacterized_count: {str(e)}", exc_info=True)
        return handle_service_error(e)

@merchants_bp.route('/api/merchants/<merchant_name>', methods=['DELETE'])
def delete_merchant(merchant_name: str) -> Tuple[Dict[str, Any], int]:
    """API endpoint to delete a merchant."""
    try:
        success = merchant_service.merchant_categorizer.delete_merchant(merchant_name)
        if not success:
            raise ServiceError(f"Failed to delete merchant {merchant_name}")
        
        return jsonify({
            "success": True,
            "message": f"Successfully deleted merchant {merchant_name}"
        }), 200
    except Exception as e:
        logger.error(f"Error in delete_merchant: {str(e)}", exc_info=True)
        return handle_service_error(e) 