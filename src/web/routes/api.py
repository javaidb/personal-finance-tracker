from flask import Blueprint, jsonify, request, current_app
from pathlib import Path
import os
import json
from ..services.transaction_service import TransactionService
from ..services.merchant_service import MerchantService
import pandas as pd

api_bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize global service
transaction_service = None

def init_transaction_service():
    """Initialize the transaction service if not already initialized."""
    global transaction_service
    try:
        if transaction_service is None:
            base_path = Path(current_app.root_path).parent.parent
            transaction_service = TransactionService(base_path=base_path)
            # Try to process statements immediately
            transaction_service.process_statements()
        return transaction_service
    except Exception as e:
        print(f"Error initializing transaction service: {str(e)}")
        return None

@api_bp.route('/transactions')
def get_transactions():
    """Get transactions with optional date filtering."""
    service = init_transaction_service()
    if service is None or service.processed_df is None:
        return jsonify({"error": "No data has been processed yet"}), 404
    
    # Convert DataFrame to dict for JSON response, handling NaN values
    df_copy = service.processed_df.copy()
    
    # Get date range filter parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Sort by date in descending order (most recent first)
    df_copy['DateTime'] = pd.to_datetime(df_copy['DateTime'])
    df_copy = df_copy.sort_values(by='DateTime', ascending=False)
    
    # Apply date range filtering if parameters are provided
    if start_date:
        start_date = pd.to_datetime(start_date)
        df_copy = df_copy[df_copy['DateTime'] >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        df_copy = df_copy[df_copy['DateTime'] <= end_date]
    
    # If no date filters provided, default to last 3 months
    if not start_date and not end_date and not df_copy.empty:
        latest_date = df_copy['DateTime'].max()
        three_months_ago = latest_date - pd.DateOffset(months=3)
        df_copy = df_copy[df_copy['DateTime'] >= three_months_ago]
    
    # Replace NaN values with None for JSON serialization
    df_copy = df_copy.where(pd.notna(df_copy), None)
    
    transactions = df_copy.to_dict(orient='records')
    return jsonify(transactions)

@api_bp.route('/categories')
def get_categories():
    """Get category data including counts, spending, and colors."""
    service = init_transaction_service()
    if service is None or service.processed_df is None:
        return jsonify({"error": "No data has been processed yet"}), 404
    
    category_data = service.get_category_data()
    return jsonify(category_data)

@api_bp.route('/balance-chart')
def get_balance_chart():
    """Get balance chart data."""
    service = init_transaction_service()
    if service is None or service.processed_df is None:
        return jsonify({"error": "No data has been processed yet"}), 404
    
    chart_data = service.get_balance_chart_data()
    return jsonify(chart_data)

@api_bp.route('/pelt-analysis')
def get_pelt_analysis():
    """Get PELT analysis data."""
    service = init_transaction_service()
    if service is None or service.processed_df is None:
        return jsonify({"error": "No data has been processed yet"}), 404
    
    analysis_data = service.get_pelt_analysis_data()
    return jsonify(analysis_data)

@api_bp.route('/monthly-trends')
def get_monthly_trends():
    """Get monthly trends data."""
    service = init_transaction_service()
    if service is None or service.processed_df is None:
        return jsonify({"error": "No data has been processed yet"}), 404
    
    trends_data = service.get_monthly_trends_data()
    return jsonify(trends_data)

@api_bp.route('/merchants')
def get_merchants():
    """Get all merchants."""
    base_path = Path(current_app.root_path).parent.parent
    merchant_service = MerchantService(base_path=base_path)
    merchants = merchant_service.get_all_merchants()
    return jsonify(merchants)

@api_bp.route('/merchants/search')
def search_merchants():
    """Search merchants by name."""
    search_term = request.args.get('q', '')
    base_path = Path(current_app.root_path).parent.parent
    merchant_service = MerchantService(base_path=base_path)
    merchants = merchant_service.search_merchants(search_term)
    return jsonify(merchants)

@api_bp.route('/merchants', methods=['POST'])
def add_merchant():
    """Add a new merchant."""
    data = request.get_json()
    if not data or 'name' not in data or 'category' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    base_path = Path(current_app.root_path).parent.parent
    merchant_service = MerchantService(base_path=base_path)
    success = merchant_service.add_merchant(data['name'], data['category'])
    
    if success:
        return jsonify({"message": "Merchant added successfully"})
    else:
        return jsonify({"error": "Failed to add merchant"}), 500

@api_bp.route('/merchants/alias', methods=['POST'])
def add_alias():
    """Add a merchant alias."""
    data = request.get_json()
    if not data or 'alias' not in data or 'merchant' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    base_path = Path(current_app.root_path).parent.parent
    merchant_service = MerchantService(base_path=base_path)
    success = merchant_service.add_alias(data['alias'], data['merchant'])
    
    if success:
        return jsonify({"message": "Alias added successfully"})
    else:
        return jsonify({"error": "Failed to add alias"}), 500

@api_bp.route('/merchants/stats')
def get_merchant_stats():
    """Get merchant statistics."""
    base_path = Path(current_app.root_path).parent.parent
    merchant_service = MerchantService(base_path=base_path)
    stats = merchant_service.get_merchant_stats()
    return jsonify(stats)

@api_bp.route('/merchants/uncharacterized')
def get_uncharacterized_merchants():
    """Get uncharacterized merchants."""
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 10))
    
    base_path = Path(current_app.root_path).parent.parent
    merchant_service = MerchantService(base_path=base_path)
    merchants = merchant_service.get_uncharacterized_merchants(page, limit)
    return jsonify(merchants)

@api_bp.route('/merchants/uncharacterized', methods=['POST'])
def categorize_uncharacterized_merchant():
    """Categorize an uncharacterized merchant."""
    data = request.get_json()
    if not data or 'merchant' not in data or 'category' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    base_path = Path(current_app.root_path).parent.parent
    merchant_service = MerchantService(base_path=base_path)
    success = merchant_service.categorize_uncharacterized_merchant(data['merchant'], data['category'])
    
    if success:
        return jsonify({"message": "Merchant categorized successfully"})
    else:
        return jsonify({"error": "Failed to categorize merchant"}), 500

@api_bp.route('/merchants/has-uncharacterized')
def has_uncharacterized_merchants():
    """Check if there are any uncharacterized merchants."""
    base_path = Path(current_app.root_path).parent.parent
    merchant_service = MerchantService(base_path=base_path)
    has_uncharacterized = merchant_service.has_uncharacterized_merchants()
    return jsonify({"has_uncharacterized": has_uncharacterized})

@api_bp.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the PDF cache."""
    global transaction_service
    service = init_transaction_service()
    if service is None:
        return jsonify({"error": "No statement reader initialized"}), 400
    
    try:
        success = service.clear_cache()
        if success:
            # Reset the transaction service to force reinitialization
            transaction_service = None
            # Also reset the service instance to ensure it's fully cleared
            service.processed_df = None
            # Force a reload of the data
            service = init_transaction_service()
            if service is not None:
                service.process_statements()
            return jsonify({"success": True, "message": "PDF cache cleared successfully and data reloaded"})
        else:
            return jsonify({"error": "Failed to clear cache"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to clear cache: {str(e)}"}), 500 