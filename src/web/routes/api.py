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
    
    # Get filter parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    description_search = request.args.get('description')
    account_type = request.args.get('account_type')
    category = request.args.get('category')
    show_recent = request.args.get('show_recent', 'false').lower() == 'true'
    
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
    
    # If no date filters provided and show_recent is true, show last 3 months
    if not start_date and not end_date and show_recent and not df_copy.empty:
        latest_date = df_copy['DateTime'].max()
        three_months_ago = latest_date - pd.DateOffset(months=3)
        df_copy = df_copy[df_copy['DateTime'] >= three_months_ago]
    
    # Apply description search if provided
    if description_search:
        df_copy = df_copy[df_copy['Details'].str.contains(description_search, case=False, na=False)]
    
    # Apply account type filter if provided
    if account_type:
        df_copy = df_copy[df_copy['Account Type'] == account_type]
    
    # Apply category filter if provided
    if category:
        df_copy = df_copy[df_copy['Classification'] == category]
    
    # Replace NaN values with None for JSON serialization
    df_copy = df_copy.where(pd.notna(df_copy), None)
    
    transactions = df_copy.to_dict(orient='records')
    return jsonify(transactions)

@api_bp.route('/filter-options')
def get_filter_options():
    """Get available options for filters."""
    service = init_transaction_service()
    if service is None or service.processed_df is None:
        return jsonify({"error": "No data has been processed yet"}), 404
    
    df = service.processed_df
    
    # Get unique values for account types and categories
    account_types = sorted(df['Account Type'].unique().tolist())
    categories = sorted(df['Classification'].unique().tolist())
    
    # Remove None/NaN values
    account_types = [str(at) for at in account_types if at is not None and str(at) != 'nan']
    categories = [str(cat) for cat in categories if cat is not None and str(cat) != 'nan']
    
    return jsonify({
        'account_types': account_types,
        'categories': categories
    })

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
    try:
        base_path = Path(current_app.root_path).parent.parent
        merchant_service = MerchantService(base_path=base_path)
        merchants = merchant_service.get_all_merchants()
        
        # Log the response for debugging
        print(f"Merchants API response: {merchants}")
        
        return jsonify({
            "success": True,
            "merchants": merchants,
            "count": len(merchants)
        })
    except Exception as e:
        print(f"Error in get_merchants: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

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
    try:
        data = request.get_json()
        if not data or 'name' not in data or 'category' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required fields"
            }), 400
        
        base_path = Path(current_app.root_path).parent.parent
        
        # Initialize services
        merchant_service = MerchantService(base_path=base_path)
        transaction_service = init_transaction_service()
        
        # Add merchant
        success = merchant_service.add_merchant(data['name'], data['category'])
        
        if success:
            # Force transaction service to reload data
            if transaction_service:
                transaction_service.recategorize_transactions()
            
            return jsonify({
                "success": True,
                "message": "Merchant added successfully",
                "merchant": {
                    "name": data['name'],
                    "category": data['category']
                }
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to add merchant"
            }), 500
    except Exception as e:
        print(f"Error in add_merchant: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@api_bp.route('/merchants/<merchant_name>', methods=['DELETE'])
def delete_merchant(merchant_name):
    """Delete a merchant."""
    try:
        base_path = Path(current_app.root_path).parent.parent
        merchant_service = MerchantService(base_path=base_path)
        
        # Get the merchant's category before deleting
        merchants = merchant_service.get_all_merchants()
        merchant = next((m for m in merchants if m['name'].lower() == merchant_name.lower()), None)
        
        if not merchant:
            return jsonify({"error": "Merchant not found"}), 404
            
        # Remove the merchant's pattern from the databank
        if merchant['category'] in merchant_service.databank.get('categories', {}):
            patterns = merchant_service.databank['categories'][merchant['category']]['patterns']
            merchant_service.databank['categories'][merchant['category']]['patterns'] = [
                p for p in patterns if ' '.join(p['terms']).lower() != merchant_name.lower()
            ]
            merchant_service.save_databank()
        
        # Delete from merchant database
        success = merchant_service.merchant_categorizer.delete_merchant(merchant_name)
        
        if success:
            return jsonify({"message": "Merchant deleted successfully"})
        else:
            return jsonify({"error": "Failed to delete merchant"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

@api_bp.route('/merchants/categories')
def get_merchant_categories():
    """Get all available merchant categories."""
    base_path = Path(current_app.root_path).parent.parent
    merchant_service = MerchantService(base_path=base_path)
    categories = merchant_service.get_categories()
    return jsonify(categories)

@api_bp.route('/merchants/categories', methods=['POST'])
def add_merchant_category():
    """Add a new merchant category."""
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify({"error": "Missing category name"}), 400
    
    base_path = Path(current_app.root_path).parent.parent
    merchant_service = MerchantService(base_path=base_path)
    
    success = merchant_service.add_category(data['name'])
    
    if success:
        return jsonify({"message": "Category added successfully"})
    else:
        return jsonify({"error": "Category already exists or failed to add"}), 400

@api_bp.route('/merchants/categories/<old_name>', methods=['PUT'])
def rename_merchant_category(old_name):
    """Rename a merchant category."""
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify({"error": "Missing new category name"}), 400
    
    base_path = Path(current_app.root_path).parent.parent
    merchant_service = MerchantService(base_path=base_path)
    
    success = merchant_service.rename_category(old_name, data['name'])
    
    if success:
        return jsonify({"message": "Category renamed successfully"})
    else:
        return jsonify({"error": "Category not found or new name already exists"}), 400

@api_bp.route('/merchants/categories/<name>', methods=['DELETE'])
def delete_merchant_category(name):
    """Delete a merchant category."""
    base_path = Path(current_app.root_path).parent.parent
    merchant_service = MerchantService(base_path=base_path)
    
    success = merchant_service.delete_category(name)
    
    if success:
        return jsonify({"message": "Category deleted successfully"})
    else:
        return jsonify({"error": "Category not found or failed to delete"}), 400 