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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the path to bank statements
STATEMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../bank_statements'))

# Initialize globals
statement_reader = None
processed_df = None

@app.route('/')
def index():
    # Get all account types (directories)
    account_types = []
    for item in os.listdir(STATEMENTS_DIR):
        if os.path.isdir(os.path.join(STATEMENTS_DIR, item)):
            account_types.append(item)
    
    # Get statement counts by type
    statement_counts = {}
    for account_type in account_types:
        account_type_dir = os.path.join(STATEMENTS_DIR, account_type)
        account_names = []
        # Get account names
        for item in os.listdir(account_type_dir):
            if os.path.isdir(os.path.join(account_type_dir, item)):
                account_names.append(item)
                
        statement_counts[account_type] = {}
        for account_name in account_names:
            account_dir = os.path.join(account_type_dir, account_name)
            # Count PDFs in this account directory
            pdf_count = len(glob.glob(os.path.join(account_dir, "*.pdf")))
            statement_counts[account_type][account_name] = pdf_count
    
    # Get info about cached PDFs
    cache_info = {"cached_pdfs_count": 0, "cache_size_kb": 0}
    try:
        # Initialize PDFReader to get cache info
        temp_reader = PDFReader(base_path=Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))))
        cache_info = temp_reader.get_cache_info()
    except Exception as e:
        print(f"Error getting cache info: {str(e)}")
    
    return render_template('index.html', 
                           account_types=account_types, 
                           statement_counts=statement_counts,
                           cache_info=cache_info)

@app.route('/process_statements', methods=['POST'])
def process_statements():
    global statement_reader, processed_df
    
    selected_types = request.form.getlist('account_types')
    
    # If none selected, use all
    if not selected_types:
        selected_types = [d for d in os.listdir(STATEMENTS_DIR) 
                          if os.path.isdir(os.path.join(STATEMENTS_DIR, d))]

    # Process statements for selected account types
    try:
        # Initialize PDFReader with base_path pointing to the project root
        statement_reader = PDFReader(base_path=Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))))
        
        # Process statements and get the resulting DataFrame
        processed_df = statement_reader.process_raw_df()
        
        # Ensure processed_df is not None before proceeding
        if processed_df is None or processed_df.empty:
            error_message = "No transactions found in the selected statements."
            return render_template('error.html', message=error_message, show_details=False)
        
        # Basic stats to confirm processing
        stats = {
            "total_transactions": len(processed_df),
            "date_range": ["N/A", "N/A"]  # Default values
        }
        
        # Safely extract date range if the DataFrame is not empty and contains DateTime column
        if not processed_df.empty and 'DateTime' in processed_df.columns:
            # Ensure DateTime is properly converted to datetime objects
            processed_df['DateTime'] = pd.to_datetime(processed_df['DateTime'], errors='coerce')
            
            # Remove NaT values for min/max calculations
            date_df = processed_df[pd.notna(processed_df['DateTime'])]
            
            if not date_df.empty:
                stats["date_range"] = [
                    date_df['DateTime'].min().strftime('%Y-%m-%d'),
                    date_df['DateTime'].max().strftime('%Y-%m-%d')
                ]
                
        stats["total_accounts"] = processed_df['Account Name'].nunique()
        
        return redirect(url_for('dashboard'))
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        error_message = f"Error processing statements: {str(e)}\n\nDetailed error:\n{error_details}"
        print("=" * 80)
        print(error_message)
        print("=" * 80)
        return render_template('error.html', message=error_message, show_details=True)

@app.route('/dashboard')
def dashboard():
    global processed_df, statement_reader
    
    if processed_df is None:
        # No data processed yet, try to initialize from statements directory
        try:
            statement_reader = PDFReader(base_path=Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))))
            processed_df = statement_reader.process_raw_df()
            
            # Check if we have data after processing
            if processed_df is None or processed_df.empty:
                return render_template('error.html', message="No transaction data found. Please upload some bank statements first.", show_details=False)
            
            # If we got here, processing succeeded
            return render_template('dashboard.html')
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return render_template('error.html', message=f"Could not auto-process statements: {str(e)}", show_details=True, error_details=error_details)
    
    return render_template('dashboard.html')

@app.route('/api/transactions')
def get_transactions():
    if processed_df is None:
        return jsonify({"error": "No data has been processed yet"}), 404
    
    # Convert DataFrame to dict for JSON response, handling NaN values
    df_copy = processed_df.copy()
    
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

@app.route('/api/categories')
def get_categories():
    if processed_df is None:
        return jsonify({"error": "No data has been processed yet"}), 404
    
    try:
        # Fill NaN values in Classification with 'Uncategorized'
        df_copy = processed_df.copy()
        df_copy['Classification'] = df_copy['Classification'].fillna('Uncategorized')
        
        # Count transactions by category and sum amounts (negative for expenses)
        category_counts = df_copy['Classification'].value_counts().to_dict()
        
        # Calculate total spending by category (for negative amounts only - expenses)
        spending_df = df_copy[df_copy['Amount'] < 0]
        
        # Handle case where there are no negative amounts (expenses)
        if spending_df.empty:
            category_spending = {}
            print("Warning: No transactions with negative amounts found for spending chart")
        else:
            category_spending = spending_df.groupby('Classification')['Amount'].sum().abs().to_dict()
        
        # Create color mapping for each category
        category_colors = {}
        all_categories = set(list(category_counts.keys()) + list(category_spending.keys()))
        for category in all_categories:
            category_colors[category] = get_category_color(category)
        
        return jsonify({
            "counts": category_counts,
            "spending": category_spending,
            "colors": category_colors,
            "has_spending_data": len(category_spending) > 0
        })
    except Exception as e:
        print(f"Error in categories API: {str(e)}")
        return jsonify({
            "error": str(e),
            "counts": {},
            "spending": {},
            "colors": {},
            "has_spending_data": False
        }), 500

@app.route('/api/balance_chart')
def get_balance_chart():
    if processed_df is None or statement_reader is None:
        return jsonify({"error": "No data has been processed yet"}), 404
    
    # Generate the balance plot data
    helper = GeneralHelperFns()
    df = processed_df.copy()
    df = df[df[['Balance']].notnull().all(axis=1)]
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values(by='DateTime')
    
    # Prepare data for chart.js
    chart_data = {
        "labels": df['DateTime'].dt.strftime('%Y-%m-%d').tolist(),
        "datasets": [{
            "label": "Balance",
            "data": df['Balance'].tolist(),
            "borderColor": "#BB2525",
            "pointRadius": 2,
            "fill": False
        }]
    }
    
    return jsonify(chart_data)

@app.route('/api/pelt_analysis')
def get_pelt_analysis():
    if processed_df is None or statement_reader is None:
        return jsonify({"error": "No data has been processed yet"}), 404
    
    # Set PELT parameters
    segment_settings = {
        'penalty': 100,
        'min_size': 50,
        'jump': 50,
        'model': "l2"
    }
    
    # Generate the PELT analysis data
    helper = GeneralHelperFns()
    df = processed_df.copy()
    df = df[df[['Balance']].notnull().all(axis=1)]
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values(by='DateTime')
    
    # Calculate numeric time and change points
    df['numeric_time'] = (df['DateTime'] - df['DateTime'].min()).dt.total_seconds()
    
    try:
        change_points = helper._GeneralHelperFns__detect_change_points(
            df['Balance'].values,
            penalty=segment_settings['penalty'],
            min_size=segment_settings['min_size'],
            jump=segment_settings['jump'],
            model=segment_settings['model']
        )
        
        segments = helper._GeneralHelperFns__linearize_segments(
            df['numeric_time'].values, 
            df['Balance'].values, 
            change_points
        )
        
        # Prepare segment data
        concat_y_segments = []
        concat_coeffs = []
        change_dates = []
        
        for start, end, coeffs, _ in segments:
            segment_y_values = list(np.polyval(coeffs, df['numeric_time'][start:end]))
            concat_y_segments.extend(segment_y_values)
            # Convert slope to weekly rate of change for better visualization
            weekly_change_rate = coeffs[0] * 86400 * 7  # seconds in a day * 7 days
            concat_coeffs.extend([weekly_change_rate] * (end - start))
        
        # Get the dates for change points
        for cp in change_points:
            if cp < len(df):
                change_dates.append(df['DateTime'].iloc[cp].strftime('%Y-%m-%d'))
        
        # Prepare data for chart.js, making sure all values are JSON serializable
        pelt_data = {
            "labels": df['DateTime'].dt.strftime('%Y-%m-%d').tolist(),
            "datasets": [
                {
                    "label": "Balance",
                    "data": [float(x) if pd.notna(x) else None for x in df['Balance'].tolist()],
                    "borderColor": "#BB2525",
                    "pointRadius": 2,
                    "fill": False,
                    "yAxisID": "y"
                },
                {
                    "label": "Trend Segments",
                    "data": [float(x) if pd.notna(x) else None for x in concat_y_segments],
                    "borderColor": "#BCBF07",
                    "borderWidth": 3,
                    "pointRadius": 0,
                    "fill": False,
                    "yAxisID": "y"
                }
            ],
            "changePoints": change_dates,
            "rateOfChange": {
                "labels": df['DateTime'].dt.strftime('%Y-%m-%d').tolist(),
                "data": [float(x) if pd.notna(x) else None for x in concat_coeffs]
            }
        }
        
        return jsonify(pelt_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/monthly_trends')
def get_monthly_trends():
    if processed_df is None:
        return jsonify({"error": "No data has been processed yet"}), 404
    
    try:
        # Make a copy of the dataframe and ensure DateTime is in datetime format
        df = processed_df.copy()
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        
        # Extract month and year for grouping
        df['YearMonth'] = df['DateTime'].dt.strftime('%Y-%m')
        
        # Fill missing classifications
        df['Classification'] = df['Classification'].fillna('Uncategorized')
        
        # Separate income (positive amounts) and expenses (negative amounts)
        income_df = df[df['Amount'] > 0].copy()
        expense_df = df[df['Amount'] < 0].copy()
        # Don't make expenses positive, keep them negative for proper axis display
        # expense_df['Amount'] = expense_df['Amount'].abs()  # Make expenses positive for charting
        
        # Aggregate monthly income (total)
        monthly_income = income_df.groupby('YearMonth')['Amount'].sum().to_dict()
        
        # Aggregate monthly expenses by category
        monthly_expenses_by_category = expense_df.groupby(['YearMonth', 'Classification'])['Amount'].sum().reset_index()
        
        # Convert to the format needed for stacked bar chart
        categories = sorted(df['Classification'].unique())
        months = sorted(df['YearMonth'].unique())
        
        # Prepare datasets for chart.js
        expense_datasets = []
        
        # Create a dataset for each category
        for category in categories:
            category_data = []
            for month in months:
                # Find the amount for this category and month
                amount = monthly_expenses_by_category[
                    (monthly_expenses_by_category['YearMonth'] == month) & 
                    (monthly_expenses_by_category['Classification'] == category)
                ]['Amount'].sum()
                
                category_data.append(float(amount) if pd.notna(amount) else 0)
            
            expense_datasets.append({
                'label': category,
                'data': category_data,
                'backgroundColor': get_category_color(category),
                'stack': 'expenses'
            })
        
        # Create income dataset
        income_data = [float(monthly_income.get(month, 0)) for month in months]
        income_dataset = {
            'label': 'Income',
            'data': income_data,
            'backgroundColor': get_category_color('Income'),
            'stack': 'income',
            'type': 'bar'  # This allows mixing with the stacked bars
        }
        
        # Create net dataset (income - expenses)
        total_expenses_by_month = expense_df.groupby('YearMonth')['Amount'].sum().to_dict()
        net_data = []
        for month in months:
            income = monthly_income.get(month, 0)
            expense = total_expenses_by_month.get(month, 0)
            net = income + expense  # Expense is already negative
            net_data.append(float(net) if pd.notna(net) else 0)
        
        net_dataset = {
            'label': 'Net',
            'data': net_data,
            'borderColor': 'rgba(54, 162, 235, 1)',
            'backgroundColor': 'rgba(54, 162, 235, 0.2)',
            'borderWidth': 2,
            'type': 'line',
            'fill': False,
            'yAxisID': 'y'
        }
        
        # All datasets including income
        all_datasets = expense_datasets + [income_dataset, net_dataset]
        
        chart_data = {
            'labels': months,
            'datasets': all_datasets
        }
        
        return jsonify(chart_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    global statement_reader
    
    if statement_reader is None:
        return jsonify({"error": "No statement reader initialized"}), 400
    
    try:
        statement_reader.clear_pdf_cache()
        return jsonify({"success": True, "message": "PDF cache cleared successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to clear cache: {str(e)}"}), 500

@app.route('/merchants')
def merchants_dashboard():
    """Render the merchant management dashboard"""
    try:
        # Initialize merchant categorizer
        merchant_categorizer = MerchantCategorizer(base_path=Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))))
        
        # Get merchant counts
        merchant_count = len(merchant_categorizer.get_all_merchants())
        alias_count = len(merchant_categorizer.get_all_aliases())
        
        # Load all potential categories
        categories = []
        try:
            databank_path = os.path.join(os.path.dirname(__file__), '..', '..', 'cached_data', 'databank.json')
            with open(databank_path, 'r') as f:
                databank = json.load(f)
            categories = list(databank.get('categories', {}).keys())
        except Exception as e:
            print(f"Error loading categories: {str(e)}")
            categories = ["Groceries", "Dining", "Transport", "Shopping", "Bills", "Entertainment", "Uncategorized"]
        
        # Check if uncharacterized merchants file exists
        review_path = os.path.join(os.path.dirname(__file__), '..', '..', 'cached_data', 'uncharacterized_merchants.json')
        has_review_data = os.path.exists(review_path)
        
        return render_template('merchants.html', 
                               merchant_count=merchant_count, 
                               alias_count=alias_count,
                               categories=categories,
                               has_review_data=has_review_data)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return render_template('error.html', 
                               message=f"Error initializing merchant dashboard: {str(e)}",
                               show_details=True, 
                               error_details=error_details)

@app.route('/merchants/review')
def merchants_review():
    """Render the page for reviewing uncharacterized merchants"""
    try:
        # Check if uncharacterized merchants file exists
        review_path = os.path.join(os.path.dirname(__file__), '..', '..', 'cached_data', 'uncharacterized_merchants.json')
        if not os.path.exists(review_path):
            return redirect(url_for('merchants_dashboard'))
        
        # Load the uncharacterized merchants
        try:
            with open(review_path, 'r') as f:
                content = f.read().strip()
                if content:
                    merchants_data = json.load(f)
                else:
                    merchants_data = {}
            
            # Count merchants
            merchant_count = len(merchants_data)
            
            # If no merchants to review, redirect back to dashboard
            if merchant_count == 0:
                return redirect(url_for('merchants_dashboard'))
                
        except json.JSONDecodeError:
            # If file exists but is invalid JSON, initialize it as empty
            merchants_data = {}
            merchant_count = 0
            with open(review_path, 'w') as f:
                json.dump({}, f)
            return redirect(url_for('merchants_dashboard'))
        
        # Load categories
        categories = []
        try:
            databank_path = os.path.join(os.path.dirname(__file__), '..', '..', 'cached_data', 'databank.json')
            with open(databank_path, 'r') as f:
                databank = json.load(f)
            categories = list(databank.get('categories', {}).keys())
        except Exception as e:
            print(f"Error loading categories: {str(e)}")
            categories = ["Groceries", "Dining", "Transport", "Shopping", "Bills", "Entertainment", "Uncategorized"]
        
        return render_template('merchants_review.html', 
                               merchant_count=merchant_count,
                               categories=categories)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return render_template('error.html', 
                               message=f"Error initializing merchant review: {str(e)}",
                               show_details=True, 
                               error_details=error_details)

@app.route('/api/merchants')
def get_merchants():
    """API endpoint to get merchants"""
    try:
        # Get search term if provided
        search_term = request.args.get('search', '')
        
        # Initialize merchant categorizer
        merchant_categorizer = MerchantCategorizer(base_path=Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))))
        
        # Get merchants (filtered by search term if provided)
        if search_term:
            merchants = merchant_categorizer.search_merchants(search_term)
        else:
            merchants = merchant_categorizer.get_all_merchants()
            
        # Format as list for easier handling in frontend
        merchant_list = [{"name": name, "category": category} for name, category in merchants.items()]
        
        # Sort by name
        merchant_list.sort(key=lambda x: x["name"])
        
        return jsonify({
            "success": True,
            "merchants": merchant_list,
            "count": len(merchant_list)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/merchants', methods=['POST'])
def update_merchant():
    """API endpoint to update a merchant's category"""
    try:
        data = request.json
        if not data or 'merchant' not in data or 'category' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required fields"
            }), 400
            
        merchant_name = data['merchant']
        category = data['category']
        
        # Initialize merchant categorizer
        merchant_categorizer = MerchantCategorizer(base_path=Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))))
        
        # Update merchant
        success = merchant_categorizer.add_merchant(merchant_name, category)
        
        return jsonify({
            "success": success,
            "merchant": merchant_name,
            "category": category
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/aliases', methods=['POST'])
def add_alias():
    """API endpoint to add a merchant alias"""
    try:
        data = request.json
        if not data or 'alias' not in data or 'merchant' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required fields"
            }), 400
            
        alias = data['alias']
        merchant = data['merchant']
        
        # Initialize merchant categorizer
        merchant_categorizer = MerchantCategorizer(base_path=Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))))
        
        # Add alias
        success = merchant_categorizer.add_alias(alias, merchant)
        
        if not success:
            return jsonify({
                "success": False,
                "error": f"Could not add alias. Ensure '{merchant}' exists in the database."
            }), 400
        
        return jsonify({
            "success": True,
            "alias": alias,
            "merchant": merchant
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/merchants/uncharacterized')
def get_uncharacterized_merchants():
    """API endpoint to get uncharacterized merchants for review"""
    try:
        # Get page and limit parameters
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        
        # Check if uncharacterized merchants file exists
        review_path = os.path.join(os.path.dirname(__file__), '..', '..', 'cached_data', 'uncharacterized_merchants.json')
        if not os.path.exists(review_path):
            return jsonify({
                "success": True,
                "merchants": [],
                "total": 0,
                "page": page,
                "limit": limit,
                "pages": 0
            })
        
        # Load the uncharacterized merchants
        try:
            with open(review_path, 'r') as f:
                content = f.read().strip()
                if content:
                    merchants_data = json.load(f)
                else:
                    merchants_data = {}
        except json.JSONDecodeError:
            # Handle invalid JSON by initializing an empty merchants data
            merchants_data = {}
            # Fix the file by writing an empty JSON object
            with open(review_path, 'w') as f:
                json.dump({}, f)
        
        # Sort by frequency
        sorted_merchants = sorted(merchants_data.items(), 
                                 key=lambda x: x[1]["count"],
                                 reverse=True)
        
        total_merchants = len(sorted_merchants)
        total_pages = (total_merchants + limit - 1) // limit if total_merchants > 0 else 0
        
        # Adjust page if out of bounds
        if page > total_pages and total_pages > 0:
            page = 1
        
        # Paginate results
        start = (page - 1) * limit
        end = start + limit
        paginated_merchants = sorted_merchants[start:end] if total_merchants > 0 else []
        
        # Format for response
        result = []
        for merchant, data in paginated_merchants:
            result.append({
                "merchant": merchant,
                "count": data["count"],
                "total_amount": data["total_amount"],
                "examples": data["examples"]
            })
        
        return jsonify({
            "success": True,
            "merchants": result,
            "total": total_merchants,
            "page": page,
            "limit": limit,
            "pages": total_pages
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/merchants/uncharacterized', methods=['POST'])
def categorize_uncharacterized_merchant():
    """API endpoint to categorize an uncharacterized merchant"""
    try:
        data = request.json
        if not data or 'merchant' not in data or 'category' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required fields"
            }), 400
            
        merchant_name = data['merchant']
        category = data['category']
        
        # Initialize merchant categorizer
        merchant_categorizer = MerchantCategorizer(base_path=Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))))
        
        # Add merchant to database
        success = merchant_categorizer.add_merchant(merchant_name, category)
        
        if success:
            # Remove from uncharacterized list
            review_path = os.path.join(os.path.dirname(__file__), '..', '..', 'cached_data', 'uncharacterized_merchants.json')
            if os.path.exists(review_path):
                with open(review_path, 'r') as f:
                    merchants_data = json.load(f)
                
                # Remove this merchant
                if merchant_name in merchants_data:
                    del merchants_data[merchant_name]
                
                # Save updated data
                with open(review_path, 'w') as f:
                    json.dump(merchants_data, f, indent=2)
        
        return jsonify({
            "success": success,
            "merchant": merchant_name,
            "category": category
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/merchants/uncharacterized/count')
def get_uncharacterized_count():
    """API endpoint to get the count of uncharacterized merchants"""
    try:
        # Check if uncharacterized merchants file exists
        review_path = os.path.join(os.path.dirname(__file__), '..', '..', 'cached_data', 'uncharacterized_merchants.json')
        if not os.path.exists(review_path):
            return jsonify({
                "success": True,
                "count": 0
            })
        
        # Load the uncharacterized merchants
        try:
            with open(review_path, 'r') as f:
                content = f.read().strip()
                if content:
                    merchants_data = json.load(f)
                else:
                    merchants_data = {}
        except json.JSONDecodeError:
            # Handle invalid JSON
            merchants_data = {}
            # Fix the file by writing an empty JSON object
            with open(review_path, 'w') as f:
                json.dump({}, f)
        
        return jsonify({
            "success": True,
            "count": len(merchants_data)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 