from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
import sys
import pandas as pd
import json
from pathlib import Path
import glob
import numpy as np

# Add parent directory to path so we can import our existing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.modules.pdf_interpreter import PDFReader
from src.modules.helper_fns import GeneralHelperFns, CategoryUpdater

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
    
    return render_template('index.html', 
                           account_types=account_types, 
                           statement_counts=statement_counts)

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
        
        # Basic stats to confirm processing
        stats = {
            "total_transactions": len(processed_df),
            "date_range": [processed_df['DateTime'].min().strftime('%Y-%m-%d'), 
                          processed_df['DateTime'].max().strftime('%Y-%m-%d')],
            "total_accounts": processed_df['Account Name'].nunique()
        }
        
        return redirect(url_for('dashboard'))
    except Exception as e:
        return render_template('error.html', message=f"Error processing statements: {str(e)}")

@app.route('/dashboard')
def dashboard():
    global processed_df, statement_reader
    
    if processed_df is None:
        # No data processed yet, try to initialize from statements directory
        try:
            statement_reader = PDFReader(base_path=Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))))
            processed_df = statement_reader.process_raw_df()
            
            # If we got here, processing succeeded
            return render_template('dashboard.html')
        except Exception as e:
            return render_template('error.html', message=f"Could not auto-process statements: {str(e)}")
    
    return render_template('dashboard.html')

@app.route('/api/transactions')
def get_transactions():
    if processed_df is None:
        return jsonify({"error": "No data has been processed yet"}), 404
    
    # Convert DataFrame to dict for JSON response, handling NaN values
    df_copy = processed_df.head(100).copy()
    
    # Replace NaN values with None for JSON serialization
    df_copy = df_copy.where(pd.notna(df_copy), None)
    
    transactions = df_copy.to_dict(orient='records')
    return jsonify(transactions)

@app.route('/api/categories')
def get_categories():
    if processed_df is None:
        return jsonify({"error": "No data has been processed yet"}), 404
    
    # Fill NaN values in Classification with 'Uncategorized'
    df_copy = processed_df.copy()
    df_copy['Classification'] = df_copy['Classification'].fillna('Uncategorized')
    
    # Count transactions by category and sum amounts (negative for expenses)
    category_counts = df_copy['Classification'].value_counts().to_dict()
    
    # Calculate total spending by category (for negative amounts only - expenses)
    category_spending = df_copy[df_copy['Amount'] < 0].groupby('Classification')['Amount'].sum().abs().to_dict()
    
    return jsonify({
        "counts": category_counts,
        "spending": category_spending
    })

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

if __name__ == '__main__':
    app.run(debug=True) 