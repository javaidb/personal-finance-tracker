from flask import Blueprint, render_template, request, redirect, url_for, current_app, jsonify
import os
from pathlib import Path
from ..services.transaction_service import TransactionService
from ..services.bank_branding_service import BankBrandingService
from typing import List, Dict, Any
import logging
from .api import init_transaction_service
from ..utils.statement_coverage import get_simple_coverage, get_coverage_summary, format_date_range

logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)

def has_bank_statements() -> bool:
    """Check if any bank statements exist."""
    statements_dir = current_app.config.get('STATEMENTS_DIR')
    if not statements_dir or not os.path.exists(statements_dir):
        return False
    
    # Check if any bank folders exist
    for item in os.listdir(statements_dir):
        if os.path.isdir(os.path.join(statements_dir, item)):
            # Check if this bank folder has any PDF files
            bank_dir = os.path.join(statements_dir, item)
            for root, dirs, files in os.walk(bank_dir):
                if any(f.endswith('.pdf') for f in files):
                    return True
    return False

@main_bp.route('/')
def index() -> str:
    """Render the main page."""
    try:
        # Check if this is first launch (no bank statements)
        # Allow users to skip setup by adding ?skip_setup=true to URL
        skip_setup = request.args.get('skip_setup', 'false').lower() == 'true'
        if not has_bank_statements() and not skip_setup:
            return redirect(url_for('main.setup'))
        
        # Initialize bank branding service
        bank_branding_service = BankBrandingService()
        
        # Detect which bank is being used
        detected_bank = None
        try:
            from src.config.bank_config import BankConfig
            bank_config = BankConfig()
            detected_bank = bank_config.detect_bank_from_structure()
        except Exception as e:
            logger.error(f"Error detecting bank: {str(e)}")
        
        # Get bank branding info
        bank_branding = None
        if detected_bank:
            try:
                bank_branding = bank_branding_service.get_bank_display_info(detected_bank)
            except Exception as e:
                logger.error(f"Error getting bank branding for {detected_bank}: {str(e)}")
        
        # Get list of banks (directories)
        banks = []
        statements_dir = current_app.config.get('STATEMENTS_DIR')
        if statements_dir and os.path.exists(statements_dir):
            for item in os.listdir(statements_dir):
                if os.path.isdir(os.path.join(statements_dir, item)):
                    banks.append(item)
        
        # Get statement counts by bank and account type
        statement_counts = {}
        for bank_name in banks:
            bank_dir = os.path.join(statements_dir, bank_name)
            account_types = []
            
            # Get account types for this bank
            for item in os.listdir(bank_dir):
                if os.path.isdir(os.path.join(bank_dir, item)):
                    account_types.append(item)
            
            statement_counts[bank_name] = {}
            for account_type in account_types:
                account_type_dir = os.path.join(bank_dir, account_type)
                account_names = []
                
                # Get account names for this account type
                for item in os.listdir(account_type_dir):
                    if os.path.isdir(os.path.join(account_type_dir, item)):
                        account_names.append(item)
                
                statement_counts[bank_name][account_type] = {}
                for account_name in account_names:
                    account_dir = os.path.join(account_type_dir, account_name)
                    # Count PDFs in this account directory
                    pdf_count = len([f for f in os.listdir(account_dir) if f.endswith('.pdf')])
                    statement_counts[bank_name][account_type][account_name] = pdf_count
        
        # Get cache info
        cache_info = {"cached_pdfs_count": 0, "cache_size_kb": 0}
        try:
            service = init_transaction_service()
            if service:
                cache_info = service.pdf_reader.get_cache_info()
        except Exception as e:
            logger.error(f"Error getting cache info: {str(e)}")
        
        # Get statement coverage information
        coverage_info = {}
        coverage_summary = {}
        try:
            coverage_info = get_simple_coverage(statements_dir)
            coverage_summary = get_coverage_summary(coverage_info)
        except Exception as e:
            logger.error(f"Error getting coverage info: {str(e)}")
        
        return render_template('index.html',
                             banks=banks,
                             statement_counts=statement_counts,
                             cache_info=cache_info,
                             coverage_info=coverage_info,
                             coverage_summary=coverage_summary,
                             bank_branding=bank_branding,
                             detected_bank=detected_bank)
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}", exc_info=True)
        return render_template('error.html', message="Failed to load account types")

@main_bp.route('/setup')
def setup() -> str:
    """First launch setup page."""
    try:
        # Initialize bank branding service
        bank_branding_service = BankBrandingService()
        
        # Get all available banks
        try:
            from src.config.bank_config import BankConfig
            bank_config = BankConfig()
            available_banks = bank_config.get_available_banks()
        except Exception as e:
            logger.error(f"Error getting available banks: {str(e)}")
            available_banks = []
        
        # Get bank display info for each available bank
        banks_info = []
        for bank_name in available_banks:
            try:
                bank_info = bank_branding_service.get_bank_display_info(bank_name)
                banks_info.append(bank_info)
            except Exception as e:
                logger.error(f"Error getting bank info for {bank_name}: {str(e)}")
                # Add basic info if branding service fails
                banks_info.append({
                    'name': bank_name,
                    'display_name': bank_name.title(),
                    'logo_path': '/static/images/banks/default/logo.svg',
                    'theme_class': f'theme-{bank_name.lower()}',
                    'logo_exists': False
                })
        
        return render_template('setup.html', banks=banks_info)
    except Exception as e:
        logger.error(f"Error in setup route: {str(e)}", exc_info=True)
        return render_template('error.html', message="Failed to load setup page")

@main_bp.route('/process_statements', methods=['POST'])
def process_statements():
    """Process bank statements route."""
    try:
        service = init_transaction_service()
        if not service:
            return render_template('error.html',
                                message="Could not initialize transaction service",
                                show_details=False)
        
        success = service.process_statements()
        
        if success:
            return redirect(url_for('main.dashboard'))
        else:
            return render_template('error.html',
                                message="No transactions found in the selected statements.",
                                show_details=False)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return render_template('error.html',
                            message=f"Error processing statements: {str(e)}",
                            show_details=True,
                            error_details=error_details)

@main_bp.route('/dashboard')
def dashboard():
    """Dashboard route."""
    try:
        # Initialize bank branding service
        bank_branding_service = BankBrandingService()
        
        # Detect which bank is being used
        detected_bank = None
        try:
            from src.config.bank_config import BankConfig
            bank_config = BankConfig()
            detected_bank = bank_config.detect_bank_from_structure()
        except Exception as e:
            logger.error(f"Error detecting bank: {str(e)}")
        
        # Get bank branding info
        bank_branding = None
        if detected_bank:
            try:
                bank_branding = bank_branding_service.get_bank_display_info(detected_bank)
            except Exception as e:
                logger.error(f"Error getting bank branding for {detected_bank}: {str(e)}")
        
        service = init_transaction_service()
        if not service or service.processed_df is None:
            # No data processed yet, try to process
            if service:
                success = service.process_statements()
                if success:
                    return render_template('dashboard.html', 
                                         categoryColors={},
                                         bank_branding=bank_branding,
                                         detected_bank=detected_bank)
            
            return render_template('error.html',
                                message="No transaction data found. Please upload some bank statements first.",
                                show_details=False)
        
        # Get category colors from the service
        category_colors = service.get_category_colors() if hasattr(service, 'get_category_colors') else {}
        return render_template('dashboard.html', 
                             categoryColors=category_colors,
                             bank_branding=bank_branding,
                             detected_bank=detected_bank)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return render_template('error.html',
                            message=f"Could not load dashboard: {str(e)}",
                            show_details=True,
                            error_details=error_details)

@main_bp.route('/api/accounts/<bank_name>/<account_type>')
def get_accounts(bank_name: str, account_type: str) -> Dict[str, Any]:
    """Get list of accounts for a given bank and account type."""
    try:
        statements_dir = current_app.config.get('STATEMENTS_DIR')
        if not statements_dir:
            return jsonify({
                "success": False,
                "error": "STATEMENTS_DIR not configured"
            }), 500

        account_type_dir = os.path.join(statements_dir, bank_name, account_type)
        if not os.path.exists(account_type_dir):
            return jsonify({
                "success": False,
                "error": f"Account type '{account_type}' not found for bank '{bank_name}'"
            }), 404
        
        accounts = []
        for item in os.listdir(account_type_dir):
            if os.path.isdir(os.path.join(account_type_dir, item)):
                accounts.append(item)
        
        return jsonify({
            "success": True,
            "accounts": accounts
        })
    except Exception as e:
        logger.error(f"Error getting accounts for {bank_name}/{account_type}: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Failed to get accounts"
        }), 500 

@main_bp.route('/create-folders', methods=['POST'])
def create_folders():
    """Create folder structure for selected bank."""
    try:
        data = request.get_json()
        bank_name = data.get('bank_name')
        
        if not bank_name:
            return jsonify({'success': False, 'error': 'Bank name is required'}), 400
        
        # Validate that the bank exists in configurations
        try:
            from src.config.bank_config import BankConfig
            bank_config = BankConfig()
            available_banks = bank_config.get_available_banks()
            
            if bank_name not in available_banks:
                return jsonify({'success': False, 'error': f'Bank "{bank_name}" is not configured'}), 400
        except Exception as e:
            logger.error(f"Error validating bank: {str(e)}")
            return jsonify({'success': False, 'error': 'Error validating bank configuration'}), 500
        
        # Create folder structure
        statements_dir = current_app.config.get('STATEMENTS_DIR')
        if not statements_dir:
            return jsonify({'success': False, 'error': 'Statements directory not configured'}), 500
        
        # Ensure bank_statements directory exists
        os.makedirs(statements_dir, exist_ok=True)
        
        # Create bank folder structure
        folders_to_create = [
            os.path.join(statements_dir, bank_name),
            os.path.join(statements_dir, bank_name, 'Credit'),
            os.path.join(statements_dir, bank_name, 'Credit', 'YourCreditCard'),
            os.path.join(statements_dir, bank_name, 'Chequing'),
            os.path.join(statements_dir, bank_name, 'Chequing', 'YourChequingAccount'),
            os.path.join(statements_dir, bank_name, 'Savings'),
            os.path.join(statements_dir, bank_name, 'Savings', 'YourSavingsAccount')
        ]
        
        created_folders = []
        for folder in folders_to_create:
            try:
                os.makedirs(folder, exist_ok=True)
                created_folders.append(folder)
            except Exception as e:
                logger.error(f"Error creating folder {folder}: {str(e)}")
                return jsonify({'success': False, 'error': f'Error creating folder: {str(e)}'}), 500
        
        # Create a README file in the bank folder
        readme_content = f"""{bank_name.title()} Bank Statements

This folder contains your {bank_name.title()} bank statements organized by account type.

Folder Structure:
- Credit/          - Credit card statements
- Chequing/        - Current account statements  
- Savings/         - Savings account statements

Each account type folder contains subfolders for individual accounts.
Place your PDF statements in the appropriate account folder.

Example:
bank_statements/{bank_name}/
├── Credit/
│   └── YourCreditCard/
│       ├── January 2024 statement.pdf
│       └── February 2024 statement.pdf
├── Chequing/
│   └── YourChequingAccount/
│       ├── January 2024 statement.pdf
│       └── February 2024 statement.pdf
└── Savings/
    └── YourSavingsAccount/
        ├── January 2024 statement.pdf
        └── February 2024 statement.pdf

The system will automatically detect {bank_name.title()} and apply the appropriate
branding and parsing patterns for your statements.

IMPORTANT: Replace the placeholder folder names (like "YourCreditCard") with your actual account names!
"""
        
        readme_path = os.path.join(statements_dir, bank_name, 'README.txt')
        try:
            with open(readme_path, 'w') as f:
                f.write(readme_content)
        except Exception as e:
            logger.error(f"Error creating README file: {str(e)}")
            # Don't fail the whole operation if README creation fails
        
        return jsonify({
            'success': True, 
            'message': f'Folder structure created for {bank_name}',
            'created_folders': created_folders
        })
        
    except Exception as e:
        logger.error(f"Error creating folders: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': f'Error creating folders: {str(e)}'}), 500 