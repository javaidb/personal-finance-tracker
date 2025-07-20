from flask import Blueprint, render_template, request, redirect, url_for, current_app, jsonify
from werkzeug.utils import secure_filename
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
            # Check if this bank folder has any statement files (PDF or CSV)
            bank_dir = os.path.join(statements_dir, item)
            for root, dirs, files in os.walk(bank_dir):
                if any(f.lower().endswith(('.pdf', '.csv')) for f in files):
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
                    # Count statement files (PDFs and CSVs) in this account directory
                    statement_count = len([f for f in os.listdir(account_dir) if f.lower().endswith(('.pdf', '.csv'))])
                    statement_counts[bank_name][account_type][account_name] = statement_count
        
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

@main_bp.route('/api/accounts/<bank_name>/<account_type>', methods=['POST'])
def create_account_folder(bank_name, account_type):
    """Create a new account folder under the given bank and account type."""
    try:
        data = request.get_json()
        account_name = data.get('account_name')
        if not account_name:
            return jsonify({'success': False, 'error': 'Missing account_name'}), 400
        
        statements_dir = current_app.config.get('STATEMENTS_DIR')
        if not statements_dir:
            return jsonify({'success': False, 'error': 'STATEMENTS_DIR not configured'}), 500
        
        account_type_dir = os.path.join(statements_dir, bank_name, account_type)
        if not os.path.exists(account_type_dir):
            return jsonify({'success': False, 'error': f'Account type {account_type} not found'}), 404
        
        new_folder = os.path.join(account_type_dir, account_name)
        if os.path.exists(new_folder):
            return jsonify({'success': False, 'error': 'Account folder already exists'}), 400
        os.makedirs(new_folder)
        return jsonify({'success': True, 'account_name': account_name})
    except Exception as e:
        logger.error(f"Error creating account folder: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@main_bp.route('/api/accounts/<bank_name>/<account_type>/<old_account_name>', methods=['PUT'])
def rename_account_folder(bank_name, account_type, old_account_name):
    """Rename an account folder under the given bank and account type."""
    try:
        data = request.get_json()
        new_name = data.get('new_name')
        if not new_name:
            return jsonify({'success': False, 'error': 'Missing new_name'}), 400
        
        statements_dir = current_app.config.get('STATEMENTS_DIR')
        if not statements_dir:
            return jsonify({'success': False, 'error': 'STATEMENTS_DIR not configured'}), 500
        
        account_type_dir = os.path.join(statements_dir, bank_name, account_type)
        old_folder = os.path.join(account_type_dir, old_account_name)
        new_folder = os.path.join(account_type_dir, new_name)
        if not os.path.exists(old_folder):
            return jsonify({'success': False, 'error': 'Old account folder does not exist'}), 404
        if os.path.exists(new_folder):
            return jsonify({'success': False, 'error': 'New account folder already exists'}), 400
        os.rename(old_folder, new_folder)
        return jsonify({'success': True, 'old_name': old_account_name, 'new_name': new_name})
    except Exception as e:
        logger.error(f"Error renaming account folder: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@main_bp.route('/api/accounts/<bank_name>/<account_type>/<account_name>', methods=['DELETE'])
def delete_account_folder(bank_name, account_type, account_name):
    """Delete an account folder under the given bank and account type (only if empty)."""
    try:
        statements_dir = current_app.config.get('STATEMENTS_DIR')
        if not statements_dir:
            return jsonify({'success': False, 'error': 'STATEMENTS_DIR not configured'}), 500
        
        account_type_dir = os.path.join(statements_dir, bank_name, account_type)
        folder = os.path.join(account_type_dir, account_name)
        if not os.path.exists(folder):
            return jsonify({'success': False, 'error': 'Account folder does not exist'}), 404
        if os.listdir(folder):
            return jsonify({'success': False, 'error': 'Account folder is not empty'}), 400
        os.rmdir(folder)
        return jsonify({'success': True, 'account_name': account_name})
    except Exception as e:
        logger.error(f"Error deleting account folder: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

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
        
        # Get bank configuration to determine file format
        bank_config_obj = BankConfig()
        bank_config_data = bank_config_obj.get_bank_config(bank_name)
        file_format = bank_config_data.get('file_format', 'pdf')
        
        # Determine file type text based on bank configuration
        file_type_text = 'PDFs' if file_format == 'pdf' else 'CSV files'
        file_extension = '.pdf' if file_format == 'pdf' else '.csv'
        
        # Create README files for each account type
        readme_templates = {
            'Credit': f"""This folder is meant to contain CREDIT accounts.
In this folder make a folder for EACH credit account, and store {file_type_text} as such.
Example: 'Visa Card' account can be called as such here as the folder name, then dump all bank statement {file_type_text} here.
**Do NOT adjust names of files, script assumes they are called the defaults assigned by {bank_name.title()} Online.""",
            
            'Chequing': f"""This folder is meant to contain CHEQUING accounts.
In this folder make a folder for EACH chequing account, and store {file_type_text} as such.
Example: 'Student Banking' account can be called as such here as the folder name, then dump all bank statement {file_type_text} here.
**Do NOT adjust names of files, script assumes they are called the defaults assigned by {bank_name.title()} Online.""",
            
            'Savings': f"""This folder is meant to contain SAVINGS accounts.
In this folder make a folder for EACH savings account, and store {file_type_text} as such.
Example: 'MoneyMaster' account can be called as such here as the folder name, then dump all bank statement {file_type_text} here.
**Do NOT adjust names of files, script assumes they are called the defaults assigned by {bank_name.title()} Online."""
        }
        
        # Create main bank README
        main_readme_content = f"""{bank_name.title()} Bank Statements

This folder contains your {bank_name.title()} bank statements organized by account type.

Folder Structure:
- Credit/          - Credit card statements
- Chequing/        - Current account statements  
- Savings/         - Savings account statements

Each account type folder contains subfolders for individual accounts.
Place your {file_type_text} in the appropriate account folder.

Example:
bank_statements/{bank_name}/
├── Credit/
│   └── YourCreditCard/
│       ├── January 2024 statement{file_extension}
│       └── February 2024 statement{file_extension}
├── Chequing/
│   └── YourChequingAccount/
│       ├── January 2024 statement{file_extension}
│       └── February 2024 statement{file_extension}
└── Savings/
    └── YourSavingsAccount/
        ├── January 2024 statement{file_extension}
        └── February 2024 statement{file_extension}

The system will automatically detect {bank_name.title()} and apply the appropriate
branding and parsing patterns for your statements.

IMPORTANT: Replace the placeholder folder names (like "YourCreditCard") with your actual account names!
"""
        
        # Create main bank README
        main_readme_path = os.path.join(statements_dir, bank_name, 'README.txt')
        try:
            with open(main_readme_path, 'w', encoding='utf-8') as f:
                f.write(main_readme_content)
        except Exception as e:
            logger.error(f"Error creating main README file: {str(e)}")
        
        # Create README files for each account type
        for account_type, readme_content in readme_templates.items():
            readme_path = os.path.join(statements_dir, bank_name, account_type, 'README.txt')
            try:
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(readme_content)
            except Exception as e:
                logger.error(f"Error creating {account_type} README file: {str(e)}")
                # Don't fail the whole operation if README creation fails
        
        return jsonify({
            'success': True, 
            'message': f'Folder structure created for {bank_name}',
            'created_folders': created_folders
        })
        
    except Exception as e:
        logger.error(f"Error creating folders: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': f'Error creating folders: {str(e)}'}), 500

@main_bp.route('/upload-statements', methods=['POST'])
def upload_statements():
    """Upload bank statements to the appropriate account type/account folder."""
    try:
        # Get form data
        bank_name = request.form.get('bank_name')
        account_type = request.form.get('account_type')
        account_name = request.form.get('account_name')
        files = request.files.getlist('files')
        
        if not bank_name or not account_type or not account_name or not files:
            return jsonify({'success': False, 'error': 'Missing required parameters'}), 400
        
        # Validate bank name
        try:
            from src.config.bank_config import BankConfig
            bank_config = BankConfig()
            available_banks = bank_config.get_available_banks()
            
            if bank_name not in available_banks:
                return jsonify({'success': False, 'error': f'Bank "{bank_name}" is not configured'}), 400
        except Exception as e:
            logger.error(f"Error validating bank: {str(e)}")
            return jsonify({'success': False, 'error': 'Error validating bank configuration'}), 500
        
        # Validate account type and convert to proper case
        account_type_mapping = {
            'credit': 'Credit',
            'chequing': 'Chequing', 
            'savings': 'Savings'
        }
        
        if account_type.lower() not in account_type_mapping:
            return jsonify({'success': False, 'error': f'Invalid account type: {account_type}'}), 400
        
        account_type = account_type_mapping[account_type.lower()]
        
        # Get statements directory
        statements_dir = current_app.config.get('STATEMENTS_DIR')
        if not statements_dir:
            return jsonify({'success': False, 'error': 'Statements directory not configured'}), 500
        
        # Create target directory path
        target_dir = os.path.join(statements_dir, bank_name, account_type, account_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # Get bank configuration to determine allowed file types
        bank_config_obj = BankConfig()
        bank_config_data = bank_config_obj.get_bank_config(bank_name)
        file_format = bank_config_data.get('file_format', 'pdf')  # Default to PDF if not specified
        
        # Determine allowed extensions based on bank configuration
        allowed_extensions = []
        if file_format == 'csv':
            allowed_extensions = ['.csv']
        else:
            allowed_extensions = ['.pdf']
        
        uploaded_files = []
        for file in files:
            if file and file.filename:
                # Validate file type based on bank configuration
                file_ext = os.path.splitext(file.filename.lower())[1]
                if file_ext not in allowed_extensions:
                    continue
                
                # Secure filename
                filename = secure_filename(file.filename)
                
                # Save file
                file_path = os.path.join(target_dir, filename)
                file.save(file_path)
                uploaded_files.append(filename)
        
        if not uploaded_files:
            expected_format = 'CSV' if file_format == 'csv' else 'PDF'
            return jsonify({'success': False, 'error': f'No valid {expected_format} files were uploaded'}), 400
        
        return jsonify({
            'success': True,
            'message': f'Successfully uploaded {len(uploaded_files)} files to {account_type}/{account_name}',
            'uploaded_files': uploaded_files,
            'target_directory': target_dir
        })
        
    except Exception as e:
        logger.error(f"Error uploading statements: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': f'Error uploading statements: {str(e)}'}), 500 