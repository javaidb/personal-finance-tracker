import os
import re
from datetime import datetime, date
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import calendar

def extract_date_from_filename(filename: str) -> Tuple[int, int]:
    """
    Extract year and month from a statement filename.
    Returns (year, month) or (None, None) if no date found.
    """
    # Remove file extension
    name = os.path.splitext(filename)[0].lower()
    
    # Common patterns for statement filenames
    patterns = [
        r'(\w+)\s+(\d{4})\s+e?-?statement',  # "May 2025 e-statement"
        r'(\w+)\s+(\d{4})\s+e?-?statement\s*',  # "May 2025 e-Statement "
        r'(\w+)\s+(\d{4})_e?statement',  # "APR 2022_eStatement"
        r'(\w+)\s+(\d{4})',  # "May 2025"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            month_str = match.group(1)
            year = int(match.group(2))
            
            # Convert month name to number
            month_map = {
                'jan': 1, 'january': 1,
                'feb': 2, 'february': 2,
                'mar': 3, 'march': 3,
                'apr': 4, 'april': 4,
                'may': 5,
                'jun': 6, 'june': 6,
                'jul': 7, 'july': 7,
                'aug': 8, 'august': 8,
                'sep': 9, 'september': 9,
                'oct': 10, 'october': 10,
                'nov': 11, 'november': 11,
                'dec': 12, 'december': 12
            }
            
            month = month_map.get(month_str[:3].lower())
            if month:
                return year, month
    
    return None, None

def get_simple_coverage(statements_dir: str) -> Dict[str, Dict]:
    """
    Get simplified coverage information for each account (variant) under each account type.
    Returns coverage data with monthly indicators from earliest statement to current month.
    Now supports the new bank structure: bank_statements/[bank_name]/[account_type]/[account_name]/
    """
    coverage = {}
    
    if not os.path.exists(statements_dir):
        return coverage
    
    # Get current date
    now = datetime.now()
    current_year = now.year
    current_month = now.month
    
    # Collect all statement dates across all accounts to determine global min/max
    all_statement_dates = set()
    
    # First, check if we have the new bank structure or old structure
    has_bank_folders = False
    for item in os.listdir(statements_dir):
        item_path = os.path.join(statements_dir, item)
        if os.path.isdir(item_path):
            # Check if this looks like a bank folder (contains account types)
            sub_items = os.listdir(item_path)
            if any(sub_item.lower() in ['credit', 'chequing', 'savings'] for sub_item in sub_items):
                has_bank_folders = True
                break
    
    if has_bank_folders:
        # New structure: bank_statements/[bank_name]/[account_type]/[account_name]/
        for bank_name in os.listdir(statements_dir):
            bank_path = os.path.join(statements_dir, bank_name)
            if not os.path.isdir(bank_path):
                continue
            for account_type in os.listdir(bank_path):
                account_type_path = os.path.join(bank_path, account_type)
                if not os.path.isdir(account_type_path):
                    continue
                for account_name in os.listdir(account_type_path):
                    account_path = os.path.join(account_type_path, account_name)
                    if not os.path.isdir(account_path):
                        continue
                    pdf_files = [f for f in os.listdir(account_path) if f.lower().endswith('.pdf')]
                    for pdf_file in pdf_files:
                        year, month = extract_date_from_filename(pdf_file)
                        if year and month:
                            all_statement_dates.add((year, month))
    else:
        # Old structure: bank_statements/[account_type]/[account_name]/
        for account_type in os.listdir(statements_dir):
            account_type_path = os.path.join(statements_dir, account_type)
            if not os.path.isdir(account_type_path):
                continue
            for account_name in os.listdir(account_type_path):
                account_path = os.path.join(account_type_path, account_name)
                if not os.path.isdir(account_path):
                    continue
                pdf_files = [f for f in os.listdir(account_path) if f.lower().endswith('.pdf')]
                for pdf_file in pdf_files:
                    year, month = extract_date_from_filename(pdf_file)
                    if year and month:
                        all_statement_dates.add((year, month))
    
    if not all_statement_dates:
        return coverage
    min_date = min(all_statement_dates)
    # Use current month as end date (but don't include current month in coverage)
    end_year = current_year
    end_month = current_month - 1  # Go up to previous month
    if end_month == 0:
        end_month = 12
        end_year = current_year - 1
    # Generate all expected months from earliest to current
    expected_months = []
    current_date = min_date
    while current_date <= (end_year, end_month):
        expected_months.append(current_date)
        year, month = current_date
        if month == 12:
            current_date = (year + 1, 1)
        else:
            current_date = (year, month + 1)
    
    # Now, for each account type and account, build monthly coverage
    if has_bank_folders:
        # New structure: bank_statements/[bank_name]/[account_type]/[account_name]/
        for bank_name in os.listdir(statements_dir):
            bank_path = os.path.join(statements_dir, bank_name)
            if not os.path.isdir(bank_path):
                continue
            for account_type in os.listdir(bank_path):
                account_type_path = os.path.join(bank_path, account_type)
                if not os.path.isdir(account_type_path):
                    continue
                if account_type not in coverage:
                    coverage[account_type] = {}
                for account_name in os.listdir(account_type_path):
                    account_path = os.path.join(account_type_path, account_name)
                    if not os.path.isdir(account_path):
                        continue
                    account_dates = set()
                    pdf_files = [f for f in os.listdir(account_path) if f.lower().endswith('.pdf')]
                    for pdf_file in pdf_files:
                        year, month = extract_date_from_filename(pdf_file)
                        if year and month:
                            account_dates.add((year, month))
                    # Generate monthly coverage for this account
                    monthly_coverage = {}
                    for year, month in expected_months:
                        if year not in monthly_coverage:
                            monthly_coverage[year] = {}
                        has_coverage = (year, month) in account_dates
                        monthly_coverage[year][month] = has_coverage
                    coverage[account_type][account_name] = {
                        'monthly_coverage': monthly_coverage,
                        'total_statements': len(account_dates),
                        'date_range': (min_date, (end_year, end_month))
                    }
    else:
        # Old structure: bank_statements/[account_type]/[account_name]/
        for account_type in os.listdir(statements_dir):
            account_type_path = os.path.join(statements_dir, account_type)
            if not os.path.isdir(account_type_path):
                continue
            coverage[account_type] = {}
            for account_name in os.listdir(account_type_path):
                account_path = os.path.join(account_type_path, account_name)
                if not os.path.isdir(account_path):
                    continue
                account_dates = set()
                pdf_files = [f for f in os.listdir(account_path) if f.lower().endswith('.pdf')]
                for pdf_file in pdf_files:
                    year, month = extract_date_from_filename(pdf_file)
                    if year and month:
                        account_dates.add((year, month))
                # Generate monthly coverage for this account
                monthly_coverage = {}
                for year, month in expected_months:
                    if year not in monthly_coverage:
                        monthly_coverage[year] = {}
                    has_coverage = (year, month) in account_dates
                    monthly_coverage[year][month] = has_coverage
                coverage[account_type][account_name] = {
                    'monthly_coverage': monthly_coverage,
                    'total_statements': len(account_dates),
                    'date_range': (min_date, (end_year, end_month))
                }
    return coverage

def get_coverage_summary(coverage: Dict) -> Dict:
    total_statements = 0
    total_months = 0
    covered_months = 0
    account_types = len(coverage)
    account_variants = 0
    for account_type, accounts in coverage.items():
        for account_name, account_data in accounts.items():
            account_variants += 1
            total_statements += account_data.get('total_statements', 0)
            monthly_coverage = account_data.get('monthly_coverage', {})
            for year, months in monthly_coverage.items():
                for month, has_coverage in months.items():
                    total_months += 1
                    if has_coverage:
                        covered_months += 1
    coverage_percentage = (covered_months / total_months * 100) if total_months > 0 else 0
    return {
        'total_statements': total_statements,
        'account_types': account_types,
        'account_variants': account_variants,
        'total_months': total_months,
        'covered_months': covered_months,
        'coverage_percentage': round(coverage_percentage, 1)
    }

def format_date_range(date_tuple: Tuple[int, int]) -> str:
    """Format a (year, month) tuple as a readable string."""
    year, month = date_tuple
    month_name = calendar.month_name[month]
    return f"{month_name} {year}" 