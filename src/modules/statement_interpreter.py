"""
General Statement Interpreter

This module provides a unified interface for parsing bank statements in various formats (PDF, CSV, etc.)
using a modular architecture with format-specific parsers and bank-specific configurations.
"""

import os
import json
import pandas as pd
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import hashlib

from src.modules.helper_fns import GeneralHelperFns
from src.modules.merchant_categorizer import MerchantCategorizer
from src.config.paths import (
    DATABANK_PATH, 
    UNCATEGORIZED_MERCHANTS_PATH, 
    DINING_KEYWORDS_PATH, 
    SHOPPING_KEYWORDS_PATH,
    MANUAL_CATEGORIES_PATH,
    CATEGORY_COLORS_PATH,
    paths
)


class StatementParser(ABC):
    """Abstract base class for statement parsers."""
    
    @abstractmethod
    def extract_text(self, file_path: str) -> List[str]:
        """Extract text content from the statement file."""
        pass
    
    @abstractmethod
    def parse_transactions(self, lines: List[str], config: Dict) -> List[Dict]:
        """Parse transactions from extracted text lines."""
        pass


class PDFParser(StatementParser):
    """PDF statement parser using pdfplumber."""
    
    def __init__(self, x_tolerance: int = 2):
        self.x_tolerance = x_tolerance
    
    def extract_text(self, file_path: str) -> List[str]:
        """Extract text from PDF using pdfplumber."""
        import pdfplumber
        
        text = ''
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Processing PDF: {os.path.basename(file_path)} ({total_pages} pages)")
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # First try to extract text without cropping
                    page_text = page.extract_text(x_tolerance=self.x_tolerance)
                    if page_text:
                        text += page_text + '\n'
                        continue

                    # If no text was extracted, try with cropping
                    page_width = float(page.width)
                    page_height = float(page.height)
                    
                    # Try different margin sizes
                    margin_sizes = [(0.05, 0.05), (0.08, 0.08), (0.10, 0.10), (0.15, 0.15)]
                    
                    for margin_x_pct, margin_y_pct in margin_sizes:
                        try:
                            margin_x = page_width * margin_x_pct
                            margin_y = page_height * margin_y_pct
                            
                            crop_box = (
                                margin_x, margin_y,
                                page_width - margin_x, page_height - margin_y
                            )
                            
                            cropped = page.crop(crop_box)
                            if cropped is not None:
                                page_text = cropped.extract_text(x_tolerance=self.x_tolerance)
                                if page_text:
                                    text += page_text + '\n'
                                    break
                        except Exception as e:
                            continue
                    
                except Exception as e:
                    print(f"Failed to process page {page_num}/{total_pages}: {str(e)}")
                    continue

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        print(f"Extracted {len(lines)} non-empty lines from PDF")
        return lines
    
    def parse_transactions(self, lines: List[str], config: Dict) -> List[Dict]:
        """Parse transactions from PDF lines using bank-specific patterns."""
        transactions = []
        pattern_config = config.get('patterns', {})
        statement_parsing = config.get('statement_parsing', {})
        account_types_config = config.get('account_types', {})
        
        # Find opening balance
        opening_balance = self._find_opening_balance(lines, statement_parsing)
        running_balance = opening_balance or 0.0
        
        # Get account type pattern - map account type to statement format
        account_type = config.get('account_type', 'deposit_account')
        statement_format = 'deposit_account'  # default
        
        # Look up the statement format for this account type
        if account_type in account_types_config:
            statement_format = account_types_config[account_type].get('statement_format', 'deposit_account')
        
        pattern_info = pattern_config.get(statement_format, {})
        pattern = pattern_info.get('pattern')
        groups = pattern_info.get('groups', {})
        
        if not pattern:
            print(f"No pattern found for account type: {account_type} (statement_format: {statement_format})")
            return transactions
        
        
        # Special handling for Barclays multi-line format
        if config.get('bank_name') == 'barclays':
            return self._parse_barclays_transactions(lines, pattern, groups, running_balance, config)
        
        # Process transactions (original logic for other banks)
        for i, line in enumerate(lines):
            match = self._match_pattern(line, pattern, groups)
            if match:
                transaction = self._create_transaction(match, groups, running_balance, config)
                if transaction:
                    transactions.append(transaction)
                    # Convert Balance back to float for running balance calculation
                    balance_str = transaction.get('Balance', str(running_balance))
                    running_balance = float(balance_str) if balance_str else running_balance
        
        return transactions
    
    def _find_opening_balance(self, lines: List[str], statement_parsing: Dict) -> Optional[float]:
        """Find opening balance from statement lines."""
        import re
        
        opening_keywords = statement_parsing.get('opening_balance_keywords', [])
        for line in lines:
            for keyword in opening_keywords:
                if keyword.upper() in line.upper():
                    try:
                        balance_match = re.search(r'balance\s+([\d,]+\.\d{2})', line, re.IGNORECASE)
                        if balance_match:
                            return float(balance_match.group(1).replace(',', ''))
                    except Exception as e:
                        print(f"Error extracting opening balance from line: {line}")
        return None
    
    def _match_pattern(self, line: str, pattern: str, groups: Dict) -> Optional[Dict]:
        """Match a line against the pattern and return groups."""
        import re
        # Fix double-escaped backslashes in JSON patterns
        fixed_pattern = pattern.replace('\\\\', '\\')
        # Use re.search instead of re.match to handle lines that don't start with the pattern
        match = re.search(fixed_pattern, line)
        if match:
            result = {}
            for group_name, group_index in groups.items():
                if group_index <= len(match.groups()):
                    result[group_name] = match.group(group_index)
            return result
        return None
    
    def _create_transaction(self, match: Dict, groups: Dict, running_balance: float, config: Dict) -> Optional[Dict]:
        """Create a transaction dictionary from pattern match."""
        try:
            # Handle different account types - map account type to statement format
            account_type = config.get('account_type', 'deposit_account')
            account_types_config = config.get('account_types', {})
            statement_format = 'deposit_account'  # default
            
            # Look up the statement format for this account type
            if account_type in account_types_config:
                statement_format = account_types_config[account_type].get('statement_format', 'deposit_account')
            
            if statement_format == 'credit_card':
                return self._create_credit_transaction(match, running_balance)
            else:
                return self._create_deposit_transaction(match, running_balance)
        except Exception as e:
            print(f"Error creating transaction: {str(e)}")
            return None
    
    def _create_credit_transaction(self, match: Dict, running_balance: float) -> Dict:
        """Create credit card transaction."""
        ref_num = match.get('reference_number', '')
        transaction_date = match.get('transaction_date', '')
        post_date = match.get('post_date', '')
        details_raw = match.get('details', '')
        details = details_raw.strip() if details_raw else ''
        amount_str = match.get('amount', '0')
        amount = float(amount_str) if amount_str else 0.0
        
        # Handle negative indicator
        if match.get('negative_indicator') == '-':
            amount = -amount
        
        running_balance += amount
        
        return {
            'Reference #': ref_num,
            'Transaction Date': transaction_date,
            'Post Date': post_date,
            'Details': details,
            'Amount': str(amount),
            'Balance': str(running_balance),
            'Transaction Type': '',
        }
    
    def _create_deposit_transaction(self, match: Dict, running_balance: float) -> Dict:
        """Create deposit account transaction."""
        date = match.get('date', '')
        transaction_type = match.get('description', '').strip()
        
        # Handle None values safely
        amount_str = match.get('amount', '0')
        if amount_str is None:
            amount_str = '0'
        amount = float(amount_str.replace(',', ''))
        
        balance_str = match.get('balance', '0')
        if balance_str is None:
            balance_str = '0'
        balance = float(balance_str.replace(',', ''))
        
        return {
            'Transaction Date': date,
            'Transaction Type': transaction_type,
            'Amount': str(amount),
            'Balance': str(balance),
            'Details': transaction_type,
        }
    
    def _parse_barclays_transactions(self, lines: List[str], pattern: str, groups: Dict, running_balance: float, config: Dict) -> List[Dict]:
        """Special processor for Barclays multi-line transactions with Date Description Amount Balance format."""
        import re
        
        print(f"DEBUG: Starting Barclays transaction processing with pattern: {pattern}")
        transactions = []
        i = 0
        
        # Check if this is a two-column format (Money In/Money Out)
        two_column_format = False
        for line in lines[:10]:  # Check first 10 lines for header
            if 'Money out' in line and 'Money in' in line:
                two_column_format = True
                print("DEBUG: Detected two-column format (Money In/Money Out)")
                break
        
        # Get two-column pattern if available
        two_column_pattern = None
        two_column_groups = {}
        if two_column_format:
            patterns_config = config.get('patterns', {})
            two_column_config = patterns_config.get('two_column_format', {})
            two_column_pattern = two_column_config.get('pattern')
            two_column_groups = two_column_config.get('groups', {})
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and headers
            if not line or line in ['Date Description Money out Money in Balance', 'Your transactions'] or 'Date' in line and 'Description' in line and 'Money out' in line and 'Money in' in line and 'Balance' in line:
                i += 1
                continue
            
            # Check if this line starts with a date (DD MMM format)
            # Only match actual month names, not any 3-letter word
            date_match = re.match(r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))', line)
            if date_match:
                date = date_match.group(1)
                print(f"DEBUG: Found date line: '{line}'")
                
                # Skip balance lines and header lines
                if any(skip_word in line.lower() for skip_word in ['start balance', 'end balance']):
                    i += 1
                    continue
                
                # Try two-column format first if detected
                if two_column_format and two_column_pattern:
                    two_column_match = re.match(two_column_pattern, line)
                    if two_column_match:
                        print(f"DEBUG: Two-column format match: '{line}'")
                        match_groups = list(two_column_match.groups())
                        
                        transaction_type = match_groups[two_column_groups.get('description', 1)].strip() if match_groups[two_column_groups.get('description', 1)] is not None else ''
                        money_out_str = match_groups[two_column_groups.get('money_out', 2)].strip() if match_groups[two_column_groups.get('money_out', 2)] is not None else ''
                        money_in_str = match_groups[two_column_groups.get('money_in', 3)].strip() if match_groups[two_column_groups.get('money_in', 3)] is not None else ''
                        balance_str = match_groups[two_column_groups.get('balance', 4)].strip() if match_groups[two_column_groups.get('balance', 4)] is not None else None
                        
                        # Skip balance lines and header lines
                        if any(skip_word in transaction_type.lower() for skip_word in ['start balance', 'end balance']):
                            i += 1
                            continue
                        
                        # Determine amount and sign based on which column has a value
                        amount = 0.0
                        if money_out_str and money_out_str.strip():
                            amount = -float(money_out_str.replace(',', ''))  # Money out is negative
                            print(f"DEBUG: Money out transaction: {amount}")
                        elif money_in_str and money_in_str.strip():
                            amount = float(money_in_str.replace(',', ''))  # Money in is positive
                            print(f"DEBUG: Money in transaction: {amount}")
                        
                        # Handle balance
                        if balance_str is not None:
                            balance = float(balance_str.strip().replace(',', ''))
                        else:
                            # Estimate balance if not provided
                            balance = running_balance + amount
                        
                        # Look for continuation lines (but limit the search)
                        details = transaction_type
                        j = i + 1
                        continuation_count = 0
                        while j < len(lines) and continuation_count < 3:  # Limit to 3 continuation lines
                            next_line = lines[j].strip()
                            # Stop if we hit another date or empty line
                            if not next_line or re.match(r'^\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', next_line):
                                break
                            # Stop if we hit another transaction line (has amount)
                            if re.search(r'[\d,]+\.[\d]{2}', next_line):
                                break
                            # Skip reference lines and continuation indicators
                            if (not next_line.startswith('Ref:') and 
                                not next_line.startswith('On ') and 
                                next_line != 'May' and
                                not next_line.startswith('Continued')):
                                details += f" - {next_line}"
                                continuation_count += 1
                            j += 1
                        
                        transaction = {
                            'Transaction Date': date,
                            'Transaction Type': transaction_type,
                            'Amount': str(amount),
                            'Balance': str(balance),
                            'Details': details,
                        }
                        transactions.append(transaction)
                        running_balance = balance
                        i = j  # Skip to the line after continuations
                        continue
                
                # Try to match the full pattern first (single-line transaction)
                full_match = re.match(pattern, line)
                if full_match:
                    print(f"DEBUG: Full pattern match: '{line}'")
                    # Single-line transaction with Date Description Amount Balance format
                    match_groups = list(full_match.groups())
                    transaction_type = match_groups[1].strip() if match_groups[1] is not None else ''
                    amount_str = match_groups[2].strip() if match_groups[2] is not None else ''
                    balance_str = match_groups[3].strip() if match_groups[3] is not None else None
                    
                    # Skip balance lines and header lines
                    if any(skip_word in transaction_type.lower() for skip_word in ['start balance', 'end balance']):
                        i += 1
                        continue
                    
                    # Parse amount
                    if amount_str:
                        amount = float(amount_str.replace(',', ''))
                    else:
                        amount = 0.0
                    
                    # Determine if this is money out or money in based on transaction type
                    is_money_out = any(indicator in transaction_type.lower() for indicator in [
                        'card payment', 'bill payment', 'direct debit', 'withdrawal', 'transfer out'
                    ])
                    
                    # Received From transactions are always money in
                    is_money_in = 'received from' in transaction_type.lower()
                    
                    # If it's money out, make the amount negative
                    if is_money_out and not is_money_in:
                        amount = -abs(amount)
                        print(f"DEBUG: Money out transaction: {amount}")
                    else:
                        print(f"DEBUG: Money in transaction: {amount}")
                    
                    # Handle balance
                    if balance_str is not None:
                        balance = float(balance_str.strip().replace(',', ''))
                    else:
                        # Estimate balance if not provided
                        balance = running_balance + amount
                    
                    # Look for continuation lines (but limit the search)
                    details = transaction_type
                    j = i + 1
                    continuation_count = 0
                    while j < len(lines) and continuation_count < 3:  # Limit to 3 continuation lines
                        next_line = lines[j].strip()
                        # Stop if we hit another date or empty line
                        if not next_line or re.match(r'^\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', next_line):
                            break
                        # Stop if we hit another transaction line (has amount)
                        if re.search(r'[\d,]+\.[\d]{2}', next_line):
                            break
                        # Skip reference lines and continuation indicators
                        if (not next_line.startswith('Ref:') and 
                            not next_line.startswith('On ') and 
                            next_line != 'May' and
                            not next_line.startswith('Continued')):
                            details += f" - {next_line}"
                            continuation_count += 1
                        j += 1
                    
                    transaction = {
                        'Transaction Date': date,
                        'Transaction Type': transaction_type,
                        'Amount': str(amount),
                        'Balance': str(balance),
                        'Details': details,
                    }
                    transactions.append(transaction)
                    running_balance = balance
                    i = j  # Skip to the line after continuations
                else:
                    print(f"DEBUG: No full pattern match for: '{line}'")
                    # Multi-line transaction - process all transactions for this date
                    
                    # Look ahead to find all transaction lines for this date
                    j = i + 1
                    date_transactions = []
                    
                    while j < len(lines):
                        next_line = lines[j].strip()
                        
                        # Stop if we hit another date or empty line
                        if not next_line or re.match(r'^\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', next_line):
                            break
                        
                        # Look for amount patterns in this line
                        amount_pattern = r'([\d,]+\.[\d]{2})'
                        amount_matches = list(re.finditer(amount_pattern, next_line))
                        
                        if amount_matches:
                            # This looks like a transaction line
                            date_transactions.append((j, next_line, amount_matches))
                        
                        j += 1
                    
                    print(f"DEBUG: Found {len(date_transactions)} transactions for date {date}")
                    
                    # Process each transaction for this date
                    for trans_idx, (line_idx, trans_line, amount_matches) in enumerate(date_transactions):
                        print(f"DEBUG: Processing transaction {trans_idx + 1}: '{trans_line}'")
                        
                        # Parse the transaction line
                        if len(amount_matches) >= 2:
                            # Format: description amount balance
                            amount_str = amount_matches[0].group(1)
                            balance_str = amount_matches[1].group(1)
                            description = trans_line[:amount_matches[0].start()].strip()
                            
                            amount = float(amount_str.replace(',', ''))
                            balance = float(balance_str.replace(',', ''))
                            
                        elif len(amount_matches) == 1:
                            # Format: description amount (no balance)
                            amount_str = amount_matches[0].group(1)
                            description = trans_line[:amount_matches[0].start()].strip()
                            
                            amount = float(amount_str.replace(',', ''))
                            balance = running_balance + amount  # Estimate balance
                        else:
                            # No amount found, skip this line
                            continue
                        
                        # Determine if this is money out or money in based on description
                        is_money_out = any(indicator in description.lower() for indicator in [
                            'card payment', 'bill payment', 'direct debit', 'withdrawal', 'transfer out'
                        ])
                        
                        # Received From transactions are always money in
                        is_money_in = 'received from' in description.lower()
                        
                        # If it's money out, make the amount negative
                        if is_money_out and not is_money_in:
                            amount = -abs(amount)
                            print(f"DEBUG: Money out transaction: {amount}")
                        else:
                            print(f"DEBUG: Money in transaction: {amount}")
                        
                        # Look for additional details in subsequent lines (but limit search)
                        details = description
                        k = line_idx + 1
                        continuation_count = 0
                        while k < len(lines) and continuation_count < 2:  # Limit to 2 continuation lines
                            next_line = lines[k].strip()
                            # Stop if we hit another transaction line or date
                            if not next_line or re.match(r'^\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', next_line):
                                break
                            # Stop if we hit another transaction line (has amount)
                            if re.search(r'[\d,]+\.[\d]{2}', next_line):
                                break
                            # Skip reference lines and continuation indicators
                            if (not next_line.startswith('Ref:') and 
                                not next_line.startswith('On ') and 
                                next_line != 'May' and
                                not next_line.startswith('Continued')):
                                details += f" - {next_line}"
                                continuation_count += 1
                            k += 1
                        
                        transaction = {
                            'Transaction Date': date,
                            'Transaction Type': description,
                            'Amount': str(amount),
                            'Balance': str(balance),
                            'Details': details,
                        }
                        transactions.append(transaction)
                        running_balance = balance
                    
                    i = j  # Skip to the line after all transactions for this date
            else:
                i += 1
        
        print(f"DEBUG: Barclays processing summary:")
        print(f"- Total lines: {len(lines)}")
        print(f"- Transactions extracted: {len(transactions)}")
        print(f"- Format detected: {'Two-column (Money In/Money Out)' if two_column_format else 'Single-column'}")
        
        return transactions


class CSVParser(StatementParser):
    """CSV statement parser."""
    
    def __init__(self):
        pass
    
    def extract_text(self, file_path: str) -> List[str]:
        """For CSV, we don't extract text - we read directly."""
        return []
    
    def parse_transactions(self, lines: List[str], config: Dict) -> List[Dict]:
        """Parse transactions from CSV file."""
        # Get CSV configuration from statement_parsing section
        statement_parsing = config.get('statement_parsing', {})
        transaction_mapping = config.get('transaction_mapping', {})
        category_mapping = config.get('category_mapping', {})
        
        # Get file path from config (passed by the interpreter)
        file_path = config.get('file_path')
        if not file_path:
            print("Error: No file path provided for CSV parsing")
            return []
        
        # Read CSV file
        csv_headers = statement_parsing.get('csv_headers', None)
        skip_header = statement_parsing.get('csv_skip_header', False)

        # If headers are provided in config, use them as column names
        if csv_headers and len(csv_headers) > 0:
            df = pd.read_csv(
                file_path,
                delimiter=statement_parsing.get('csv_delimiter', ','),
                encoding=statement_parsing.get('csv_encoding', 'utf-8'),
                skip_blank_lines=True,
                header=None,  # No header in file
                names=csv_headers  # Use configured column names
            )
        else:
            # Use the file's header row
            header_param = 0 if not skip_header else None
            df = pd.read_csv(
                file_path,
                delimiter=statement_parsing.get('csv_delimiter', ','),
                encoding=statement_parsing.get('csv_encoding', 'utf-8'),
                skip_blank_lines=True,
                header=header_param
            )
        
        # Map columns
        date_field = transaction_mapping.get('date_field', 'Date')
        description_field = transaction_mapping.get('description_field', 'Memo')
        amount_field = transaction_mapping.get('amount_field', 'Amount')
        category_field = transaction_mapping.get('category_field', 'Subcategory')
        account_field = transaction_mapping.get('account_field', 'Account')
        reference_field = transaction_mapping.get('reference_field', 'Number')
        balance_field = transaction_mapping.get('balance_field', 'Balance')
        
        # Check if this CSV has a balance column (for balance calculation)
        has_balance_column = balance_field and balance_field in df.columns

        if has_balance_column:
            return self._parse_csv_with_balance(df, date_field, description_field, amount_field,
                                               category_field, account_field, reference_field,
                                               balance_field, category_mapping, config)
        else:
            return self._parse_standard_csv(df, date_field, description_field, amount_field,
                                          category_field, account_field, reference_field, category_mapping)
    
    def _parse_csv_with_balance(self, df, date_field, description_field, amount_field,
                                category_field, account_field, reference_field, balance_field, category_mapping, config):
        """Parse CSV with balance calculations (supports multiple banks)."""
        transactions = []

        # Try multiple date formats based on bank
        bank_name = config.get('bank_name', '').lower()

        # For Wells Fargo, use MM/DD/YYYY format only
        if bank_name == 'wellsfargo':
            date_formats = ['%m/%d/%Y']
        else:
            # For other banks, try DD/MM/YYYY first (Barclays), then MM/DD/YYYY
            date_formats = ['%d/%m/%Y', '%m/%d/%Y']

        df['parsed_date'] = None

        for date_format in date_formats:
            try:
                df['parsed_date'] = pd.to_datetime(df[date_field], format=date_format, errors='coerce')
                # If we got valid dates, break
                if df['parsed_date'].notna().any():
                    break
            except:
                continue

        df = df.sort_values('parsed_date', ascending=True)
        
        # Find opening balance (first non-empty balance value)
        opening_balance = None
        
        # Check if there's an unnamed column with the balance value
        unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
        
        for _, row in df.iterrows():
            # First try the balance field
            balance_val = row.get(balance_field)
            if balance_val and str(balance_val).strip() and str(balance_val).strip() != 'nan':
                try:
                    opening_balance = float(balance_val)
                    break
                except (ValueError, TypeError):
                    continue
            
            # If no balance in the balance field, check unnamed columns
            for unnamed_col in unnamed_cols:
                balance_val = row.get(unnamed_col)
                if balance_val and str(balance_val).strip() and str(balance_val).strip() != 'nan':
                    try:
                        opening_balance = float(balance_val)
                        print(f"Found opening balance in {unnamed_col}: {opening_balance}")
                        break
                    except (ValueError, TypeError):
                        continue
            
            if opening_balance is not None:
                break
        
        if opening_balance is None:
            print(f"Warning: No opening balance found in CSV for {config.get('bank_name', 'unknown bank')}")
            opening_balance = 0.0

        print(f"CSV Parser: Opening balance = {opening_balance} for {config.get('bank_name', 'unknown bank')}")
        
        # Calculate running balances
        running_balance = opening_balance
        
        for _, row in df.iterrows():
            # Skip rows with empty dates
            if pd.isna(row['parsed_date']):
                continue

            # Convert date to expected format - include year for full date parsing
            formatted_date = row['parsed_date'].strftime('%d %b %Y')

            # Get amount and convert to float
            amount_str = str(row.get(amount_field, 0))
            try:
                amount = float(amount_str)
            except ValueError:
                amount = 0.0

            # Check if there's an actual balance value in the balance column
            balance_from_csv = None
            balance_str = str(row.get(balance_field, ''))
            if balance_str and balance_str.strip() and balance_str.strip() not in ['', 'nan', '0', '0.0', '0.00']:
                try:
                    balance_from_csv = float(balance_str)
                except (ValueError, TypeError):
                    pass

            # If balance is provided in CSV, use it; otherwise calculate running balance
            if balance_from_csv is not None:
                running_balance = balance_from_csv
                print(f"[DEBUG] Using CSV balance: {running_balance} for transaction on {formatted_date}")
            else:
                # Calculate running balance from amount
                running_balance += amount

            transaction = {
                'Transaction Date': formatted_date,
                'Details': str(row.get(description_field, '')),
                'Amount': str(amount),
                'Transaction Type': str(row.get(category_field, '')),
                'Account Name': str(row.get(account_field, '')),
                'Reference #': str(row.get(reference_field, '')),
                'Balance': str(running_balance),
            }

            # Apply category mapping
            original_category = transaction['Transaction Type']
            if original_category in category_mapping:
                transaction['Transaction Type'] = category_mapping[original_category]

            transactions.append(transaction)
        
        return transactions
    
    def _parse_standard_csv(self, df, date_field, description_field, amount_field,
                           category_field, account_field, reference_field, category_mapping):
        """Parse standard CSV without balance calculations."""
        transactions = []

        for _, row in df.iterrows():
            # Convert date to a format the system can handle
            date_str = str(row.get(date_field, ''))
            if date_str and '/' in date_str:
                try:
                    # Try multiple date formats
                    from datetime import datetime
                    date_obj = None

                    # Try dd/MM/yyyy format (Barclays)
                    try:
                        date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                    except ValueError:
                        pass

                    # Try MM/d/yyyy or MM/dd/yyyy format (Wells Fargo)
                    if not date_obj:
                        try:
                            date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                        except ValueError:
                            pass

                    # Convert to a format that matches the expected pattern
                    # If date includes year (Wells Fargo), keep it; otherwise use "DD Mon" format
                    if date_obj:
                        # Check if the original date had a year (4 digits)
                        has_year = any(len(part) == 4 and part.isdigit() for part in date_str.split('/'))
                        if has_year:
                            # Keep full date with year (e.g., "05 Dec 2025")
                            formatted_date = date_obj.strftime('%d %b %Y')
                        else:
                            # Use day + month only (e.g., "30 May")
                            formatted_date = date_obj.strftime('%d %b')
                    else:
                        formatted_date = date_str
                except ValueError:
                    formatted_date = date_str
            else:
                formatted_date = date_str
            
            transaction = {
                'Transaction Date': formatted_date,
                'Details': str(row.get(description_field, '')),
                'Amount': str(row.get(amount_field, 0)),
                'Transaction Type': str(row.get(category_field, '')),
                'Account Name': str(row.get(account_field, '')),
                'Reference #': str(row.get(reference_field, '')),
                'Balance': '0',  # CSV typically doesn't have running balance
            }
            
            # Apply category mapping
            original_category = transaction['Transaction Type']
            if original_category in category_mapping:
                transaction['Transaction Type'] = category_mapping[original_category]
            
            transactions.append(transaction)
        
        return transactions


class ParserFactory:
    """Factory for creating appropriate statement parsers."""
    
    @staticmethod
    def create_parser(file_format: str) -> StatementParser:
        """Create a parser based on file format."""
        if file_format.lower() == 'pdf':
            return PDFParser()
        elif file_format.lower() == 'csv':
            return CSVParser()
        else:
            raise ValueError(f"Unsupported file format: {file_format}")


class StatementInterpreter(GeneralHelperFns):
    """General statement interpreter that can handle multiple file formats."""
    
    def process_raw_df(self):
        """Process bank statements using cached data where possible."""
        print("DEBUG: Starting to process bank statements...")
        
        if hasattr(self, 'filtered_df') and self.filtered_df is not None:
            print("DEBUG: Using cached processed data")
            return self.filtered_df
            
        print("DEBUG: Processing raw data...")
        print(f"DEBUG: Raw DataFrame size: {len(self.df_raw) if self.df_raw is not None else 'None'}")
        
        filtered_df = self.df_preprocessing(self.df_raw)
        print(f"DEBUG: After preprocessing: {len(filtered_df)} rows")
        
        filtered_df = self.recalibrate_amounts(filtered_df)
        print(f"DEBUG: After recalibrating amounts: {len(filtered_df)} rows")
        
        filtered_df = self.combine_balances_across_accounts(filtered_df)
        print(f"DEBUG: After combining balances: {len(filtered_df)} rows")
        
        # Detect and remove duplicates
        duplicate_mask = self.__detect_duplicates(filtered_df)
        duplicate_count = duplicate_mask.sum()
        if duplicate_count > 0:
            print(f"DEBUG: Found {duplicate_count} duplicate transactions")
            filtered_df = filtered_df[~duplicate_mask]
            print(f"DEBUG: After removing duplicates: {len(filtered_df)} rows")
        
        self.filtered_df = self.df_postprocessing(filtered_df)
        print(f"DEBUG: Final processed DataFrame size: {len(self.filtered_df)} rows")
        
        print("DEBUG: Bank statements processed successfully.")
        return self.filtered_df
    
    def generate_fin_df(self, account_types=None):
        """Generate financial DataFrame from statement files."""
        overall_df = pd.DataFrame()
        if account_types is None:
            account_types = self.read_all_account_type_folder_names()
        
        if not account_types:
            print("No account types found. Check if bank is properly configured.")
            return overall_df
        
        cached_files = 0
        processed_files = 0
        file_row_counts = {}
        
        for account_type in account_types:
            account_names = self.read_all_account_folder_names(account_type)
            for account_name in account_names:
                files = self.read_all_files(account_type, account_name)
                print(f"\nProcessing {len(files)} statements from {self.bank_name}/{account_type}/{account_name}")
                
                for file_name in files:
                    file_attrs = self.grab_pdf_name_attributes(file_name)

                    if self.base_path:
                        file_path = os.path.join(self.base_path, "bank_statements", self.bank_name, account_type, account_name, file_name)
                    else:
                        file_path = self.process_import_path(file_name, account_type, account_name)

                    # Check if file has already been processed
                    file_hash = self.__get_file_hash(file_path)

                    # Handle files without month/year in filename (e.g., Wells Fargo CSVs)
                    # Use 'Unknown' as defaults if not found
                    metadata = {
                        "year": file_attrs.get('year', 'Unknown'),
                        "month": file_attrs.get('month', 'Unknown'),
                        "account_type": account_type,
                        "account_name": account_name,
                        "file_name": file_name
                    }
                    
                    if self.__is_file_cached(file_hash, metadata):
                        # Load from cache
                        transactions, _ = self.__load_from_cache(file_hash, metadata)
                        cached_files += 1
                    else:
                        # Process the file
                        print(f"\nProcessing {file_name}")
                        transactions = self._process_statement_file(file_path, account_type)
                        # Save to cache
                        self.__save_to_cache(file_hash, transactions, metadata)
                        processed_files += 1
                    
                    temp_df = pd.DataFrame(transactions)
                    file_row_counts[file_name] = len(temp_df)
                    
                    # Add metadata columns
                    temp_df['Statement Year'] = metadata['year']
                    temp_df['Statement Month'] = metadata['month']
                    temp_df['Account Type'] = metadata['account_type']
                    temp_df['Account Name'] = metadata['account_name']
                    overall_df = pd.concat([temp_df, overall_df], ignore_index=True)

        print(f"\nFile processing summary:")
        print(f"- {cached_files} files loaded from cache")
        print(f"- {processed_files} files newly processed")
        print(f"- {len(overall_df)} total transactions found")
        
        # Report zero-transaction files if any
        if self.zero_transaction_files:
            print(f"- WARNING: {len(self.zero_transaction_files)} files had 0 transactions (check logs/zero_transactions.log)")
        
        # Process dates and sort
        if not overall_df.empty:
            overall_df = self._process_dates(overall_df)
            overall_df = overall_df.sort_values('DateTime', ascending=True).reset_index(drop=True)
        
        return overall_df
    
    def _process_statement_file(self, file_path: str, account_type: str) -> List[Dict]:
        """Process a single statement file using the appropriate parser."""
        # Get bank configuration
        bank_config = self._get_bank_config()
        file_format = bank_config.get('file_format', 'pdf')
        
        # Create parser
        parser = ParserFactory.create_parser(file_format)
        
        # Extract text (for PDF) or read directly (for CSV)
        if file_format.lower() == 'pdf':
            lines = parser.extract_text(file_path)
            # Add account type and bank name to config for pattern matching
            config = bank_config.copy()
            config['account_type'] = account_type
            config['bank_name'] = self.bank_name
            transactions = parser.parse_transactions(lines, config)
        else:
            # For CSV, pass the file path directly
            config = bank_config.copy()
            config['account_type'] = account_type
            config['bank_name'] = self.bank_name
            config['file_path'] = file_path  # Pass file path for CSV parsing
            transactions = parser.parse_transactions([], config)
        
        # Sanity check: Log PDFs with 0 transactions
        if len(transactions) == 0:
            self._log_zero_transactions_warning(file_path, account_type)
            self.zero_transaction_files.append((file_path, account_type))
        
        return transactions
    
    def _get_bank_config(self) -> Dict:
        """Get bank-specific configuration."""
        try:
            config_path = os.path.join(self.base_path, "src", "config", "banks", f"{self.bank_name}.json")
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading bank config: {str(e)}")
            return {}
    
    def _process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and standardize date columns."""
        try:
            # Check if Statement Year/Month are Unknown (for files without date in filename)
            has_unknown_dates = (df['Statement Year'] == 'Unknown').any() or (df['Statement Month'] == 'Unknown').any()

            if has_unknown_dates:
                # Transaction dates should already have year (from CSV parser)
                # Try to parse dates with year format first (e.g., "05 Dec 2025")
                df['DateTime'] = pd.to_datetime(df['Transaction Date'], format='%d %b %Y', errors='coerce')

                # If that fails, try without year (e.g., "05 Dec") and add current year
                if df['DateTime'].isna().all():
                    import datetime
                    current_year = datetime.datetime.now().year
                    df['DateTime'] = pd.to_datetime(df['Transaction Date'] + f' {current_year}', format='%d %b %Y', errors='coerce')
                return df

            df['Statement Year'] = pd.to_numeric(df['Statement Year'], errors='coerce')
            df['Statement Month'] = df['Statement Month'].astype(str)
            df['Transaction Date'] = df['Transaction Date'].astype(str)

            # Calculate Transaction Year
            month_str = df['Transaction Date'].str.split().str[0].str.lower()
            statement_month = df['Statement Month'].str.lower()
            is_prev_year = (month_str.isin(['nov', 'dec'])) & (statement_month == 'january')
            df['Transaction Year'] = df['Statement Year'].where(~is_prev_year, df['Statement Year'] - 1)

            # Create DateTime column
            df['DateTime'] = df['Transaction Date'] + ' ' + df['Transaction Year'].astype(str)
            # Handle abbreviated month names by converting them to full names
            df['DateTime'] = df['DateTime'].str.replace(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b',
                lambda x: {
                    'Jan': 'January', 'Feb': 'February', 'Mar': 'March', 'Apr': 'April',
                    'May': 'May', 'Jun': 'June', 'Jul': 'July', 'Aug': 'August',
                    'Sep': 'September', 'Oct': 'October', 'Nov': 'November', 'Dec': 'December'
                }[x.group(1)], regex=True)
            df['DateTime'] = pd.to_datetime(df['DateTime'])

        except Exception as e:
            print(f"Error processing dates: {str(e)}")
            # Fallback processing
            df['Transaction Year'] = df.apply(self.__calculate_transaction_year, axis=1)
            df['DateTime'] = df['Transaction Date'] + ' ' + df['Transaction Year'].astype(str)
            df['DateTime'] = pd.to_datetime(df['DateTime'])

        return df
    
    def __calculate_transaction_year(self, row):
        """Calculate the transaction year based on statement month and transaction date."""
        try:
            if 'Transaction Date' not in row or 'Statement Month' not in row or 'Statement Year' not in row:
                return None
                
            month_str = str(row['Transaction Date']).split()[0].lower()
            statement_month = str(row['Statement Month']).lower()
            statement_year = int(row['Statement Year'])
            
            if month_str in ['nov', 'dec'] and statement_month == 'january':
                return statement_year - 1
            return statement_year
        except Exception as e:
            print(f"Error calculating transaction year: {str(e)}")
            return row.get('Statement Year', None)
    
    def __get_file_hash(self, file_path: str) -> str:
        """Generate a hash for the file to use as cache identifier."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def __get_cache_path(self, file_hash: str, metadata: Dict) -> str:
        """Get the path to the cached data for a file."""
        cache_dir = paths.ensure_pdf_cache_exists()
        
        filename = f"{metadata['account_type']}_{metadata['account_name']}_{metadata['month']}_{metadata['year']}.json"
        filename = re.sub(r'[^\w\-_.]', '_', filename)
        return os.path.join(cache_dir, filename)
    
    def __is_file_cached(self, file_hash: str, metadata: Dict) -> bool:
        """Check if a file has already been processed and cached."""
        cache_path = self.__get_cache_path(file_hash, metadata)
        return os.path.exists(cache_path)
    
    def __save_to_cache(self, file_hash: str, transactions: List[Dict], metadata: Dict):
        """Save processed transactions to cache."""
        cache_data = {
            "metadata": {
                "account_type": metadata['account_type'],
                "account_name": metadata['account_name'],
                "statement_month": metadata['month'],
                "statement_year": metadata['year'],
                "file_name": metadata['file_name'],
                "cached_date": datetime.now().isoformat(),
                "file_hash": file_hash
            },
            "transactions": transactions
        }
        
        cache_path = self.__get_cache_path(file_hash, metadata)
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=4, sort_keys=True)
    
    def __load_from_cache(self, file_hash: str, metadata: Dict):
        """Load processed transactions from cache."""
        cache_path = self.__get_cache_path(file_hash, metadata)
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        return cache_data["transactions"], cache_data["metadata"]
    
    def get_cache_info(self):
        """Get information about cached files"""
        cache_dir = paths.ensure_pdf_cache_exists()
        if not os.path.exists(cache_dir):
            return {"cached_pdfs_count": 0, "cache_size_kb": 0}
        
        cached_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
        return {
            "cached_pdfs_count": len(cached_files),
            "cache_size_kb": sum(os.path.getsize(os.path.join(cache_dir, f)) for f in cached_files) // 1024
        }
    
    # Include all the existing processing methods from the original PDFReader
    # (df_preprocessing, recalibrate_amounts, combine_balances_across_accounts, etc.)
    # These methods remain the same as in the original pdf_interpreter.py
    
    def __init__(self, base_path=None, bank_name=None):
        if base_path is None:
            base_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        super().__init__(base_path=base_path, bank_name=bank_name)
        
        if not self.bank_name:
            raise ValueError("bank_name is required and cannot be None or empty")
        
        self.base_path = base_path
        self.databank_path = DATABANK_PATH
        self.uncategorized_path = UNCATEGORIZED_MERCHANTS_PATH
        
        # Create cached_data directory if it doesn't exist
        paths.ensure_cached_data_exists()
        self.cached_files_count = 0
        self.zero_transaction_files = []  # Track files with 0 transactions
        self.df_raw = self.generate_fin_df()
        
        # Initialize processor
        from src.modules.statement_processor import StatementProcessor
        self.processor = StatementProcessor(base_path, bank_name=self.bank_name)
    
    def df_preprocessing(self, df_in):
        """Preprocess the DataFrame by cleaning and converting data types."""
        return self.processor.preprocess_dataframe(df_in)
    
    def recalibrate_amounts(self, df_in):
        """Adjusts amounts based on account type."""
        return self.processor.recalibrate_amounts(df_in)
    
    def combine_balances_across_accounts(self, df_in):
        """Combines transactions from all accounts and calculates running balances."""
        return self.processor.combine_balances_across_accounts(df_in)
    
    def __detect_duplicates(self, df):
        """Detect duplicate transactions."""
        return self.processor.detect_duplicates(df)
    
    def df_postprocessing(self, df_in):
        """Post-process the DataFrame."""
        return self.processor.postprocess_dataframe(df_in)
    
    def _log_zero_transactions_warning(self, file_path: str, account_type: str):
        """Log warning when a PDF has 0 transactions - may indicate parsing issue."""
        import os
        from datetime import datetime
        
        file_name = os.path.basename(file_path)
        warning_msg = f"[SANITY CHECK] PDF with 0 transactions: {account_type}/{file_name}"
        
        # Log to console
        print(f"WARNING: {warning_msg}")
        
        # Log to file for review
        try:
            log_dir = os.path.join(self.base_path, "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, "zero_transactions.log")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {warning_msg} (Path: {file_path})\n")

        except Exception as e:
            print(f"Failed to write to zero transactions log: {e}")

    def clear_pdf_cache(self):
        """Clear all cached statement data"""
        import shutil
        import traceback

        try:
            print("\n[DEBUG] Starting cache clear operation...")

            # Get the cache directory path
            cache_dir = paths.pdf_cache
            print(f"[DEBUG] Cache directory path: {cache_dir}")
            print(f"[DEBUG] Cache directory exists: {os.path.exists(cache_dir)}")

            if os.path.exists(cache_dir):
                try:
                    # Get list of cached files before deleting
                    cached_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]

                    # Print summary of what will be deleted
                    print(f"\nClearing statement cache:")
                    print(f"Found {len(cached_files)} cached files:")
                    for file in cached_files:
                        try:
                            with open(os.path.join(cache_dir, file), 'r') as f:
                                cache_data = json.load(f)
                                metadata = cache_data.get('metadata', {})
                                print(f"- {file}: {metadata.get('account_type', 'Unknown')} statement for "
                                      f"{metadata.get('statement_month', 'Unknown')} {metadata.get('statement_year', 'Unknown')}")
                        except Exception as e:
                            print(f"- {file}: Could not read metadata ({str(e)})")

                    # Delete the cache directory
                    print(f"[DEBUG] Attempting to delete cache directory: {cache_dir}")
                    shutil.rmtree(cache_dir)
                    print(f"[DEBUG] Cache directory deleted successfully")
                except PermissionError as pe:
                    print(f"[ERROR] Permission denied when trying to delete cache: {str(pe)}")
                    print(f"[ERROR] Traceback: {traceback.format_exc()}")
                    raise
                except Exception as e:
                    print(f"[ERROR] Error during cache deletion: {str(e)}")
                    print(f"[ERROR] Traceback: {traceback.format_exc()}")
                    raise
            else:
                print(f"[DEBUG] Cache directory does not exist, nothing to clear")

            # Recreate the cache directory
            print(f"[DEBUG] Recreating cache directory...")
            os.makedirs(cache_dir, exist_ok=True)
            print(f"[DEBUG] Cache directory recreated")

            # Reset instance variables
            print(f"[DEBUG] Resetting instance variables...")
            self.cached_files_count = 0
            self.df_raw = None
            if hasattr(self, 'filtered_df'):
                self.filtered_df = None
            print(f"[DEBUG] Instance variables reset")

            print(f"\nStatement cache cleared successfully")
            return True
        except Exception as e:
            print(f"[ERROR] Error clearing statement cache: {str(e)}")
            print(f"[ERROR] Full traceback: {traceback.format_exc()}")
            return False 