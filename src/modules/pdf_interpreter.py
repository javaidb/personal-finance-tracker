import pdfplumber
import re
import pandas as pd
from tqdm import tqdm
import json
import os
import hashlib
from pathlib import Path
from datetime import datetime
import numpy as np

from src.modules.helper_fns import GeneralHelperFns
from src.modules.merchant_categorizer import MerchantCategorizer

# Try to import config, but don't fail if it doesn't exist
try:
    from src.config import config
except ImportError:
    print("Warning: Could not import config, using defaults")
    # Create a simple empty config
    import types
    config = types.SimpleNamespace()
    config.rent_ranges = []

from src.config.paths import (
    DATABANK_PATH, 
    UNCATEGORIZED_MERCHANTS_PATH, 
    DINING_KEYWORDS_PATH, 
    SHOPPING_KEYWORDS_PATH,
    MANUAL_CATEGORIES_PATH,
    CATEGORY_COLORS_PATH,
    paths
)

class PDFReader(GeneralHelperFns):

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
        self.cached_pdfs_count = 0  # Track number of cached PDFs
        self.df_raw = self.generate_fin_df()

    def process_raw_df(self):
        """Process bank statements using cached data where possible."""
        print("DEBUG: Starting to process bank statements...")
        
        # If we already have processed data, return it
        if hasattr(self, 'filtered_df') and self.filtered_df is not None:
            print("DEBUG: Using cached processed data")
            return self.filtered_df
            
        # Process the raw data
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

    def extract_features_from_pdf(self, pdf_file, x_tolerance=2):
        parent_dir = os.path.dirname(pdf_file)
        account_dir = os.path.basename(os.path.dirname(parent_dir))
        
        text = ''
        # Open the PDF file
        with pdfplumber.open(pdf_file) as pdf:
            total_pages = len(pdf.pages)
            print(f"Processing PDF: {os.path.basename(pdf_file)} ({total_pages} pages)")
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # First try to extract text without cropping
                    page_text = page.extract_text(x_tolerance=x_tolerance)
                    if page_text:
                        text += page_text + '\n'
                        print(f"DEBUG: Successfully extracted text from page {page_num} without cropping")
                        continue

                    print(f"DEBUG: No text extracted without cropping for page {page_num}, trying with cropping")
                    # If no text was extracted, try with cropping
                    page_width = float(page.width)
                    page_height = float(page.height)
                    
                    # Try different margin sizes, from smallest to largest
                    margin_sizes = [
                        (0.05, 0.05),  # 5% margins
                        (0.08, 0.08),  # 8% margins
                        (0.10, 0.10),  # 10% margins
                        (0.15, 0.15),  # 15% margins
                    ]
                    
                    text_extracted = False
                    for margin_x_pct, margin_y_pct in margin_sizes:
                        try:
                            margin_x = page_width * margin_x_pct
                            margin_y = page_height * margin_y_pct
                            
                            crop_box = (
                                margin_x,                    # left
                                margin_y,                    # top
                                page_width - margin_x,       # right
                                page_height - margin_y       # bottom
                            )
                            
                            cropped = page.crop(crop_box)
                            if cropped is not None:
                                page_text = cropped.extract_text(x_tolerance=x_tolerance)
                                if page_text:
                                    text += page_text + '\n'
                                    text_extracted = True
                                    print(f"DEBUG: Successfully extracted text from page {page_num} with {margin_x_pct*100}% margins")
                                    break
                        except Exception as e:
                            print(f"DEBUG: Failed to extract text with margins {margin_x_pct*100}%/{margin_y_pct*100}%: {str(e)}")
                            continue
                    
                    if not text_extracted:
                        print(f"DEBUG: Failed to extract any text from page {page_num} with any margin size")
                    
                except Exception as e:
                    print(f"DEBUG: Failed to process page {page_num}/{total_pages}: {str(e)}")
                    continue

        # Process the extracted text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        print(f"DEBUG: Extracted {len(lines)} non-empty lines from PDF")
        if len(lines) == 0:
            print("DEBUG: WARNING - No lines extracted from PDF!")
            print("DEBUG: Raw text length:", len(text))
            print("DEBUG: First 500 characters of raw text:", text[:500])
        return lines

    def __grab_pattern(self, account_type):
        """Get pattern based on bank and account type."""
        if self.bank_config is None:
            # Fallback to original hard-coded patterns if bank config is not available
            if account_type == "Credit":
                pattern = r'(\d{3})\s+(\w{3}\s+\d{1,2})\s+(\w{3}\s+\d{1,2})\s+(.+?)\s+(\d+\.\d{2})(-)?$'
                print(f"DEBUG: Using fallback Credit card pattern: {pattern}")
                return pattern
            elif account_type in ["Chequing", "Savings"]:
                pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{1,2})\s+([^$\d]+?)\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?))'
                print(f"DEBUG: Using fallback Chequing/Savings pattern: {pattern}")
                return pattern
            else:
                print(f"DEBUG: WARNING - Unknown account type: {account_type}")
                return None
        
        try:
            pattern_config = self.bank_config.get_bank_pattern(self.bank_name, account_type)
            pattern = pattern_config['pattern']
            print(f"DEBUG: Using {self.bank_name} {account_type} pattern: {pattern}")
            print(f"DEBUG: Example match: {pattern_config.get('example', 'No example provided')}")
            return pattern
        except Exception as e:
            print(f"Error getting pattern for {self.bank_name}/{account_type}: {str(e)}")
            # Fallback to original patterns
            return self.__grab_pattern_fallback(account_type)
    
    def __grab_pattern_fallback(self, account_type):
        """Fallback to original hard-coded patterns."""
        if account_type == "Credit":
            pattern = r'(\d{3})\s+(\w{3}\s+\d{1,2})\s+(\w{3}\s+\d{1,2})\s+(.+?)\s+(\d+\.\d{2})(-)?$'
            print(f"DEBUG: Using fallback Credit card pattern: {pattern}")
            return pattern
        elif account_type in ["Chequing", "Savings"]:
            pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{1,2})\s+([^$\d]+?)\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?))'
            print(f"DEBUG: Using fallback Chequing/Savings pattern: {pattern}")
            return pattern
        else:
            print(f"DEBUG: WARNING - Unknown account type: {account_type}")
            return None

    def process_transactions_from_lines(self, pdf_lines, account_type):
        transactions = []
        pattern = self.__grab_pattern(account_type)
        running_balance = 0.0  # Initialize running balance
        
        print(f"DEBUG: Found {len(pdf_lines)} lines to process")
        print(f"DEBUG: Using pattern for {account_type}: {pattern}")
        
        # First pass: find opening balance if it exists
        opening_balance = None
        for line in pdf_lines:
            if 'OPENING BALANCE' in line.upper() or 'START BALANCE' in line.upper():
                try:
                    # Try to extract the opening balance amount - look specifically for balance after "balance" keyword
                    balance_match = re.search(r'balance\s+([\d,]+\.\d{2})', line, re.IGNORECASE)
                    if balance_match:
                        opening_balance = float(balance_match.group(1).replace(',', ''))
                        running_balance = opening_balance
                        print(f"DEBUG: Found opening balance: {opening_balance}")
                        break
                except Exception as e:
                    print(f"DEBUG: Error extracting opening balance from line: {line}")
                    print(f"DEBUG: Error details: {str(e)}")
        
        # Special handling for Barclays multi-line transactions
        if self.bank_name == "barclays" and account_type in ["Chequing", "Savings"]:
            return self.__process_barclays_transactions(pdf_lines, account_type, pattern, running_balance)
        
        # Process transactions (original logic for other banks)
        skipped_lines = 0
        matched_lines = 0
        
        for i, line in enumerate(pdf_lines):
            match = re.match(pattern, line)
            if match:
                matched_lines += 1
                try:
                    match_groups = list(match.groups())  # Get the groups from the match object
                    if account_type == "Credit":
                        ref_num = match_groups[0]
                        transaction_date = match_groups[1]
                        post_date = match_groups[2]
                        details = match_groups[3].strip()
                        amount = float(match_groups[4])
                        
                        # Keep original sign from PDF
                        if match_groups[5] == "-":
                            amount = -amount
                            
                        # Update running balance
                        running_balance = running_balance + amount
                            
                        transaction = {
                            'Reference #': ref_num,
                            'Transaction Date': transaction_date,
                            'Post Date': post_date,
                            'Details': details,
                            'Amount': str(amount),
                            'Balance': str(running_balance),
                            'Transaction Type': '',
                        }
                    elif account_type in ["Chequing", "Savings"]:
                        date = match_groups[0]
                        transaction_type = match_groups[1].strip()
                        amount = float(match_groups[2].replace(',', ''))
                        balance = float(match_groups[3].replace(',', ''))
                        
                        # Get additional details from reference numbers or descriptions
                        details = transaction_type
                        if i + 1 < len(pdf_lines):
                            next_line = pdf_lines[i + 1].strip()
                            if not re.match(pattern, next_line) and next_line:
                                details = f"{transaction_type} - {next_line}"
                        
                        transaction = {
                            'Transaction Date': date,
                            'Transaction Type': transaction_type,
                            'Amount': str(amount),
                            'Balance': str(balance),
                            'Details': details,
                        }
                    
                    transactions.append(transaction)
                except Exception as e:
                    print(f"DEBUG: Error processing line: {line}")
                    print(f"DEBUG: Error details: {str(e)}")
                    continue
            else:
                # Print first few unmatched lines to help debug pattern issues
                if len(transactions) == 0 and i < 5:
                    print(f"DEBUG: Line did not match pattern: {line}")

        print(f"DEBUG: Processing summary for this PDF:")
        print(f"- Total lines: {len(pdf_lines)}")
        print(f"- Skipped lines: {skipped_lines}")
        print(f"- Lines matching pattern: {matched_lines}")
        print(f"- Transactions extracted: {len(transactions)}")
        if len(transactions) == 0:
            print("DEBUG: WARNING - No transactions extracted from this PDF!")
            print("DEBUG: First few lines of PDF content:")
            for i, line in enumerate(pdf_lines[:5]):
                print(f"DEBUG: Line {i+1}: {line}")

        return transactions

    def __process_barclays_transactions(self, pdf_lines, account_type, pattern, running_balance):
        """Special processor for Barclays multi-line transactions with Date Description Amount Balance format."""
        print(f"DEBUG: Starting Barclays transaction processing with pattern: {pattern}")
        transactions = []
        i = 0
        
        while i < len(pdf_lines):
            line = pdf_lines[i].strip()
            
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
                    while j < len(pdf_lines) and continuation_count < 3:  # Limit to 3 continuation lines
                        next_line = pdf_lines[j].strip()
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
                    
                    while j < len(pdf_lines):
                        next_line = pdf_lines[j].strip()
                        
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
                        while k < len(pdf_lines) and continuation_count < 2:  # Limit to 2 continuation lines
                            next_line = pdf_lines[k].strip()
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
        print(f"- Total lines: {len(pdf_lines)}")
        print(f"- Transactions extracted: {len(transactions)}")
        
        return transactions

    def __calculate_transaction_year(self, row):
        """Calculate the transaction year based on statement month and transaction date."""
        try:
            # Check if required columns exist
            if 'Transaction Date' not in row or 'Statement Month' not in row or 'Statement Year' not in row:
                print(f"Missing required columns in row: {row.keys()}")
                return None
                
            month_str = str(row['Transaction Date']).split()[0].lower()
            statement_month = str(row['Statement Month']).lower()
            statement_year = int(row['Statement Year'])
            
            # If transaction is in Nov/Dec but statement is January, it's from previous year
            if month_str in ['nov', 'dec'] and statement_month == 'january':
                return statement_year - 1
            return statement_year
        except Exception as e:
            print(f"Error calculating transaction year: {str(e)}")
            return row.get('Statement Year', None)  # fallback to statement year

    def __get_pdf_hash(self, pdf_file_path):
        """Generate a hash for the PDF file to use as cache identifier"""
        with open(pdf_file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
            
    def __get_cache_path(self, pdf_hash, metadata=None):
        """Get the path to the cached data for a PDF"""
        cache_dir = paths.ensure_pdf_cache_exists()
        
        if metadata:
            # Create a descriptive filename using metadata
            filename = f"{metadata['account_type']}_{metadata['account_name']}_{metadata['month']}_{metadata['year']}.json"
            # Replace any spaces or special characters in the filename
            filename = re.sub(r'[^\w\-_.]', '_', filename)
            return os.path.join(cache_dir, filename)
        else:
            # Fallback to hash if no metadata
            return os.path.join(cache_dir, f"{pdf_hash}.json")
            
    def __is_pdf_cached(self, pdf_hash, metadata=None):
        """Check if a PDF has already been processed and cached"""
        cache_path = self.__get_cache_path(pdf_hash, metadata)
        return os.path.exists(cache_path)
        
    def __save_to_cache(self, pdf_hash, transactions, metadata):
        """Save processed transactions to cache with pretty formatting"""
        # Convert transactions to DataFrame to check for duplicates
        df = pd.DataFrame(transactions)
        if not df.empty and 'DateTime' in df.columns and 'account_balance' in df.columns and 'Amount' in df.columns:
            # Detect duplicates
            duplicate_mask = self.__detect_duplicates(df)
            if duplicate_mask.any():
                print(f"DEBUG: Removing {duplicate_mask.sum()} duplicate transactions before caching")
                # Keep only non-duplicate transactions
                df = df[~duplicate_mask]
                transactions = df.to_dict('records')
        
        cache_data = {
            "metadata": {
                "account_type": metadata['account_type'],
                "account_name": metadata['account_name'],
                "statement_month": metadata['month'],
                "statement_year": metadata['year'],
                "pdf_file": metadata['pdf_file'],
                "cached_date": datetime.now().isoformat(),
                "pdf_hash": pdf_hash
            },
            "transactions": transactions
        }
        
        cache_path = self.__get_cache_path(pdf_hash, metadata)
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=4, sort_keys=True)
            
    def __load_from_cache(self, pdf_hash, metadata=None):
        """Load processed transactions from cache"""
        cache_path = self.__get_cache_path(pdf_hash, metadata)
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        return cache_data["transactions"], cache_data["metadata"]
    
    def clear_pdf_cache(self):
        """Clear all cached PDF data"""
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
                    print(f"\nClearing PDF cache:")
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
            self.cached_pdfs_count = 0
            self.df_raw = None
            if hasattr(self, 'filtered_df'):
                self.filtered_df = None
            print(f"[DEBUG] Instance variables reset")

            print(f"\nPDF cache cleared successfully")
            return True
        except Exception as e:
            print(f"[ERROR] Error clearing PDF cache: {str(e)}")
            print(f"[ERROR] Full traceback: {traceback.format_exc()}")
            return False

    def get_cache_info(self):
        """Get information about cached PDFs"""
        cache_dir = paths.pdf_cache
        if not os.path.exists(cache_dir):
            return {"cached_pdfs_count": 0, "cache_size_kb": 0}
        
        cached_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
        return {
            "cached_pdfs_count": len(cached_files),
            "cache_size_kb": sum(os.path.getsize(os.path.join(cache_dir, f)) for f in cached_files) // 1024
        }

    def generate_fin_df(self, account_types=None):
        overall_df = pd.DataFrame()
        if account_types is None:
            account_types = self.read_all_account_type_folder_names()
        
        if not account_types:
            print("No account types found. Check if bank is properly configured.")
            return overall_df
        
        cached_pdfs = 0
        processed_pdfs = 0
        pdf_row_counts = {}  # Track rows per PDF
        
        for account_type in account_types:
            account_names = self.read_all_account_folder_names(account_type)
            for account_name in account_names:
                pdf_files = self.read_all_files(account_type, account_name)
                print(f"\nProcessing {len(pdf_files)} statements from {self.bank_name}/{account_type}/{account_name}")
                
                for pdf_file in tqdm(pdf_files, desc=f"Reading PDFs from '{account_name}'"):
                    pdf_file_atts = self.grab_pdf_name_attributes(pdf_file)
                    # Use base_path if provided
                    if self.base_path:
                        if not self.bank_name:
                            raise ValueError("bank_name is required and cannot be None or empty")
                        pdf_file_path = os.path.join(self.base_path, "bank_statements", self.bank_name, account_type, account_name, pdf_file)
                    else:
                        pdf_file_path = self.process_import_path(pdf_file, account_type, account_name)
                    
                    # Check if the PDF has already been processed
                    pdf_hash = self.__get_pdf_hash(pdf_file_path)
                    metadata = {
                        "year": pdf_file_atts['year'],
                        "month": pdf_file_atts['month'],
                        "account_type": account_type,
                        "account_name": account_name,
                        "pdf_file": pdf_file
                    }
                    
                    if self.__is_pdf_cached(pdf_hash, metadata):
                        # Load from cache if available
                        transactions, _ = self.__load_from_cache(pdf_hash, metadata)
                        cached_pdfs += 1
                    else:
                        # Process the PDF if not cached
                        print(f"\nProcessing {pdf_file}")
                        lines = self.extract_features_from_pdf(pdf_file_path, x_tolerance=2)
                        transactions = self.process_transactions_from_lines(lines, account_type)
                        # Save to cache for future use
                        self.__save_to_cache(pdf_hash, transactions, metadata)
                        processed_pdfs += 1
                    
                    temp_df = pd.DataFrame(transactions)
                    pdf_row_counts[pdf_file] = len(temp_df)  # Track row count for this PDF
                    
                    # Add metadata columns one by one
                    temp_df['Statement Year'] = metadata['year']
                    temp_df['Statement Month'] = metadata['month']
                    temp_df['Account Type'] = metadata['account_type']
                    temp_df['Account Name'] = metadata['account_name']
                    overall_df = pd.concat([temp_df, overall_df], ignore_index=True)

        print(f"\nPDF processing summary:")
        print(f"- {cached_pdfs} PDFs loaded from cache")
        print(f"- {processed_pdfs} PDFs newly processed")
        print(f"- {len(overall_df)} total transactions found")
        print("\nRows extracted per PDF:")
        for pdf_file, row_count in sorted(pdf_row_counts.items()):
            print(f"- {pdf_file}: {row_count} rows")
        
        # Check if DataFrame is empty
        if overall_df.empty:
            print("No transactions found - returning empty DataFrame")
            return overall_df
            
        # Calculate Transaction Year and DateTime
        try:
            # First ensure all required columns exist and are of the right type
            overall_df['Statement Year'] = pd.to_numeric(overall_df['Statement Year'], errors='coerce')
            overall_df['Statement Month'] = overall_df['Statement Month'].astype(str)
            overall_df['Transaction Date'] = overall_df['Transaction Date'].astype(str)
            
            # Special handling for Barclays date format (DD MMM)
            if self.bank_name == "barclays":
                # For Barclays, Transaction Date is already in DD MMM format
                # We need to add the year to make it a complete date
                overall_df['Transaction Year'] = overall_df['Statement Year']
                
                # Create DateTime column with proper format for Barclays
                overall_df['DateTime'] = overall_df['Transaction Date'] + ' ' + overall_df['Transaction Year'].astype(str)
                # Use a specific format for parsing DD MMM YYYY
                overall_df['DateTime'] = pd.to_datetime(overall_df['DateTime'], format='%d %b %Y', errors='coerce')
            else:
                # Original logic for other banks
                # Calculate Transaction Year using vectorized operations
                month_str = overall_df['Transaction Date'].str.split().str[0].str.lower()
                statement_month = overall_df['Statement Month'].str.lower()
                is_prev_year = (month_str.isin(['nov', 'dec'])) & (statement_month == 'january')
                overall_df['Transaction Year'] = overall_df['Statement Year'].where(~is_prev_year, overall_df['Statement Year'] - 1)
                
                # Create DateTime column
                overall_df['DateTime'] = overall_df['Transaction Date'] + ' ' + overall_df['Transaction Year'].astype(str)
                overall_df['DateTime'] = pd.to_datetime(overall_df['DateTime'])
        except Exception as e:
            print(f"Error processing dates: {str(e)}")
            if overall_df.empty:
                print("DataFrame is empty - skipping date processing")
                return overall_df
            print("Falling back to row-by-row processing...")
            # Fallback to row-by-row processing if vectorized operations fail
            overall_df['Transaction Year'] = overall_df.apply(self.__calculate_transaction_year, axis=1)
            overall_df['DateTime'] = overall_df['Transaction Date'] + ' ' + overall_df['Transaction Year'].astype(str)
            overall_df['DateTime'] = pd.to_datetime(overall_df['DateTime'])
        
        # Sort transactions by DateTime to ensure chronological order
        if not overall_df.empty and 'DateTime' in overall_df.columns:
            overall_df = overall_df.sort_values('DateTime', ascending=True).reset_index(drop=True)
            
        return overall_df

    def df_preprocessing(self, df_in):
        """Preprocess the DataFrame by cleaning and converting data types."""
        print(f"Pre-processing bank statements.")

        df = df_in.copy()
        
        # Process transaction details first
        df = self.__process_transaction_details(df)
        
        # Convert numeric columns
        for col in ['Balance', 'Amount']:
            if col in df.columns:
                # First remove any commas
                df[col] = df[col].str.replace(',', '', regex=False)
                # Then convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill NaN values with 0
                df[col] = df[col].fillna(0)
        
        # Classify transactions
        df = self.__classify_transactions(df)
        
        # Categorize dining transactions
        df = self.__categorize_food_transactions(df)
        
        # Categorize shopping transactions
        df = self.__categorize_shopping_transactions(df)
        
        return df

    def __process_transaction_details(self, df):
        def process_row(details_row, transaction_type_row):
            concat_row = details_row + " " + transaction_type_row
            def process_string(s):
                return re.sub(r'[^a-z\s&]', '', s.lower())
            
            # Split the string by space and process each element
            return [process_string(elem) for elem in concat_row.split() if process_string(elem)]

        # Apply the function to a DataFrame column
        df['Processed Details'] = df.apply(lambda row: process_row(row['Details'], row['Transaction Type']), axis=1)

        def filter_df(df, column, substrings):
            def check_row(row):
                for item in substrings:
                    if isinstance(item, tuple):
                        if all(any(sub.lower() in s.lower() for s in row) for sub in item):
                            return True
                    elif any(item.lower() in s.lower() for s in row):
                        return True
                return False
            
            return df[~df[column].apply(check_row)]
        
        substrings = ['MB-', ('payment', 'from')]
        df = filter_df(df, 'Processed Details', substrings)

        return df

    def __classify_transactions(self, df):
        """
        Classifies transactions using merchant matching first, then falling back to pattern matching.
        
        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with 'Classification' and 'Matched Keyword' columns.
        """
        print(f"Classifying transactions using merchant categorization with pattern matching fallback...")
        
        # Initialize merchant categorizer
        merchant_categorizer = MerchantCategorizer(self.base_path)
        
        # Import patterns for fallback classification
        try:
            if os.path.exists(self.databank_path):
                with open(self.databank_path, 'r') as f:
                    databank = json.load(f)
                    imported_json = databank.get('categories', {})
            else:
                imported_json = {}
        except Exception as e:
            print(f"Error loading databank.json: {str(e)}")
            imported_json = {}
        
        # Stats for reporting
        merchant_matches = 0
        pattern_matches = 0
        uncategorized = 0
        
        # For collecting training data
        uncategorized_merchants = {}

        def categorize_strings(row, categories=imported_json):
            nonlocal merchant_matches, pattern_matches, uncategorized, uncategorized_merchants
            
            processed_details = row['Processed Details']
            
            # Try merchant-based categorization first
            category, merchant = merchant_categorizer.categorize_transaction(processed_details)
            
            # If we got a match from the merchant categorizer, use that
            if category != "uncategorized":
                merchant_matches += 1
                return pd.Series([category, f"Merchant: {merchant}"])
            
            # Fall back to pattern matching
            s = ' '.join(processed_details)
            s_lower = s.lower()
            found_categories = []
            matched_keywords = []
            pattern_indices = []
            
            for category, category_data in categories.items():
                patterns = category_data.get('patterns', [])
                for pattern_ind, pattern in enumerate(patterns):
                    terms = pattern.get('terms', [])
                    if all(term.lower() in s_lower for term in terms):
                        found_categories.append(category)
                        matched_keywords.append(' '.join(terms))
                        pattern_indices.append(pattern_ind)
                        # Increment match tally
                        patterns[pattern_ind]['matchCount'] = patterns[pattern_ind].get('matchCount', 0) + 1
                        category_data['totalMatches'] = category_data.get('totalMatches', 0) + 1

            # Now assign category/associated match keywords
            if not found_categories:
                found_category = "uncategorized"
                matched_keyword = None
                uncategorized += 1
                
                # Store uncategorized merchants for later review
                if merchant != "Unknown":
                    # Extract amount from the row
                    amount = abs(row['Amount']) if isinstance(row['Amount'], (int, float)) else 0
                    # Group similar merchants
                    if merchant in uncategorized_merchants:
                        uncategorized_merchants[merchant]["count"] += 1
                        uncategorized_merchants[merchant]["total_amount"] += amount
                        if len(uncategorized_merchants[merchant]["examples"]) < 5:  # Store up to 5 examples
                            uncategorized_merchants[merchant]["examples"].append(s)
                    else:
                        uncategorized_merchants[merchant] = {
                            "count": 1,
                            "total_amount": amount,
                            "examples": [s]
                        }
            else:
                # If multiple matches found, grab the one with most strings matched
                if len(matched_keywords) > 1:
                    index = max(range(len(matched_keywords)), key=lambda i: len(matched_keywords[i].split()))
                else:
                    index = 0
                
                found_category = found_categories[index]
                matched_keyword = matched_keywords[index]
                pattern_matches += 1
                
                # Auto-learn this merchant-category mapping
                if found_category != "uncategorized":
                    merchant_categorizer.auto_learn(processed_details, found_category)
            
            # Save updated databank
            try:
                with open(self.databank_path, 'w') as f:
                    json.dump({"categories": categories}, f, indent=2)
            except Exception as e:
                print(f"Error saving databank.json: {str(e)}")
            
            return pd.Series([found_category, matched_keyword])

        df[['Classification', 'Matched Keyword']] = df.apply(categorize_strings, axis=1)
        
        # Save uncategorized merchants to a review file
        if uncategorized > 0:
            self.__save_uncategorized_merchants(uncategorized_merchants)
        
        # Report classification stats
        total = len(df)
        print(f"Classification complete: {merchant_matches} by merchant ({merchant_matches/total:.1%}), "
              f"{pattern_matches} by pattern ({pattern_matches/total:.1%}), "
              f"{uncategorized} uncategorized ({uncategorized/total:.1%})")
        
        return df
        
    def __save_uncategorized_merchants(self, merchants):
        """Save uncategorized merchants to a file for later review."""
        try:
            # Load existing uncategorized merchants
            existing_merchants = {}
            if os.path.exists(self.uncategorized_path):
                try:
                    with open(self.uncategorized_path, 'r') as f:
                        content = f.read().strip()
                        if content:
                            existing_merchants = json.loads(content)
                except Exception as e:
                    print(f"Error loading existing uncategorized merchants file: {str(e)}")

            # Merge with new merchants
            for merchant, data in merchants.items():
                if merchant in existing_merchants:
                    existing_merchants[merchant]["count"] += data["count"]
                    existing_merchants[merchant]["total_amount"] += data["total_amount"]
                    # Add new examples if we have space
                    existing_examples = existing_merchants[merchant]["examples"]
                    for example in data["examples"]:
                        if example not in existing_examples and len(existing_examples) < 5:
                            existing_examples.append(example)
                else:
                    existing_merchants[merchant] = data

            # Save merged data
            with open(self.uncategorized_path, 'w') as f:
                json.dump(existing_merchants, f, indent=2)
            print(f"Saved {len(existing_merchants)} uncategorized merchants to {self.uncategorized_path}")
        except Exception as e:
            print(f"Error saving uncategorized merchants: {str(e)}")

    def recalibrate_amounts(self, df_in):
        """
        Adjusts amounts based on account type:
        - For Chequing/Savings: Use balance changes to determine amount sign
        - For Credit: Force all amounts to be negative
        """
        print(f"Recalibrating amounts in bank statements.")

        df = df_in.copy()
        for account_type in df['Account Type'].unique():
            mask = df['Account Type'] == account_type
            account_df = df[mask].copy()
            
            if account_type == 'Credit':
                # For credit accounts, ensure all amounts are negative
                df.loc[mask, 'Amount'] = -abs(df.loc[mask, 'Amount'])
            elif account_type in ['Chequing', 'Savings']:
                # Add a temporary column for the index
                account_df['__orig_index'] = account_df.index
                account_df = account_df.sort_values(['DateTime', '__orig_index'])
                # Calculate balance differences using Balance (not account_balance)
                account_df['balance_diff'] = account_df['Balance'].diff()
                # Drop the temporary column
                account_df = account_df.drop(columns=['__orig_index'])

                # Debug print for the date range in question
                debug_rows = account_df[(account_df['DateTime'] >= '2025-01-15') & (account_df['DateTime'] <= '2025-01-17')]
                if not debug_rows.empty:
                    print('\nDEBUG: recalibrate_amounts rows for Chequing/Ultimate Package 2025-01-15 to 2025-01-17:')
                    print(debug_rows[['DateTime', 'Account Name', 'Balance', 'Amount', 'balance_diff']])
                
                # Update amount signs based on balance differences
                account_df['Amount'] = account_df.apply(
                    lambda row: (
                        # If balance decreased, amount should be negative
                        -abs(row['Amount']) if row['balance_diff'] is not None and row['balance_diff'] < 0
                        # If balance increased, amount should be positive
                        else abs(row['Amount']) if row['balance_diff'] is not None and row['balance_diff'] > 0
                        # If no balance difference (first row), keep original sign
                        else row['Amount']
                    ),
                    axis=1
                )
                
                # Drop temporary column
                account_df.drop(columns=['balance_diff'], inplace=True)
                
                # Update the main dataframe
                df.loc[mask] = account_df
            
        return df

    def combine_balances_across_accounts(self, df_in):
        """
        Combines transactions from all accounts and calculates running balances.
        For fixed accounts (Chequing/Savings), uses the sum of their balances.
        For credit accounts, sets account_balance to 0 and calculates running balance based on the last fixed balance point.
        """
        df = df_in.copy()
        
        # Sort transactions by DateTime, with specific account type ordering
        def account_type_order(x):
            order = {'Chequing': 0, 'Savings': 1, 'Credit': 2}
            return order.get(x, 3)
            
        df['account_order'] = df['Account Type'].apply(account_type_order)
        df = df.sort_values(['DateTime', 'account_order']).drop('account_order', axis=1)
                        
        # Rename original Balance column to account_balance and ensure it's numeric
        df = df.rename(columns={'Balance': 'account_balance'})
        df['account_balance'] = pd.to_numeric(df['account_balance'].astype(str).str.replace(',', ''), errors='coerce')
        
        # Set credit account balances to 0
        df.loc[df['Account Type'] == 'Credit', 'account_balance'] = 0
        
        # Initialize running_balance column
        df['running_balance'] = None
        
        # First pass: calculate fixed account balances
        fixed_balances = {}  # {date: total_balance}
        for idx in df.index:
            current_date = df.loc[idx, 'DateTime']
            account_type = df.loc[idx, 'Account Type']
            
            if account_type in ['Chequing', 'Savings']:
                # Get all fixed account balances up to this point
                fixed_mask = (
                    (df['DateTime'] <= current_date) & 
                    df['Account Type'].isin(['Chequing', 'Savings'])
                )
                fixed_df = df[fixed_mask]
                latest_fixed_balances = fixed_df.groupby('Account Name')['account_balance'].last()
                total_fixed_balance = latest_fixed_balances.sum()
                fixed_balances[current_date] = float(total_fixed_balance)
        
        # Second pass: calculate running balances
        last_fixed_balance = None
        last_fixed_date = None
        credit_sum = 0
        
        for idx in df.index:
            current_date = df.loc[idx, 'DateTime']
            account_type = df.loc[idx, 'Account Type']
            
            if account_type in ['Chequing', 'Savings']:
                # Use the pre-calculated fixed balance
                last_fixed_balance = fixed_balances[current_date]
                last_fixed_date = current_date
                credit_sum = 0
                df.loc[idx, 'running_balance'] = float(last_fixed_balance)
            else:  # Credit accounts
                if last_fixed_balance is None:
                    # If no fixed balance seen yet, look ahead for the next fixed balance
                    future_dates = [d for d in fixed_balances.keys() if d > current_date]
                    if future_dates:
                        next_fixed_date = min(future_dates)
                        last_fixed_balance = fixed_balances[next_fixed_date]
                        last_fixed_date = next_fixed_date
                        credit_sum = 0
                    else:
                        # No future fixed balances, use the last known fixed balance
                        last_fixed_date = max(fixed_balances.keys())
                        last_fixed_balance = fixed_balances[last_fixed_date]
                        credit_sum = 0
                
                # Get all credit transactions since the last fixed balance
                credit_mask = (
                    (df['DateTime'] > last_fixed_date) &
                    (df['DateTime'] <= current_date) &
                    (df['Account Type'] == 'Credit')
                )
                credit_sum = float(df[credit_mask]['Amount'].sum())
                
                # Calculate running balance
                df.loc[idx, 'running_balance'] = float(last_fixed_balance + credit_sum)
        
        # Ensure both balance columns are numeric
        df['account_balance'] = pd.to_numeric(df['account_balance'], errors='coerce')
        df['running_balance'] = pd.to_numeric(df['running_balance'], errors='coerce')
        
        print(f"DEBUG: After combining balances: {len(df)} rows")
        print(f"DEBUG: Sample account_balance values: {df['account_balance'].head()}")
        print(f"DEBUG: Sample running_balance values: {df['running_balance'].head()}")
        
        return df

    def tabulate_gap_balances(self, df_in):
        """Fill in missing balances by calculating from previous balance and current amount."""
        df = df_in.copy()
        
        # Process each account type and name separately
        for account_type in df['Account Type'].unique():
            account_df = df[df['Account Type'] == account_type]
            
            for account_name in account_df['Account Name'].unique():
                # Get subset for this account
                mask = (df['Account Type'] == account_type) & (df['Account Name'] == account_name)
                account_subset = df[mask].sort_values('DateTime').copy()
                
                # Find first valid balance
                first_valid_balance = account_subset['Balance'].first_valid_index()
                if first_valid_balance is None:
                    # If no valid balance exists, start from 0
                    account_subset.loc[account_subset.index[0], 'Balance'] = 0.0
                
                # Forward fill missing balances
                for i in range(1, len(account_subset)):
                    curr_idx = account_subset.index[i]
                    prev_idx = account_subset.index[i-1]
                    
                    if pd.isna(account_subset.loc[curr_idx, 'Balance']):
                        prev_balance = account_subset.loc[prev_idx, 'Balance']
                        curr_amount = account_subset.loc[curr_idx, 'Amount']
                        
                        if not pd.isna(prev_balance) and not pd.isna(curr_amount):
                            # Calculate new balance based on previous balance and current amount
                            new_balance = prev_balance + curr_amount
                            account_subset.loc[curr_idx, 'Balance'] = new_balance
                
                # Update the main dataframe
                df.loc[mask, 'Balance'] = account_subset['Balance']
        
        # Final forward fill for any remaining NaN values
        df['Balance'] = df.groupby(['Account Type', 'Account Name'])['Balance'].transform('ffill')
        
        return df

    def apply_manual_categories(self, df):
        """
        Apply manual categories from manual_categories.json to the DataFrame.
        For each entry:
          - If index is not null, try to match by datetime and amount first
          - If no match found by datetime/amount, fall back to index if provided
          - If index is null, find row(s) matching datetime and amount
        """
        import json
        import os
        manual_path = MANUAL_CATEGORIES_PATH
        if not os.path.exists(manual_path):
            return df
        try:
            with open(manual_path, 'r') as f:
                manual_cats = json.load(f)
        except Exception as e:
            print(f"Error loading manual_categories.json: {e}")
            return df
        if not isinstance(manual_cats, list):
            print("manual_categories.json is not a list; skipping manual category application.")
            return df

        # Convert DateTime column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['DateTime']):
            df['DateTime'] = pd.to_datetime(df['DateTime'])

        for entry in manual_cats:
            dt = entry.get('datetime')
            amt = entry.get('amount')
            cat = entry.get('category')
            idx = entry.get('index')

            # Convert entry datetime to datetime object
            entry_dt = pd.to_datetime(dt)

            # Try to match by datetime and amount first
            mask = (df['DateTime'] == entry_dt) & (df['Amount'] == amt)
            
            if mask.any():
                # If we found matches by datetime and amount, use those
                df.loc[mask, 'Classification'] = cat
            elif idx is not None and idx in df.index:
                # Fall back to index matching if datetime/amount match failed
                df.at[idx, 'Classification'] = cat
            else:
                # If no match found at all, print a warning
                print(f"Warning: Could not find match for manual category entry: {entry}")

        return df

    def df_postprocessing(self, df_in):
        print(f"Post-processing bank statements.")

        df = df_in.copy()

        try:
            from src.config import config
            df = self.__identify_rent_payments(df, config.rent_ranges)
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not load rent ranges from config: {e}")
            # Use default empty list if config not available
            df = self.__identify_rent_payments(df, [])
            
        df = self.__apply_custom_conditions(df)

        # Modified exclusion list to only exclude balance entries
        substring_exclusion_list = ['opening balance', 'closing balance']
        fullstring_exclusion_list = []
        
        df['details_str'] = df['Processed Details'].apply(lambda x: ' '.join(x).lower())

        def exclude_rows(details_str, substring_exclusion_list):
            for excl in substring_exclusion_list:
                # Check for exact match (space-separated or concatenated without spaces)
                if excl in details_str or excl.replace(' ', '') in details_str:
                    return False
            for excl in fullstring_exclusion_list:
                if details_str == excl:
                    return False
            return True

        df_filtered = df[df['details_str'].apply(lambda x: exclude_rows(x, substring_exclusion_list))]
        df_filtered = df_filtered.drop(columns=['details_str'])
        
        # Apply manual categories at the end
        df_filtered = self.apply_manual_categories(df_filtered)
        
        # After all processing is done, filter out transfer transactions
        def is_transfer(row):
            details_lower = str(row['Details']).lower()
            transaction_type_lower = str(row['Transaction Type']).lower()
            
            # Check for various types of transfers
            transfer_indicators = [
                'mb-transfer',
                'transfer to',
                'transfer from',
                'mb-credit card/loc pay',
                'credit card payment',
                'from -',  # New pattern for transfers starting with "FROM -"
                'mb credit card/loc pay. from',  # New pattern for credit card payments
                'mb credit card/loc pay from',  # Alternative format
                'mb credit card payment from',  # Another alternative format
                'mb credit card payment to',  # Outgoing credit card payment
                'mb credit card/loc pay to'  # Another outgoing format
            ]
            
            return any(indicator in details_lower for indicator in transfer_indicators)
        
        # Create a non-transfer version of the DataFrame
        df_no_transfers = df_filtered[~df_filtered.apply(is_transfer, axis=1)]
        
        # Store both versions in the object
        self.df_with_transfers = self.sort_df(df_filtered)  # Keep full version for internal use
        
        # Calculate running_balance_plus_investments
        df_no_transfers = self.sort_df(df_no_transfers)
        
        # Create a copy of running balance
        df_no_transfers['running_balance_plus_investments'] = df_no_transfers['running_balance'].copy()
        
        # Identify investment transactions
        investment_mask = df_no_transfers['Classification'].str.lower().isin(['investment', 'investments'])
        
        # Add back investment amounts to create new running balance
        if investment_mask.any():
            # Sort by DateTime to ensure correct running balance calculation
            df_no_transfers = df_no_transfers.sort_values('DateTime')
            
            # Calculate the difference between running balances
            df_no_transfers['balance_diff'] = df_no_transfers['running_balance'].diff()
            
            # Initialize the investment adjustment
            investment_adjustment = 0
            
            # Process each row in chronological order
            for idx in df_no_transfers.index:
                # If this is an investment transaction, update the adjustment
                if investment_mask[idx]:
                    # Subtract the investment amount to reverse its effect
                    investment_adjustment -= df_no_transfers.loc[idx, 'Amount']
                
                # Apply the current investment adjustment to this row's running balance
                df_no_transfers.loc[idx, 'running_balance_plus_investments'] = (
                    df_no_transfers.loc[idx, 'running_balance'] + investment_adjustment
                )
            
            # Drop the temporary column
            df_no_transfers = df_no_transfers.drop(columns=['balance_diff'])
        
        return df_no_transfers  # Return version without transfers

    def __identify_rent_payments(self, df_in, rent_ranges):
        """
        Identifies rent payments by looking for common transactions that occur around month transitions.
        Treats the last 4 days of a month and the first 4 days of the next month as a single 8-day period.
        Only one transaction per such period is allowed, and transactions are compared with adjacent periods for similarity.
        """
        df = df_in.copy()
        df = df.sort_values('DateTime')

        # Constants
        MIN_RENT_AMOUNT = 500.0
        AMOUNT_BUFFER = 150.0
        RENT_PERIOD_DAYS = 8
        PERIOD_HALF = RENT_PERIOD_DAYS // 2

        # Convert DateTime to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['DateTime']):
            df['DateTime'] = pd.to_datetime(df['DateTime'])

        # Helper columns
        df['year'] = df['DateTime'].dt.year
        df['month'] = df['DateTime'].dt.month
        df['day'] = df['DateTime'].dt.day
        df['days_in_month'] = df['DateTime'].dt.days_in_month

        # Assign each date to a period: period is the year+month of the month containing the 4th day of the period
        def get_period_id(row):
            if row['day'] > row['days_in_month'] - PERIOD_HALF:
                # Last 4 days of month: period is next month
                next_month = row['month'] + 1
                next_year = row['year']
                if next_month > 12:
                    next_month = 1
                    next_year += 1
                return f"{next_year:04d}-{next_month:02d}"
            elif row['day'] <= PERIOD_HALF:
                # First 4 days of month: period is this month
                return f"{row['year']:04d}-{row['month']:02d}"
            else:
                return np.nan
        df['rent_period'] = df.apply(get_period_id, axis=1)

        # Only consider transactions in a rent period and above min rent amount
        period_mask = df['rent_period'].notna() & (df['Amount'].abs() >= MIN_RENT_AMOUNT)
        period_df = df[period_mask].copy()

        # For each period, find the first transaction that has a similar transaction in either adjacent period
        rent_mask = pd.Series(False, index=df.index)
        unique_periods = sorted(period_df['rent_period'].unique())
        for i, period in enumerate(unique_periods):
            current_period_df = period_df[period_df['rent_period'] == period]
            if current_period_df.empty:
                continue
            # Find adjacent periods
            prev_period = unique_periods[i-1] if i > 0 else None
            next_period = unique_periods[i+1] if i < len(unique_periods)-1 else None
            # For each transaction in this period
            for idx, row in current_period_df.iterrows():
                # Only negative amounts (rent is outgoing)
                if row['Amount'] >= 0:
                    continue
                # Check for similar transaction in previous or next period
                def has_similar(period_id):
                    if period_id is None:
                        return False
                    adj_df = period_df[period_df['rent_period'] == period_id]
                    if adj_df.empty:
                        return False
                    amt = row['Amount']
                    return not adj_df[(adj_df['Amount'] >= amt - AMOUNT_BUFFER) & (adj_df['Amount'] <= amt + AMOUNT_BUFFER)].empty
                if has_similar(prev_period) or has_similar(next_period):
                    # Mark only the first matching transaction in this period
                    rent_mask.at[idx] = True
                    # Debug logging
                    print(f"\nIdentified rent transaction:")
                    print(f"Date: {row['DateTime'].date()}")
                    print(f"Amount: {row['Amount']}")
                    print(f"Found match in: {prev_period if has_similar(prev_period) else next_period}")
                    break  # Only one per period

        # Update classification to 'Rent' for identified transactions
        df.loc[rent_mask, 'Classification'] = 'Rent'
        # Drop helper columns
        df = df.drop(columns=['year', 'month', 'day', 'days_in_month', 'rent_period'])
        return df

    def __apply_custom_conditions(self, df):
        """
        Adjusts the 'Amount' column in the DataFrame based on the transaction details and type.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame with transaction details.
        
        Returns:
        pd.DataFrame: Modified DataFrame with adjusted Amounts.
        """
        df = df.copy()
        
        def determine_amount_sign(row):
            details_lower = str(row['Details']).lower()
            transaction_type_lower = str(row['Transaction Type']).lower()
            amount = row['Amount']
            
            # Handle special deposit types - these should always be positive
            if any(deposit_type in transaction_type_lower for deposit_type in ['payroll dep', 'deposit']):
                return abs(amount)
            
            # Handle transfers
            if 'transfer' in details_lower or 'mb-transfer' in details_lower:
                # If it's a transfer to another account, it should be negative
                if 'to' in details_lower:
                    return -abs(amount)
                # If it's a transfer from another account, it should be positive
                elif 'from' in details_lower:
                    return abs(amount)
            
            # Handle withdrawals
            if transaction_type_lower == 'withdrawal':
                return -abs(amount)
            
            # Handle point of sale purchases
            if transaction_type_lower == 'point of sale purchase':
                return -abs(amount)
            
            # Handle government payments - these should be positive
            if any(payment in details_lower for payment in ['gst', 'climate action incentive', 'provincial payment']):
                return abs(amount)
            
            return amount
        
        df['Amount'] = df.apply(determine_amount_sign, axis=1)
        return df

    def import_json(self):
        if hasattr(self, 'base_path') and self.base_path:
            json_file_path = DATABANK_PATH
            category_colors_path = CATEGORY_COLORS_PATH
        else:
            # Use os.path.join for OS-independence
            json_file_path = DATABANK_PATH
            category_colors_path = CATEGORY_COLORS_PATH
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        
        # Create a default databank.json if it doesn't exist
        if not os.path.exists(json_file_path):
            if os.path.exists(category_colors_path):
                # Load categories from category_colors.json
                with open(category_colors_path, 'r') as f:
                    category_colors = json.load(f)
                    default_categories = {
                        "categories": {
                            category: {
                                "totalMatches": 0,
                                "patterns": []
                            } for category in category_colors.keys()
                        }
                    }
            else:
                # If category_colors.json doesn't exist, use minimal default
                default_categories = {
                    "categories": {
                                        "uncategorized": {
                    "totalMatches": 0,
                    "patterns": []
                }
                    }
                }
            with open(json_file_path, 'w') as json_file:
                json.dump(default_categories, json_file, indent=2)
                print(f"Created default databank.json at '{json_file_path}'.")
        
        with open(json_file_path, 'r') as file:
            return json.load(file)

    def export_json(self, updated_json, print_statement=False):
        if hasattr(self, 'base_path') and self.base_path:
            json_file_path = DATABANK_PATH
        else:
            # Use os.path.join for OS-independence
            json_file_path = DATABANK_PATH
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
            
        with open(json_file_path, 'w') as json_file:
            json.dump(updated_json, json_file, indent=2)
        if print_statement: print(f"Exported updated JSON to '{json_file_path}'.")

    def reset_json_matches(self, imported_json):
        for category, keyword_patterns in imported_json.items():
            imported_json[category]['totalMatches'] = 0
            for pattern_ind, _ in enumerate(keyword_patterns.get('patterns')):
                imported_json[category]['patterns'][pattern_ind]['matchCount'] = 0
        return imported_json

    def account_for_investments(self, df_in):
        """
        Accounts for investments by ensuring they are marked as negative amounts when money leaves the account.
        
        Args:
            df_in (pd.DataFrame): The input DataFrame.
            
        Returns:
            pd.DataFrame: The DataFrame with corrected investment transaction signs.
        """
        print(f"Accounting for investments in bank statements.")
        
        df = df_in.copy()
        
        # Find investment transactions
        investment_mask = df['Classification'] == 'Investment'
        
        # Group by date and details to find duplicate investment transactions
        investment_df = df[investment_mask].copy()
        if not investment_df.empty:
            investment_df['Date'] = pd.to_datetime(investment_df['DateTime']).dt.date
            duplicates = investment_df.groupby(['Date', 'Details'])
            
            for (date, details), group in duplicates:
                if len(group) > 1:
                    # For duplicates, keep the negative amount and remove the positive one
                    # This assumes that when we see both positive and negative, the negative is correct
                    # (since it represents money leaving the account)
                    negative_rows = group[group['Amount'] < 0]
                    if not negative_rows.empty:
                        # Keep the negative rows and remove the positive ones
                        df = df.drop(group[group['Amount'] > 0].index)
                    else:
                        # If no negative amounts, make all amounts negative
                        for idx in group.index:
                            df.loc[idx, 'Amount'] = -abs(df.loc[idx, 'Amount'])
                else:
                    # For non-duplicates, check if it's a Wealthsimple transaction
                    idx = group.index[0]
                    details_lower = str(df.loc[idx, 'Details']).lower()
                    transaction_type_lower = str(df.loc[idx, 'Transaction Type']).lower()
                    
                    if 'wealthsimple' in details_lower:
                        # For Wealthsimple transactions, preserve the original sign from the PDF
                        # The sign should already be correct from the PDF parsing
                        continue
                    elif any(keyword in details_lower or keyword in transaction_type_lower
                           for keyword in ['return', 'deposit', 'dividend', 'interest', 'income']):
                        # For returns/deposits, ensure they are positive
                        df.loc[idx, 'Amount'] = abs(df.loc[idx, 'Amount'])
                    else:
                        # For other investment transactions, make them negative
                        df.loc[idx, 'Amount'] = -abs(df.loc[idx, 'Amount'])
        
        return df

    def __detect_duplicates(self, df):
        """
        Detect duplicate transactions based on datetime, account balance and absolute amount.
        Returns a boolean mask where True indicates a duplicate entry.
        """
        # Create a copy of Amount column with absolute values
        df['abs_amount'] = df['Amount'].abs()
        
        # Group by DateTime, account_balance, and absolute amount
        # Keep the first occurrence (False) and mark others as duplicates (True)
        duplicate_mask = df.duplicated(
            subset=['DateTime', 'account_balance', 'abs_amount'],
            keep='first'
        )
        
        # Drop the temporary column
        df.drop('abs_amount', axis=1, inplace=True)
        
        return duplicate_mask

    def __categorize_food_transactions(self, df):
        """
        Categorizes dining-related transactions using a predefined list of restaurant keywords
        and optionally Yelp's business categories. This should be run after initial classification
        but before post-processing.
        """
        print("Categorizing dining transactions...")
        
        # Path to dining keywords file
        dining_keywords_path = DINING_KEYWORDS_PATH
        
        # Create default dining keywords if file doesn't exist
        if not os.path.exists(dining_keywords_path):
            default_keywords = {
                "restaurants": {
                    "keywords": [
                        # Restaurant Types
                        "restaurant", "cafe", "coffee", "bistro", "diner", "eatery", "grill",
                        "steakhouse", "pub", "bar", "brewery", "bakery", "patisserie", "deli",
                        "food court", "food truck", "pop-up", "supper club", "supperclub",
                        
                        # Major Chains
                        "starbucks", "tim hortons", "mcdonalds", "burger king", "wendys",
                        "kfc", "popeyes", "dairy queen", "swiss chalet", "harveys", "a&w",
                        "pizza pizza", "dominos", "pizza hut", "subway", "chipotle", "freshii",
                        "pita pit", "pita land", "five guys", "shoppers drug mart", "walmart",
                        "costco", "sobeys", "loblaws", "metro", "farm boy", "whole foods",
                        "trader joes", "save on foods", "safeway", "superstore", "no frills",
                        
                        # Cuisine Types
                        "american", "italian", "chinese", "japanese", "korean", "vietnamese",
                        "thai", "indian", "mexican", "greek", "mediterranean", "middle eastern",
                        "caribbean", "african", "french", "german", "spanish", "portuguese",
                        "brazilian", "peruvian", "cuban", "jamaican", "ethiopian", "lebanese",
                        "turkish", "russian", "ukrainian", "polish", "hungarian", "czech",
                        "austrian", "swiss", "belgian", "dutch", "scandinavian", "fusion",
                        
                        # Food Types
                        "burger", "pizza", "sandwich", "sub", "wrap", "salad", "soup",
                        "noodles", "ramen", "pho", "curry", "taco", "burrito", "pasta",
                        "fries", "chicken", "fish", "seafood", "sushi", "sashimi", "roll",
                        "dumpling", "dim sum", "noodle", "rice", "bowl", "plate", "meal",
                        
                        # Meal Types
                        "breakfast", "brunch", "lunch", "dinner", "supper", "snack",
                        "dessert", "ice cream", "gelato", "donut", "pastry", "cake",
                        "cookie", "muffin", "bagel", "croissant",
                        
                        # Restaurant Features
                        "buffet", "all-you-can-eat", "takeout", "delivery", "catering",
                        "fine dining", "casual dining", "fast casual", "fast food",
                        "food truck", "pop-up", "supper club", "supperclub",
                        
                        # Common Restaurant Terms
                        "kitchen", "chef", "cuisine", "dining", "eatery", "bistro",
                        "cafe", "coffee shop", "bakery", "patisserie", "deli", "market",
                        "grocery", "supermarket", "convenience", "corner store", "dollar store"
                    ],
                    "patterns": [
                        # Restaurant Chains
                        ["tim", "hortons"],
                        ["starbucks", "coffee"],
                        ["mcdonalds", "restaurant"],
                        ["burger", "king"],
                        ["wendys", "burger"],
                        ["kfc", "chicken"],
                        ["popeyes", "chicken"],
                        ["dairy", "queen"],
                        ["swiss", "chalet"],
                        ["harveys", "burger"],
                        ["a&w", "burger"],
                        ["pizza", "pizza"],
                        ["dominos", "pizza"],
                        ["pizza", "hut"],
                        ["subway", "sandwich"],
                        ["chipotle", "mexican"],
                        ["freshii", "healthy"],
                        ["pita", "pit"],
                        ["pita", "land"],
                        ["five", "guys"],
                        
                        # Cuisine Types with Restaurant
                        ["american", "restaurant"],
                        ["italian", "restaurant"],
                        ["chinese", "restaurant"],
                        ["japanese", "restaurant"],
                        ["korean", "restaurant"],
                        ["vietnamese", "restaurant"],
                        ["thai", "restaurant"],
                        ["indian", "restaurant"],
                        ["mexican", "restaurant"],
                        ["greek", "restaurant"],
                        ["mediterranean", "restaurant"],
                        ["middle", "eastern", "restaurant"],
                        ["caribbean", "restaurant"],
                        ["african", "restaurant"],
                        ["french", "restaurant"],
                        ["german", "restaurant"],
                        ["spanish", "restaurant"],
                        ["portuguese", "restaurant"],
                        ["brazilian", "restaurant"],
                        ["peruvian", "restaurant"],
                        ["cuban", "restaurant"],
                        ["jamaican", "restaurant"],
                        ["ethiopian", "restaurant"],
                        ["lebanese", "restaurant"],
                        ["turkish", "restaurant"],
                        ["russian", "restaurant"],
                        ["ukrainian", "restaurant"],
                        ["polish", "restaurant"],
                        ["hungarian", "restaurant"],
                        ["czech", "restaurant"],
                        ["austrian", "restaurant"],
                        ["swiss", "restaurant"],
                        ["belgian", "restaurant"],
                        ["dutch", "restaurant"],
                        ["scandinavian", "restaurant"],
                        ["fusion", "restaurant"],
                        
                        # Common Food Combinations
                        ["coffee", "shop"],
                        ["ice", "cream"],
                        ["food", "court"],
                        ["food", "truck"],
                        ["all", "you", "can", "eat"],
                        ["fine", "dining"],
                        ["casual", "dining"],
                        ["fast", "casual"],
                        ["fast", "food"],
                        ["take", "out"],
                        ["take", "away"],
                        ["dine", "in"],
                        ["eat", "in"],
                        ["sit", "down"],
                        ["sit-down"],
                        ["sit down"],
                        ["sitdown"],
                        ["dine-in"],
                        ["dine in"],
                        ["dinein"]
                    ]
                }
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(dining_keywords_path), exist_ok=True)
            
            # Save default keywords
            with open(dining_keywords_path, 'w') as f:
                json.dump(default_keywords, f, indent=2)
            print(f"Created default dining keywords file at {dining_keywords_path}")
        
        # Load dining keywords
        try:
            with open(dining_keywords_path, 'r') as f:
                dining_data = json.load(f)
                restaurant_keywords = dining_data.get('restaurants', {}).get('keywords', [])
                restaurant_patterns = dining_data.get('restaurants', {}).get('patterns', [])
        except Exception as e:
            print(f"Error loading dining keywords: {str(e)}")
            return df
        
        # Function to check if transaction details match dining keywords
        def is_dining_transaction(details_list):
            # Convert details to lowercase string for matching
            details_str = ' '.join(details_list).lower()
            
            # Check for exact keyword matches
            for keyword in restaurant_keywords:
                if keyword.lower() in details_str:
                    return True
            
            # Check for pattern matches (all words in pattern must be present)
            for pattern in restaurant_patterns:
                if all(word.lower() in details_str for word in pattern):
                    return True
            
            return False
        
        # Apply dining categorization to uncategorized transactions
        uncategorized_mask = df['Classification'].str.lower() == 'uncategorized'
        if uncategorized_mask.any():
            # Get the uncategorized transactions
            uncategorized_df = df[uncategorized_mask].copy()
            
            # Apply dining categorization
            dining_mask = uncategorized_df['Processed Details'].apply(is_dining_transaction)
            if dining_mask.any():
                # Update classification for dining transactions
                df.loc[uncategorized_df[dining_mask].index, 'Classification'] = 'Dining'
                print(f"Identified {dining_mask.sum()} dining transactions")
        
        return df

    def __categorize_shopping_transactions(self, df):
        """
        Categorizes shopping-related transactions using a predefined list of retail keywords.
        Excludes supermarkets and grocery stores. This should be run after initial classification
        but before post-processing.
        """
        print("Categorizing shopping transactions...")
        
        # Path to shopping keywords file
        shopping_keywords_path = SHOPPING_KEYWORDS_PATH
        
        # Create default shopping keywords if file doesn't exist
        if not os.path.exists(shopping_keywords_path):
            default_keywords = {
                "retail": {
                    "keywords": [
                        # Department Stores
                        "walmart", "target", "costco", "canadian tire", "home depot", "lowes",
                        "ikea", "hudson's bay", "the bay", "sears", "winners", "marshalls",
                        "home sense", "bed bath beyond", "bed bath & beyond", "bed bath and beyond",
                        
                        # Clothing & Fashion
                        "h&m", "zara", "gap", "old navy", "banana republic", "roots", "lululemon",
                        "nike", "adidas", "under armour", "puma", "reebok", "converse", "vans",
                        "foot locker", "sport chek", "sporting life", "mec", "mountain equipment co-op",
                        "aritzia", "anthropologie", "urban outfitters", "forever 21", "uniqlo",
                        "mango", "guess", "tommy hilfiger", "ralph lauren", "calvin klein",
                        "michael kors", "kate spade", "coach", "louis vuitton", "gucci", "prada",
                        
                        # Electronics & Technology
                        "best buy", "future shop", "apple store", "microsoft store", "dell",
                        "lenovo", "hp", "samsung", "sony", "lg", "asus", "acer", "logitech",
                        "staples", "the source", "canada computers", "memory express",
                        
                        # Home & Furniture
                        "structube", "leons", "the brick", "ashley", "ashley furniture",
                        "sleep country", "sleep country canada", "wayfair", "west elm",
                        "crate and barrel", "pottery barn", "williams-sonoma", "home hardware",
                        "rona", "home depot", "lowes", "canadian tire",
                        
                        # Beauty & Cosmetics
                        "sephora", "ulta", "shoppers drug mart", "london drugs", "rexall",
                        "the body shop", "lush", "mac", "mac cosmetics", "l'oreal", "maybelline",
                        "revlon", "clinique", "estee lauder", "lancome", "chanel", "dior",
                        
                        # Books & Media
                        "chapters", "indigo", "coles", "coles books", "bookstore", "amazon",
                        "ebay", "kijiji", "facebook marketplace", "walmart marketplace",
                        
                        # Sports & Outdoor
                        "sport chek", "sporting life", "mec", "mountain equipment co-op",
                        "atmosphere", "sail", "bass pro shops", "cabela's", "decathlon",
                        
                        # Pet Supplies
                        "pet smart", "petsmart", "pet valu", "petvalue", "global pet foods",
                        
                        # Office Supplies
                        "staples", "grand & toy", "business depot", "officedepot",
                        
                        # General Retail Terms
                        "store", "shop", "boutique", "outlet", "mall", "plaza", "marketplace",
                        "retail", "department store", "specialty store", "concept store",
                        "flagship store", "pop-up", "popup", "pop up", "shop-in-shop",
                        
                        # Shopping Features
                        "sale", "clearance", "discount", "outlet", "factory", "warehouse",
                        "showroom", "gallery", "studio", "atelier", "workshop", "market",
                        "bazaar", "emporium", "arcade", "complex", "center", "centre"
                    ],
                    "patterns": [
                        # Department Stores
                        ["the", "bay"],
                        ["hudson's", "bay"],
                        ["hudsons", "bay"],
                        ["canadian", "tire"],
                        ["home", "depot"],
                        ["bed", "bath", "beyond"],
                        ["bed", "bath", "&", "beyond"],
                        ["bed", "bath", "and", "beyond"],
                        
                        # Clothing & Fashion
                        ["h", "&", "m"],
                        ["banana", "republic"],
                        ["mountain", "equipment", "co-op"],
                        ["mountain", "equipment", "coop"],
                        ["mountain", "equipment", "co", "op"],
                        ["sport", "chek"],
                        ["sporting", "life"],
                        ["urban", "outfitters"],
                        ["forever", "21"],
                        ["tommy", "hilfiger"],
                        ["ralph", "lauren"],
                        ["calvin", "klein"],
                        ["michael", "kors"],
                        ["kate", "spade"],
                        ["louis", "vuitton"],
                        
                        # Electronics & Technology
                        ["best", "buy"],
                        ["future", "shop"],
                        ["apple", "store"],
                        ["microsoft", "store"],
                        ["canada", "computers"],
                        ["memory", "express"],
                        
                        # Home & Furniture
                        ["sleep", "country"],
                        ["crate", "and", "barrel"],
                        ["pottery", "barn"],
                        ["williams", "sonoma"],
                        ["home", "hardware"],
                        
                        # Beauty & Cosmetics
                        ["shoppers", "drug", "mart"],
                        ["london", "drugs"],
                        ["body", "shop"],
                        ["mac", "cosmetics"],
                        ["estee", "lauder"],
                        
                        # Books & Media
                        ["chapters", "indigo"],
                        ["coles", "books"],
                        
                        # Sports & Outdoor
                        ["bass", "pro", "shops"],
                        ["global", "pet", "foods"],
                        
                        # Office Supplies
                        ["grand", "&", "toy"],
                        ["business", "depot"],
                        ["office", "depot"],
                        
                        # Shopping Features
                        ["pop", "up", "store"],
                        ["pop-up", "store"],
                        ["popup", "store"],
                        ["shop", "in", "shop"],
                        ["shop-in-shop"],
                        ["factory", "outlet"],
                        ["warehouse", "sale"],
                        ["clearance", "sale"],
                        ["discount", "store"],
                        ["outlet", "mall"],
                        ["shopping", "center"],
                        ["shopping", "centre"],
                        ["shopping", "mall"],
                        ["shopping", "plaza"],
                        ["retail", "store"],
                        ["department", "store"],
                        ["specialty", "store"],
                        ["concept", "store"],
                        ["flagship", "store"]
                    ],
                    "exclude_keywords": [
                        # Supermarkets and Grocery Stores to Exclude
                        "supermarket", "super store", "superstore", "grocery", "groceries",
                        "food basics", "no frills", "freshco", "foodland", "farm boy",
                        "whole foods", "trader joes", "save on foods", "safeway", "sobeys",
                        "loblaws", "metro", "longos", "longo's", "zehrs", "zehr's",
                        "fortinos", "fortino's", "maxi", "maxi & cie", "maxi and cie",
                        "provigo", "iga", "co-op", "coop", "co op", "co-operative",
                        "cooperative", "co operative", "independent", "independent grocer",
                        "independent grocers", "independent grocery", "independent groceries",
                        "food store", "food market", "grocery store", "grocery market",
                        "supermarket", "super market", "super store", "superstore"
                    ]
                }
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(shopping_keywords_path), exist_ok=True)
            
            # Save default keywords
            with open(shopping_keywords_path, 'w') as f:
                json.dump(default_keywords, f, indent=2)
            print(f"Created default shopping keywords file at {shopping_keywords_path}")
        
        # Load shopping keywords
        try:
            with open(shopping_keywords_path, 'r') as f:
                shopping_data = json.load(f)
                retail_keywords = shopping_data.get('retail', {}).get('keywords', [])
                retail_patterns = shopping_data.get('retail', {}).get('patterns', [])
                exclude_keywords = shopping_data.get('retail', {}).get('exclude_keywords', [])
        except Exception as e:
            print(f"Error loading shopping keywords: {str(e)}")
            return df
        
        # Function to check if transaction details match retail keywords
        def is_shopping_transaction(details_list):
            # Convert details to lowercase string for matching
            details_str = ' '.join(details_list).lower()
            
            # First check if it matches any exclude keywords
            for keyword in exclude_keywords:
                if keyword.lower() in details_str:
                    return False
            
            # Check for exact keyword matches
            for keyword in retail_keywords:
                if keyword.lower() in details_str:
                    return True
            
            # Check for pattern matches (all words in pattern must be present)
            for pattern in retail_patterns:
                if all(word.lower() in details_str for word in pattern):
                    return True
            
            return False
        
        # Apply shopping categorization to uncategorized transactions
        uncategorized_mask = df['Classification'].str.lower() == 'uncategorized'
        if uncategorized_mask.any():
            # Get the uncategorized transactions
            uncategorized_df = df[uncategorized_mask].copy()
            
            # Apply shopping categorization
            shopping_mask = uncategorized_df['Processed Details'].apply(is_shopping_transaction)
            if shopping_mask.any():
                # Update classification for shopping transactions
                df.loc[uncategorized_df[shopping_mask].index, 'Classification'] = 'Shopping'
                print(f"Identified {shopping_mask.sum()} shopping transactions")
        
        return df