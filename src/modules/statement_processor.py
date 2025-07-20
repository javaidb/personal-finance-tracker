"""
Statement Processor

This module contains the data processing methods for cleaning, categorizing, and analyzing
bank statement transactions. These methods are used by the StatementInterpreter.
"""

import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, List

from src.modules.merchant_categorizer import MerchantCategorizer
from src.config.paths import (
    DATABANK_PATH, 
    UNCATEGORIZED_MERCHANTS_PATH, 
    DINING_KEYWORDS_PATH, 
    SHOPPING_KEYWORDS_PATH,
    MANUAL_CATEGORIES_PATH,
    CATEGORY_COLORS_PATH
)


class StatementProcessor:
    """Handles the processing and analysis of bank statement transactions."""
    
    def __init__(self, base_path=None):
        self.base_path = base_path
        self.databank_path = DATABANK_PATH
        self.uncategorized_path = UNCATEGORIZED_MERCHANTS_PATH
    
    def preprocess_dataframe(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the DataFrame by cleaning and converting data types."""
        print(f"Pre-processing bank statements.")

        df = df_in.copy()
        
        # Process transaction details first
        df = self._process_transaction_details(df)
        
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
        df = self._classify_transactions(df)
        
        # Categorize dining transactions
        df = self._categorize_food_transactions(df)
        
        # Categorize shopping transactions
        df = self._categorize_shopping_transactions(df)
        
        return df
    
    def _process_transaction_details(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean transaction details."""
        def process_row(details_row, transaction_type_row):
            concat_row = details_row + " " + transaction_type_row
            def process_string(s):
                return re.sub(r'[^a-z\s&]', '', s.lower())
            
            # Split the string by space and process each element
            return [process_string(elem) for elem in concat_row.split() if process_string(elem)]

        # Apply the function to a DataFrame column
        processed_details = df.apply(lambda row: process_row(row['Details'], row['Transaction Type']), axis=1)
        df['Processed Details'] = processed_details.values.tolist()

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
    
    def _classify_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classifies transactions using merchant matching first, then falling back to pattern matching."""
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
            self._save_uncategorized_merchants(uncategorized_merchants)
        
        # Report classification stats
        total = len(df)
        print(f"Classification complete: {merchant_matches} by merchant ({merchant_matches/total:.1%}), "
              f"{pattern_matches} by pattern ({pattern_matches/total:.1%}), "
              f"{uncategorized} uncategorized ({uncategorized/total:.1%})")
        
        return df
    
    def _save_uncategorized_merchants(self, merchants: Dict):
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
    
    def recalibrate_amounts(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Adjusts amounts based on account type."""
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
    
    def combine_balances_across_accounts(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Combines transactions from all accounts and calculates running balances."""
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
    
    def detect_duplicates(self, df: pd.DataFrame) -> pd.Series:
        """Detect duplicate transactions based on datetime, account balance and absolute amount."""
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
    
    def postprocess_dataframe(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Post-process the DataFrame."""
        print(f"Post-processing bank statements.")

        df = df_in.copy()

        try:
            from src.config import config
            df = self._identify_rent_payments(df, config.rent_ranges)
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not load rent ranges from config: {e}")
            # Use default empty list if config not available
            df = self._identify_rent_payments(df, [])
            
        df = self._apply_custom_conditions(df)

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
        df_filtered = self._apply_manual_categories(df_filtered)
        
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
        
        # Calculate running_balance_plus_investments
        df_no_transfers = df_no_transfers.sort_values('DateTime')
        
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
    
    def _identify_rent_payments(self, df_in: pd.DataFrame, rent_ranges: List) -> pd.DataFrame:
        """Identifies rent payments by looking for common transactions that occur around month transitions."""
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
    
    def _apply_custom_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adjusts the 'Amount' column in the DataFrame based on the transaction details and type."""
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
    
    def _apply_manual_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply manual categories from manual_categories.json to the DataFrame."""
        if not os.path.exists(MANUAL_CATEGORIES_PATH):
            return df
        try:
            with open(MANUAL_CATEGORIES_PATH, 'r') as f:
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
    
    def _categorize_food_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorizes dining-related transactions using predefined keywords."""
        print("Categorizing dining transactions...")
        
        # Load dining keywords
        try:
            with open(DINING_KEYWORDS_PATH, 'r') as f:
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
    
    def _categorize_shopping_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorizes shopping-related transactions using predefined keywords."""
        print("Categorizing shopping transactions...")
        
        # Load shopping keywords
        try:
            with open(SHOPPING_KEYWORDS_PATH, 'r') as f:
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