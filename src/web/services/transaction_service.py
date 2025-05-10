import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from ..models.transaction import Transaction
from src.modules.pdf_interpreter import PDFReader
from src.modules.helper_fns import GeneralHelperFns
import numpy as np
from src.modules.merchant_categorizer import MerchantCategorizer
import json
import os
import re

class TransactionService:
    """Service class for handling transaction-related operations."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.pdf_reader = PDFReader(base_path=base_path)
        self.helper = GeneralHelperFns()
        self.processed_df = None
        self.databank_path = os.path.join(base_path, "cached_data", "databank.json")
        self.load_categories()

    def load_categories(self):
        """Load categories from databank.json."""
        try:
            if os.path.exists(self.databank_path):
                with open(self.databank_path, 'r') as f:
                    databank = json.load(f)
                    self.categories = list(databank.get('categories', {}).keys())
                    self.category_patterns = databank.get('categories', {})
            else:
                # Initialize with default categories from app.py
                from ..app import CATEGORY_COLORS
                self.categories = list(CATEGORY_COLORS.keys())
                self.category_patterns = {
                    category: {
                        'patterns': [],
                        'color': color,
                        'totalMatches': 0
                    }
                    for category, color in CATEGORY_COLORS.items()
                }
                # Save the default categories to databank.json
                os.makedirs(os.path.dirname(self.databank_path), exist_ok=True)
                with open(self.databank_path, 'w') as f:
                    json.dump({'categories': self.category_patterns}, f, indent=4)
        except Exception as e:
            print(f"Error loading categories: {str(e)}")
            self.categories = []
            self.category_patterns = {}

    def categorize_transaction(self, description: str) -> str:
        """Automatically categorize a transaction based on its description and stored patterns."""
        if not description or not self.category_patterns:
            return 'Uncategorized'
        
        description = description.lower()
        best_match = None
        highest_match_count = 0
        
        for category, data in self.category_patterns.items():
            for pattern in data.get('patterns', []):
                terms = pattern.get('terms', [])
                if not terms:
                    continue
                
                # Count how many terms from the pattern appear in the description
                match_count = sum(1 for term in terms if term.lower() in description)
                if match_count > highest_match_count:
                    highest_match_count = match_count
                    best_match = category
        
        return best_match if best_match else 'Uncategorized'

    def process_statements(self) -> bool:
        """Process bank statements and store the result."""
        try:
            # Check if we already have processed data in memory
            if self.processed_df is not None:
                return True
                
            # Process the statements using cached data where possible
            self.processed_df = self.pdf_reader.process_raw_df()
            
            # Apply automatic categorization only to uncategorized transactions
            if self.processed_df is not None and not self.processed_df.empty:
                # Fill NaN classifications with 'Uncategorized'
                self.processed_df['Classification'] = self.processed_df['Classification'].fillna('Uncategorized')
                
                # Only apply categorization to uncategorized transactions
                uncategorized_mask = self.processed_df['Classification'].str.lower() == 'uncategorized'
                self.processed_df.loc[uncategorized_mask, 'Classification'] = \
                    self.processed_df.loc[uncategorized_mask, 'Details'].apply(self.categorize_transaction)
            
            return True if self.processed_df is not None and not self.processed_df.empty else False
        except Exception as e:
            print(f"Error processing statements: {str(e)}")
            return False

    def get_transactions(self, start_date=None, end_date=None) -> List[Dict[str, Any]]:
        """Get transactions with optional date filtering."""
        if self.processed_df is None:
            print("DEBUG: processed_df is None")
            return []
        
        try:
            print(f"DEBUG: Initial DataFrame size: {len(self.processed_df)}")
            df = self.processed_df.copy()
            
            # Debug: Check what columns we actually have and their data types
            print(f"DEBUG: Available columns and their types:")
            for col in df.columns:
                print(f"{col}: {df[col].dtype}")
                if col in ['account_balance', 'running_balance', 'Amount']:
                    print(f"Sample values for {col}:")
                    print(df[col].head())
                    print(f"Number of NaN values in {col}: {df[col].isna().sum()}")
            
            # Ensure DateTime is in proper format and sort by date in descending order
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df = df.sort_values(by='DateTime', ascending=False)
            
            # First check if the balance columns exist
            if 'account_balance' not in df.columns:
                print("DEBUG: account_balance column missing!")
            if 'running_balance' not in df.columns:
                print("DEBUG: running_balance column missing!")
            
            # Ensure balance columns are numeric
            df['account_balance'] = pd.to_numeric(df['account_balance'], errors='coerce')
            df['running_balance'] = pd.to_numeric(df['running_balance'], errors='coerce')
            
            print("DEBUG: Balance columns after conversion:")
            print("account_balance sample:", df['account_balance'].head())
            print("running_balance sample:", df['running_balance'].head())
            
            # Apply date filtering if provided
            if start_date:
                print(f"DEBUG: Filtering by start_date: {start_date}")
                df = df[df['DateTime'] >= pd.to_datetime(start_date)]
            if end_date:
                print(f"DEBUG: Filtering by end_date: {end_date}")
                df = df[df['DateTime'] <= pd.to_datetime(end_date)]
            
            # If no date filters provided, default to last 3 months
            if not start_date and not end_date and not df.empty:
                latest_date = df['DateTime'].max()
                three_months_ago = latest_date - pd.DateOffset(months=3)
                print(f"DEBUG: Using default 3-month filter: {three_months_ago} to {latest_date}")
                df = df[df['DateTime'] >= three_months_ago]
            
            print(f"DEBUG: Final DataFrame size after filtering: {len(df)}")
            
            # Convert DataFrame to list of dictionaries with explicit error handling
            try:
                transactions = []
                for _, row in df.iterrows():
                    transaction = {}
                    for key, value in row.items():
                        if isinstance(value, (np.int64, np.float64)):
                            transaction[key] = float(value)
                        elif pd.isna(value):
                            transaction[key] = None
                        elif isinstance(value, pd.Timestamp):
                            transaction[key] = value.strftime('%Y-%m-%d')
                        else:
                            transaction[key] = value
                    transactions.append(transaction)
                
                if transactions:
                    print("DEBUG: First transaction after conversion:")
                    print("account_balance:", transactions[0].get('account_balance'))
                    print("running_balance:", transactions[0].get('running_balance'))
                    print("Amount:", transactions[0].get('Amount'))
            except Exception as e:
                print(f"DEBUG: Error converting transactions: {str(e)}")
                return []
            
            return transactions
            
        except Exception as e:
            print(f"DEBUG: Error getting transactions: {str(e)}")
            import traceback
            print(f"DEBUG: Full traceback: {traceback.format_exc()}")
            return []

    def get_category_data(self) -> Dict[str, Any]:
        """Get category data including counts, spending, and colors."""
        if self.processed_df is None:
            return {
                "error": "No data has been processed yet",
                "counts": {},
                "spending": {},
                "colors": {},
                "has_spending_data": False
            }
        
        try:
            # Fill NaN values in Classification with 'Uncategorized'
            df = self.processed_df.copy()
            df['Classification'] = df['Classification'].fillna('Uncategorized')
            
            # Count transactions by category
            category_counts = df['Classification'].value_counts().to_dict()
            
            # Calculate total spending by category (for negative amounts only - expenses)
            spending_df = df[df['Amount'] < 0]
            
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
                category_colors[category] = self.get_category_color(category)
            
            return {
                "counts": category_counts,
                "spending": category_spending,
                "colors": category_colors,
                "has_spending_data": len(category_spending) > 0
            }
            
        except Exception as e:
            print(f"Error generating category data: {str(e)}")
            return {
                "error": str(e),
                "counts": {},
                "spending": {},
                "colors": {},
                "has_spending_data": False
            }

    def get_category_color(self, category: str) -> str:
        """Get a consistent color for a category."""
        colors = {
            'Groceries': '#4CAF50',
            'Dining': '#FF9800',
            'Transport': '#2196F3',
            'Shopping': '#E91E63',
            'Bills': '#9C27B0',
            'Entertainment': '#00BCD4',
            'Activities': '#FFEB3B',
            'Online': '#795548',
            'Income': '#4CAF50',
            'Rent': '#F44336',
            'Investment': '#009688',
            'Uncategorized': '#9E9E9E'
        }
        return colors.get(category, '#9E9E9E')

    def get_balance_chart_data(self) -> Dict[str, Any]:
        """Get balance chart data."""
        if self.processed_df is None:
            print("No processed data available")
            return {
                "labels": [],
                "datasets": [{
                    "label": "Balance",
                    "data": [],
                    "borderColor": "#BB2525",
                    "pointRadius": 2,
                    "fill": False
                }]
            }
        
        try:
            df = self.processed_df.copy()
            print(f"Processing {len(df)} transactions for balance chart")
            
            # Sort by date and ensure DateTime is in proper format
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df = df.sort_values('DateTime')
            
            # Group by date to get daily balances using running_balance
            daily_df = df.groupby('DateTime').agg({
                'running_balance': 'last'  # Take the last balance for each day
            }).reset_index()
            
            # Format dates as strings
            date_labels = daily_df['DateTime'].dt.strftime('%Y-%m-%d').tolist()
            balance_data = daily_df['running_balance'].apply(float).tolist()
            
            print(f"Generated chart data with {len(balance_data)} points")
            
            chart_data = {
                "labels": date_labels,
                "datasets": [{
                    "label": "Balance",
                    "data": balance_data,
                    "borderColor": "#BB2525",
                    "pointRadius": 2,
                    "fill": False
                }]
            }
            
            return chart_data
            
        except Exception as e:
            print(f"Error generating balance chart data: {str(e)}")
            # Return empty dataset on error
            return {
                "labels": [],
                "datasets": [{
                    "label": "Balance",
                    "data": [],
                    "borderColor": "#BB2525",
                    "pointRadius": 2,
                    "fill": False
                }]
            }

    def get_pelt_analysis_data(self) -> Dict[str, Any]:
        """Get PELT analysis data."""
        if self.processed_df is None:
            return {
                "error": "No data has been processed yet",
                "labels": [],
                "datasets": [],
                "changePoints": [],
                "rateOfChange": {"labels": [], "data": []}
            }
        
        try:
            df = self.processed_df.copy()
            
            # Sort by date and ensure DateTime is in proper format
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df = df.sort_values('DateTime')
            
            # Group by date to get daily balances using running_balance
            daily_df = df.groupby('DateTime').agg({
                'running_balance': 'last'  # Take the last balance for each day
            }).reset_index()
            
            # Ensure running_balance is numeric
            daily_df['running_balance'] = pd.to_numeric(daily_df['running_balance'], errors='coerce')
            
            # Drop any rows where running_balance is NaN after conversion
            daily_df = daily_df.dropna(subset=['running_balance'])
            
            if len(daily_df) < 2:
                return {
                    "error": "Insufficient data points for analysis",
                    "labels": [],
                    "datasets": [],
                    "changePoints": [],
                    "rateOfChange": {"labels": [], "data": []}
                }
            
            # Calculate numeric time and change points
            daily_df['numeric_time'] = (daily_df['DateTime'] - daily_df['DateTime'].min()).dt.total_seconds()
            
            # Set PELT parameters - for broader trend detection
            segment_settings = {
                'penalty': 50,  # Higher penalty for more general segmentation
                'min_size': 14,  # Minimum 2 weeks of data for a segment
                'jump': 5,     # Larger jump for broader trends
                'model': "l2"  # Linear model for financial trends
            }
            
            from ruptures import Pelt
            model = Pelt(model=segment_settings['model'], min_size=segment_settings['min_size'], jump=segment_settings['jump'])
            model.fit(daily_df['running_balance'].values.reshape(-1, 1))
            change_points = model.predict(pen=segment_settings['penalty'])
            
            # Ensure first and last points are included
            if change_points[0] != 0:
                change_points = [0] + change_points
            if change_points[-1] != len(daily_df):
                change_points = change_points + [len(daily_df)]
            
            # Calculate segments and trends
            segments = []
            concat_y_segments = []
            concat_coeffs = []
            change_dates = []
            
            # Initialize arrays with NaN to match data length
            trend_values = np.full(len(daily_df), np.nan)
            rate_of_change = np.full(len(daily_df), np.nan)
            
            for i in range(len(change_points) - 1):
                start_idx = change_points[i]
                end_idx = change_points[i + 1]
                
                # Get segment data
                segment_time = daily_df['numeric_time'].iloc[start_idx:end_idx].values
                segment_balance = daily_df['running_balance'].iloc[start_idx:end_idx].values
                
                if len(segment_time) >= 2:
                    # Fit polynomial to segment
                    coeffs = np.polyfit(segment_time, segment_balance, 1)
                    segment_y_values = np.polyval(coeffs, segment_time)
                    
                    # Store trend values in the main array
                    trend_values[start_idx:end_idx] = segment_y_values
                    
                    # Calculate weekly rate of change
                    weekly_change_rate = coeffs[0] * (7 * 24 * 60 * 60)  # Convert to weekly rate
                    rate_of_change[start_idx:end_idx] = weekly_change_rate
                    
                    # Store change point dates
                    if i < len(change_points) - 2:  # Don't include the last point
                        change_dates.append(daily_df['DateTime'].iloc[change_points[i+1]].strftime('%Y-%m-%d'))
                else:
                    # For very short segments, use linear interpolation
                    if len(segment_balance) == 2:
                        trend_values[start_idx:end_idx] = segment_balance
                        rate = (segment_balance[-1] - segment_balance[0]) / (segment_time[-1] - segment_time[0])
                        weekly_rate = rate * (7 * 24 * 60 * 60)
                        rate_of_change[start_idx:end_idx] = weekly_rate
            
            # Convert NaN to None for JSON serialization
            trend_values = [None if np.isnan(x) else float(x) for x in trend_values]
            rate_of_change = [None if np.isnan(x) else float(x) for x in rate_of_change]
            
            # Convert running_balance to float for JSON serialization
            balance_data = [float(x) for x in daily_df['running_balance'].tolist()]
            
            # Prepare data for chart.js
            chart_data = {
                "labels": daily_df['DateTime'].dt.strftime('%Y-%m-%d').tolist(),
                "datasets": [
                    {
                        "label": "Balance",
                        "data": balance_data,
                        "borderColor": "#BB2525",
                        "pointRadius": 2,
                        "fill": False,
                        "yAxisID": "y"
                    },
                    {
                        "label": "Trend Segments",
                        "data": trend_values,
                        "borderColor": "#BCBF07",
                        "borderWidth": 2,
                        "pointRadius": 0,
                        "fill": False,
                        "yAxisID": "y"
                    }
                ],
                "changePoints": change_dates,
                "rateOfChange": {
                    "labels": daily_df['DateTime'].dt.strftime('%Y-%m-%d').tolist(),
                    "data": rate_of_change
                }
            }
            
            return chart_data
            
        except Exception as e:
            print(f"Error performing PELT analysis: {str(e)}")
            return {
                "error": str(e),
                "labels": [],
                "datasets": [],
                "changePoints": [],
                "rateOfChange": {"labels": [], "data": []}
            }

    def get_monthly_trends_data(self) -> Dict[str, Any]:
        """Get monthly trends data."""
        if self.processed_df is None:
            return {}
        
        df = self.processed_df.copy()
        
        # Ensure DateTime is in proper format
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        
        # Extract month from transaction date
        df['Month'] = df['DateTime'].dt.strftime('%Y-%m')
        
        # Sort months chronologically
        months = sorted(df['Month'].unique())
        
        # Initialize datasets for each category
        datasets = []
        categories = sorted(df['Classification'].unique())
        
        for category in categories:
            category_data = []
            for month in months:
                month_df = df[(df['Month'] == month) & (df['Classification'] == category)]
                total_spending = abs(month_df[month_df['Amount'] < 0]['Amount'].sum())
                category_data.append(float(total_spending))
            
            # Get color for category
            color = self.get_category_color(category)
            
            datasets.append({
                'label': category,
                'data': category_data,
                'backgroundColor': color,
                'borderColor': color,
                'borderWidth': 1
            })
        
        # Add total income dataset
        income_data = []
        for month in months:
            month_df = df[df['Month'] == month]
            total_income = month_df[month_df['Amount'] > 0]['Amount'].sum()
            income_data.append(float(total_income))
        
        datasets.append({
            'label': 'Income',
            'data': income_data,
            'backgroundColor': '#4CAF50',
            'borderColor': '#4CAF50',
            'borderWidth': 1
        })
        
        return {
            'labels': months,
            'datasets': datasets
        }

    def clear_cache(self) -> bool:
        """Clear the PDF cache."""
        try:
            # Clear the PDF cache
            success = self.pdf_reader.clear_pdf_cache()
            if success:
                # Reset the processed data
                self.processed_df = None
                # Force a reload of the data
                self.process_statements()
            return success
        except Exception as e:
            print(f"Error clearing cache: {str(e)}")
            return False

    def df_preprocessing(self, df_in):
        """Preprocess the DataFrame by cleaning and converting data types."""
        print(f"DEBUG: Starting preprocessing with {len(df_in)} rows")

        df = df_in.copy()
        
        # Process transaction details first
        df = self.__process_transaction_details(df)
        print(f"DEBUG: After processing transaction details: {len(df)} rows")
        
        # Convert numeric columns
        numeric_columns = ['Balance', 'Amount', 'account_balance', 'running_balance']
        for col in numeric_columns:
            if col in df.columns:
                print(f"DEBUG: Converting {col} to numeric")
                # First remove any commas
                if df[col].dtype == 'object':  # Only process strings
                    df[col] = df[col].str.replace(',', '', regex=False)
                # Then convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill NaN values with 0
                df[col] = df[col].fillna(0)
                print(f"DEBUG: {col} conversion complete. Sample values: {df[col].head()}")
        
        # Classify transactions
        df = self.__classify_transactions(df)
        print(f"DEBUG: After classification: {len(df)} rows")
        
        return df

    def __process_transaction_details(self, df):
        print("DEBUG: Starting __process_transaction_details")
        def process_row(details_row, transaction_type_row):
            concat_row = details_row + " " + transaction_type_row
            def process_string(s):
                return re.sub(r'[^a-z\s&]', '', s.lower())
            
            # Split the string by space and process each element
            return [process_string(elem) for elem in concat_row.split() if process_string(elem)]

        # Apply the function to a DataFrame column
        df['Processed Details'] = df.apply(lambda row: process_row(row['Details'], row['Transaction Type']), axis=1)
        print(f"DEBUG: After creating Processed Details: {len(df)} rows")
        
        # Only filter out opening/closing balance entries
        def filter_df(df, column, substrings):
            def check_row(row):
                for item in substrings:
                    if any('opening balance' in s.lower() for s in row) or any('closing balance' in s.lower() for s in row):
                        return True
                return False
            
            filtered_df = df[~df[column].apply(check_row)]
            print(f"DEBUG: Filtered out {len(df) - len(filtered_df)} rows containing opening/closing balance")
            return filtered_df
        
        substrings = ['opening balance', 'closing balance']
        df = filter_df(df, 'Processed Details', substrings)
        print(f"DEBUG: Final row count in __process_transaction_details: {len(df)}")

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
            
        df = self.__apply_custom_conditinos(df)

        # Only filter out opening/closing balance entries
        substring_exclusion_list = ['opening balance', 'closing balance']
        
        df['details_str'] = df['Processed Details'].apply(lambda x: ' '.join(x).lower())

        def exclude_rows(details_str, substring_exclusion_list):
            for excl in substring_exclusion_list:
                if excl in details_str:
                    return False
            return True

        df_filtered = df[df['details_str'].apply(lambda x: exclude_rows(x, substring_exclusion_list))]
        df_filtered = df_filtered.drop(columns=['details_str'])
        return self.sort_df(df_filtered)

    def recategorize_transactions(self) -> bool:
        """Recategorize all transactions after merchant updates."""
        try:
            if self.processed_df is None:
                return False
            
            # Create a copy of the DataFrame
            df = self.processed_df.copy()
            
            # Initialize merchant categorizer
            merchant_categorizer = MerchantCategorizer(self.base_path)
            
            def categorize_transaction(row):
                details = row['Details']
                transaction_type = row.get('Transaction Type', '')
                
                if pd.isna(details):
                    return 'Uncategorized'
                
                # Process details similar to __process_transaction_details
                concat_details = f"{details} {transaction_type}"
                processed_details = re.sub(r'[^a-zA-Z\s&]', '', concat_details)
                
                # Try merchant-based categorization
                category, _ = merchant_categorizer.categorize_transaction(processed_details)
                return category
            
            # Apply categorization
            df['Classification'] = df.apply(categorize_transaction, axis=1)
            
            # Update the processed DataFrame
            self.processed_df = df
            
            print(f"Recategorized {len(df)} transactions")
            return True
        except Exception as e:
            print(f"Error recategorizing transactions: {str(e)}")
            return False 