import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from ..models.transaction import Transaction
from src.modules.statement_interpreter import StatementInterpreter
from src.modules.helper_fns import GeneralHelperFns
import numpy as np
from src.modules.merchant_categorizer import MerchantCategorizer, ManualTransactionCategorizer
import json
import os
import re
from ..constants.categories import CATEGORY_COLORS, get_category_color
from src.config.paths import DATABANK_PATH, CATEGORY_COLORS_PATH

class TransactionService:
    """Service class for handling transaction-related operations."""
    
    def __init__(self, base_path: Path, bank_name=None):
        self.base_path = base_path
        self.bank_name = bank_name
        self.pdf_reader = StatementInterpreter(base_path=base_path, bank_name=bank_name)
        self.helper = GeneralHelperFns(base_path=base_path, bank_name=bank_name)
        self.processed_df = None
        self.databank_path = DATABANK_PATH
        self.manual_categorizer = ManualTransactionCategorizer(base_path=base_path)
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
                # On first run, load categories from category_colors.json
                if os.path.exists(CATEGORY_COLORS_PATH):
                    with open(CATEGORY_COLORS_PATH, 'r') as f:
                        category_colors = json.load(f)
                        self.categories = list(category_colors.keys())
                        self.category_patterns = {
                            category: {
                                'patterns': [],
                                'color': color,
                                'totalMatches': 0
                            }
                            for category, color in category_colors.items()
                        }
                else:
                    # If category_colors.json doesn't exist, use minimal default
                    self.categories = ["uncategorized"]
                    self.category_patterns = {
                        "uncategorized": {
                            'patterns': [],
                            'color': '#607D8B',
                            'totalMatches': 0
                        }
                    }
                # Save the categories to databank.json
                os.makedirs(os.path.dirname(self.databank_path), exist_ok=True)
                with open(self.databank_path, 'w') as f:
                    json.dump({'categories': self.category_patterns}, f, indent=4)
        except Exception as e:
            print(f"Error loading categories: {str(e)}")
            self.categories = []
            self.category_patterns = {}

    def categorize_transaction(self, description: str, transaction_id: str = None) -> str:
        """Automatically categorize a transaction based on its description and stored patterns."""
        # First check for manual category
        if transaction_id:
            manual_category = self.manual_categorizer.get_manual_category(transaction_id)
            if manual_category:
                return manual_category

        if not description or not self.category_patterns:
            return 'uncategorized'
        
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
        
        return best_match if best_match else 'uncategorized'

    def set_manual_category(self, transaction_id: str, category: str) -> bool:
        """Set a manual category for a specific transaction."""
        try:
            print(f"[DEBUG] set_manual_category called with transaction_id={transaction_id}, category={category}")
            if self.processed_df is None:
                print("[DEBUG] processed_df is None")
                return False

            # Create composite key from DateTime and Amount if not already in that format
            if '_' not in transaction_id:
                print(f"[DEBUG] Invalid transaction_id format: {transaction_id}")
                return False

            # Split the composite key
            date_time, amount = transaction_id.split('_')
            amount = float(amount)

            # Find the matching transaction
            mask = (self.processed_df['DateTime'] == date_time) & (self.processed_df['Amount'] == amount)
            matching_transactions = self.processed_df[mask]

            if matching_transactions.empty:
                print(f"[DEBUG] No matching transaction found for {transaction_id}")
                return False

            # Get the first matching transaction's index
            transaction_idx = matching_transactions.index[0]

            # Add manual category with index
            success = self.manual_categorizer.add_manual_category(transaction_id, category)
            if success:
                # Update the DataFrame
                self.processed_df.at[transaction_idx, 'Classification'] = category
                
                # Update the index in the manual categories list
                for entry in self.manual_categorizer.manual_categories:
                    if entry['datetime'] == date_time and entry['amount'] == amount:
                        entry['index'] = int(transaction_idx)
                        break
                self.manual_categorizer.save_manual_categories()
            return success
        except Exception as e:
            print(f"Error setting manual category: {str(e)}")
            return False

    def remove_manual_category(self, transaction_id: str) -> bool:
        """Remove a manual category for a specific transaction."""
        try:
            if self.processed_df is None:
                return False

            # Create composite key from DateTime and Amount if not already in that format
            if '_' not in transaction_id:
                print(f"[DEBUG] Invalid transaction_id format: {transaction_id}")
                return False

            # Split the composite key
            date_time, amount = transaction_id.split('_')
            amount = float(amount)

            # Find the matching transaction
            mask = (self.processed_df['DateTime'] == date_time) & (self.processed_df['Amount'] == amount)
            matching_transactions = self.processed_df[mask]

            if matching_transactions.empty:
                print(f"[DEBUG] No matching transaction found for {transaction_id}")
                return False

            # Get the first matching transaction's index
            transaction_idx = matching_transactions.index[0]
            
            # Remove manual category
            success = self.manual_categorizer.remove_manual_category(transaction_id)
            if success:
                # Recategorize the transaction
                description = self.processed_df.at[transaction_idx, 'Details']
                new_category = self.categorize_transaction(description)
                self.processed_df.at[transaction_idx, 'Classification'] = new_category
            return success
        except Exception as e:
            print(f"Error removing manual category: {str(e)}")
            return False

    def process_statements(self) -> bool:
        """Process bank statements and store the result."""
        try:
            # Check if we already have processed data in memory
            if self.processed_df is not None:
                return True
            # Process the statements using cached data where possible
            self.processed_df = self.pdf_reader.process_raw_df()
            # Set index to id as string if id column exists
            if self.processed_df is not None and 'id' in self.processed_df.columns:
                self.processed_df['id'] = self.processed_df['id'].astype(str)
                self.processed_df.set_index('id', inplace=True)
            # Apply automatic categorization only to uncategorized transactions
            if self.processed_df is not None and not self.processed_df.empty:
                # Fill NaN classifications with 'uncategorized'
                self.processed_df['Classification'] = self.processed_df['Classification'].fillna('uncategorized')
                # Only apply categorization to uncategorized transactions
                uncategorized_mask = self.processed_df['Classification'] == 'uncategorized'
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
        """Get category data including counts, spending, income, and colors."""
        if self.processed_df is None:
            return {
                "error": "No data has been processed yet",
                "counts": {},
                "spending": {},
                "income": {},
                "colors": {},
                "has_spending_data": False,
                "has_income_data": False
            }
        
        try:
            # Fill NaN values in Classification with 'uncategorized'
            df = self.processed_df.copy()
            df['Classification'] = df['Classification'].fillna('uncategorized')
            
            # Count transactions by category
            category_counts = df['Classification'].value_counts().to_dict()
            
            # Calculate total spending by category (for negative amounts only - expenses)
            spending_df = df[df['Amount'] < 0]
            
            # Calculate total income by category (for positive amounts only)
            income_df = df[df['Amount'] > 0]
            
            # Handle case where there are no negative amounts (expenses)
            if spending_df.empty:
                category_spending = {}
                print("Warning: No transactions with negative amounts found for spending chart")
            else:
                category_spending = spending_df.groupby('Classification')['Amount'].sum().abs().to_dict()
            
            # Handle case where there are no positive amounts (income)
            if income_df.empty:
                category_income = {}
                print("Warning: No transactions with positive amounts found for income chart")
            else:
                category_income = income_df.groupby('Classification')['Amount'].sum().to_dict()
            
            # Create color mapping for each category
            category_colors = {}
            
            # Get all categories from both transactions and databank
            transaction_categories = set(list(category_counts.keys()) + list(category_spending.keys()) + list(category_income.keys()))
            databank_categories = set(self.categories)  # Get all categories from databank
            all_categories = transaction_categories.union(databank_categories)
            
            # Initialize counts, spending, and income for all categories
            for category in all_categories:
                category_colors[category] = get_category_color(category)
                # Initialize counts, spending, and income for categories not in transactions
                if category not in category_counts:
                    category_counts[category] = 0
                if category not in category_spending:
                    category_spending[category] = 0
                if category not in category_income:
                    category_income[category] = 0
            
            return {
                "counts": category_counts,
                "spending": category_spending,
                "income": category_income,
                "colors": category_colors,
                "has_spending_data": len(category_spending) > 0,
                "has_income_data": len(category_income) > 0
            }
            
        except Exception as e:
            print(f"Error generating category data: {str(e)}")
            return {
                "error": str(e),
                "counts": {},
                "spending": {},
                "income": {},
                "colors": {},
                "has_spending_data": False,
                "has_income_data": False
            }

    def get_category_color(self, category: str) -> str:
        """Get a consistent color for a category."""
        return get_category_color(category)

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
            
            # Group by date to get daily balances using running_balance and running_balance_plus_investments
            daily_df = df.groupby('DateTime').agg({
                'running_balance': 'last',  # Take the last balance for each day
                'running_balance_plus_investments': 'last'  # Take the last balance including investments
            }).reset_index()
            
            # Format dates as strings
            date_labels = daily_df['DateTime'].dt.strftime('%Y-%m-%d').tolist()
            balance_data = daily_df['running_balance'].apply(float).tolist()
            balance_with_investments_data = daily_df['running_balance_plus_investments'].apply(float).tolist()
            
            print(f"Generated chart data with {len(balance_data)} points")
            
            chart_data = {
                "labels": date_labels,
                "datasets": [
                    {
                        "label": "Balance",
                        "data": balance_data,
                        "borderColor": "#BB2525",
                        "pointRadius": 2,
                        "fill": False
                    },
                    {
                        "label": "Balance + Investments",
                        "data": balance_with_investments_data,
                        "borderColor": "#4CAF50",
                        "pointRadius": 2,
                        "fill": False
                    }
                ]
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
            
            # Set PELT parameters - tuned for fewer, broader segments (target ~6 segments)
            segment_settings = {
                'penalty': 300,  # Much higher penalty for fewer segments (was 50)
                'min_size': 30,  # Minimum 1 month of data for a segment (was 14)
                'jump': 7,       # Slightly larger jump for broader trends
                'model': "l2"   # Linear model for financial trends
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
            
            # --- SECOND PASS: Merge adjacent segments with similar gradients (within 25%) ---
            # First, extract gradients for each segment
            segment_gradients = []
            for i in range(len(change_points) - 1):
                start_idx = change_points[i]
                end_idx = change_points[i + 1]
                segment_time = daily_df['numeric_time'].iloc[start_idx:end_idx].values
                segment_balance = daily_df['running_balance'].iloc[start_idx:end_idx].values
                if len(segment_time) >= 2:
                    coeffs = np.polyfit(segment_time, segment_balance, 1)
                    segment_gradients.append(coeffs[0])
                else:
                    segment_gradients.append(0.0)

            # Merge logic: if next gradient is within 25% of current, merge
            merged_change_points = [change_points[0]]
            i = 0
            while i < len(segment_gradients) - 1:
                g1 = segment_gradients[i]
                g2 = segment_gradients[i + 1]
                # Avoid division by zero, treat both zero as similar
                if g1 == 0 and g2 == 0:
                    similar = True
                elif g1 == 0 or g2 == 0:
                    similar = False
                else:
                    ratio = abs(g2 - g1) / max(abs(g1), abs(g2))
                    similar = ratio <= 0.25
                if similar:
                    # Merge: skip the next change point
                    i += 1
                else:
                    merged_change_points.append(change_points[i + 1])
                    i += 1
            # Always include the last point
            if merged_change_points[-1] != change_points[-1]:
                merged_change_points.append(change_points[-1])
            change_points = merged_change_points

            # --- Continue with segment calculation as before, but using merged change_points ---
            # Calculate segments and trends
            segments = []
            concat_y_segments = []
            concat_coeffs = []
            change_dates = []
            change_directions = []  # New: direction of each changepoint
            trend_values = np.full(len(daily_df), np.nan)
            rate_of_change = np.full(len(daily_df), np.nan)
            for i in range(len(change_points) - 1):
                start_idx = change_points[i]
                end_idx = change_points[i + 1]
                segment_time = daily_df['numeric_time'].iloc[start_idx:end_idx].values
                segment_balance = daily_df['running_balance'].iloc[start_idx:end_idx].values
                if len(segment_time) >= 2:
                    coeffs = np.polyfit(segment_time, segment_balance, 1)
                    segment_y_values = np.polyval(coeffs, segment_time)
                    trend_values[start_idx:end_idx] = segment_y_values
                    weekly_change_rate = coeffs[0] * (7 * 24 * 60 * 60)
                    rate_of_change[start_idx:end_idx] = weekly_change_rate
                    if i < len(change_points) - 2:
                        change_dates.append(daily_df['DateTime'].iloc[change_points[i+1]].strftime('%Y-%m-%d'))
                        # Determine direction: compare average of last 5 points of this segment to average of first 5 points of next segment
                        this_segment_end = segment_balance[-5:] if len(segment_balance) >= 5 else segment_balance
                        next_segment_start_idx = change_points[i+1]
                        next_segment_end_idx = change_points[i+2] if i+2 < len(change_points) else len(daily_df)
                        next_segment_balance = daily_df['running_balance'].iloc[next_segment_start_idx:next_segment_end_idx].values
                        next_segment_start = next_segment_balance[:5] if len(next_segment_balance) >= 5 else next_segment_balance
                        
                        # Calculate averages
                        this_avg = np.mean(this_segment_end)
                        next_avg = np.mean(next_segment_start)
                        
                        if next_avg > this_avg:
                            change_directions.append('up')
                        else:
                            change_directions.append('down')
                else:
                    if len(segment_balance) == 2:
                        trend_values[start_idx:end_idx] = segment_balance
                        rate = (segment_balance[-1] - segment_balance[0]) / (segment_time[-1] - segment_time[0])
                        weekly_rate = rate * (7 * 24 * 60 * 60)
                        rate_of_change[start_idx:end_idx] = weekly_rate
                        # For short segments, use the same logic but with available points
                        if i < len(change_points) - 2:
                            change_dates.append(daily_df['DateTime'].iloc[change_points[i+1]].strftime('%Y-%m-%d'))
                            this_segment_end = segment_balance[-min(5, len(segment_balance)):]
                            next_segment_start_idx = change_points[i+1]
                            next_segment_end_idx = change_points[i+2] if i+2 < len(change_points) else len(daily_df)
                            next_segment_balance = daily_df['running_balance'].iloc[next_segment_start_idx:next_segment_end_idx].values
                            next_segment_start = next_segment_balance[:min(5, len(next_segment_balance))]
                            
                            # Calculate averages
                            this_avg = np.mean(this_segment_end)
                            next_avg = np.mean(next_segment_start)
                            
                            if next_avg > this_avg:
                                change_directions.append('up')
                            else:
                                change_directions.append('down')
            
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
                        "borderColor": "#5D6D7E",
                        "pointRadius": 2,
                        "fill": False,
                        "yAxisID": "y"
                    },
                    {
                        "label": "Trend Segments",
                        "data": trend_values,
                        "borderColor": "#000000",
                        "borderWidth": 2,
                        "pointRadius": 0,
                        "fill": False,
                        "yAxisID": "y"
                    }
                ],
                "changePoints": change_dates,
                "changePointDirections": change_directions,  # New: up/down for each changepoint
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
                # Calculate net amount (positive + negative) for this category in this month
                net_amount = month_df['Amount'].sum()
                category_data.append(float(net_amount))
            
            # Get color for category
            color = self.get_category_color(category)
            
            datasets.append({
                'label': category,
                'data': category_data,
                'backgroundColor': color,
                'borderColor': color,
                'borderWidth': 1
            })
        
        return {
            'labels': months,
            'datasets': datasets
        }

    def get_weekly_trends_data(self) -> Dict[str, Any]:
        """Get weekly trends data."""
        if self.processed_df is None:
            return {}

        df = self.processed_df.copy()

        # Ensure DateTime is in proper format
        df['DateTime'] = pd.to_datetime(df['DateTime'])

        # Create a helper column to group by year-month and week number within that month
        df['YearMonth'] = df['DateTime'].dt.to_period('M')
        df['WeekOfYear'] = df['DateTime'].dt.isocalendar().week

        # Calculate which week of the month each transaction belongs to
        # Group by year-month, then assign week numbers based on chronological order
        df['YearMonthWeek'] = df.groupby('YearMonth')['DateTime'].transform(
            lambda x: 'W' + (((x.dt.day - 1) // 7) + 1).astype(str)
        )

        # Create formatted label (e.g., "OCTW1", "OCTW2")
        df['WeekLabel'] = df['DateTime'].dt.strftime('%b').str.upper() + df['YearMonthWeek']

        # Create a sortable key (year-month + week number)
        df['SortKey'] = df['DateTime'].dt.strftime('%Y-%m') + '-' + df['YearMonthWeek']

        # Sort weeks chronologically using the sort key
        week_info = df[['SortKey', 'WeekLabel']].drop_duplicates().sort_values('SortKey')
        weeks = week_info['WeekLabel'].tolist()

        # Initialize datasets for each category
        datasets = []
        categories = sorted(df['Classification'].unique())

        for category in categories:
            category_data = []
            for week_label in weeks:
                week_df = df[(df['WeekLabel'] == week_label) & (df['Classification'] == category)]
                # Calculate net amount (positive + negative) for this category in this week
                net_amount = week_df['Amount'].sum()
                category_data.append(float(net_amount))

            # Get color for category
            color = self.get_category_color(category)

            datasets.append({
                'label': category,
                'data': category_data,
                'backgroundColor': color,
                'borderColor': color,
                'borderWidth': 1
            })

        return {
            'labels': weeks,
            'datasets': datasets
        }

    def get_spending_trends_data(self, start_date: str = None, end_date: str = None, group_range: str = 'month', categories: list = None) -> Dict[str, Any]:
        """Get spending trends data with configurable time and group ranges."""
        if self.processed_df is None:
            return {}
        
        df = self.processed_df.copy()
        
        # Ensure DateTime is in proper format
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        
        # Filter by date range if provided
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df['DateTime'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df['DateTime'] <= end_date]
        
        # Filter by categories if provided
        if categories and len(categories) > 0:
            df = df[df['Classification'].isin(categories)]
        
        # Determine time grouping based on group_range parameter
        if group_range == 'quarter':
            df['TimeGroup'] = df['DateTime'].dt.to_period('Q').astype(str)
        elif group_range == 'year':
            df['TimeGroup'] = df['DateTime'].dt.strftime('%Y')
        else:  # default to month
            df['TimeGroup'] = df['DateTime'].dt.strftime('%Y-%m')
        
        # Sort time groups chronologically
        time_groups = sorted(df['TimeGroup'].unique())
        
        # Initialize datasets for positive and negative flows
        positive_dataset = {
            'label': 'Positive Inflow',
            'data': [],
            'backgroundColor': 'rgba(40, 167, 69, 0.8)',  # Green
            'borderColor': 'rgba(40, 167, 69, 1)',
            'borderWidth': 1,
            'stack': 'Stack 0'
        }
        
        negative_dataset = {
            'label': 'Negative Inflow',
            'data': [],
            'backgroundColor': 'rgba(220, 53, 69, 0.8)',  # Red
            'borderColor': 'rgba(220, 53, 69, 1)',
            'borderWidth': 1,
            'stack': 'Stack 0'
        }
        
        net_dataset = {
            'label': 'Net Amount',
            'data': [],
            'backgroundColor': 'rgba(13, 110, 253, 0.8)',  # Blue
            'borderColor': 'rgba(13, 110, 253, 1)',
            'borderWidth': 1,
            'stack': 'Stack 1'  # Separate stack for net amount
        }
        
        # Calculate data for each time group
        for time_group in time_groups:
            time_df = df[df['TimeGroup'] == time_group]
            
            # Calculate positive and negative flows for all groups combined
            positive_flow = time_df[time_df['Amount'] > 0]['Amount'].sum()
            negative_flow = time_df[time_df['Amount'] < 0]['Amount'].sum()
            net_flow = positive_flow + negative_flow  # negative_flow is already negative
            
            positive_dataset['data'].append(float(positive_flow))
            negative_dataset['data'].append(float(negative_flow))
            net_dataset['data'].append(float(net_flow))
        

        
        # Get available categories for the legend (from original unfiltered data)
        original_df = self.processed_df.copy()
        original_df['DateTime'] = pd.to_datetime(original_df['DateTime'])
        
        # Apply date filtering to get categories in the date range
        if start_date:
            start_date = pd.to_datetime(start_date)
            original_df = original_df[original_df['DateTime'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            original_df = original_df[original_df['DateTime'] <= end_date]
        
        available_categories = sorted(original_df['Classification'].unique().tolist())
        
        return {
            'labels': time_groups,
            'datasets': [positive_dataset, negative_dataset, net_dataset],
            'group_range': group_range,
            'available_categories': available_categories
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
            import traceback
            print(f"Error clearing cache in TransactionService: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
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
            # Only identify rent payments for transactions that aren't already categorized by merchants
            uncategorized_mask = df['Classification'].str.lower() == 'uncategorized'
            if uncategorized_mask.any():
                uncategorized_df = df[uncategorized_mask].copy()
                uncategorized_df = self.__identify_rent_payments(uncategorized_df, config.rent_ranges)
                # Update only the uncategorized transactions
                df.loc[uncategorized_mask, 'Classification'] = uncategorized_df['Classification']
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not load rent ranges from config: {e}")
            # Use default empty list if config not available
            uncategorized_mask = df['Classification'].str.lower() == 'uncategorized'
            if uncategorized_mask.any():
                uncategorized_df = df[uncategorized_mask].copy()
                uncategorized_df = self.__identify_rent_payments(uncategorized_df, [])
                df.loc[uncategorized_mask, 'Classification'] = uncategorized_df['Classification']

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
                current_category = row['Classification']
                
                # If it's already categorized as Rent, preserve that categorization
                if current_category == 'Rent':
                    return current_category
                
                if pd.isna(details):
                    return 'uncategorized'
                
                # Process details similar to __process_transaction_details
                concat_details = f"{details} {transaction_type}"
                processed_details = re.sub(r'[^a-zA-Z\s&]', '', concat_details)
                
                # Try merchant-based categorization
                category, _ = merchant_categorizer.categorize_transaction(processed_details)
                
                # If merchant categorization found something, use it
                if category != "uncategorized":
                    return category
                
                # If no merchant match and already has a non-uncategorized category, preserve it
                if current_category != 'uncategorized':
                    return current_category
                
                return 'uncategorized'
            
            # Apply categorization
            df['Classification'] = df.apply(categorize_transaction, axis=1)
            
            # Update the processed DataFrame
            self.processed_df = df
            
            print(f"Recategorized {len(df)} transactions")
            return True
        except Exception as e:
            print(f"Error recategorizing transactions: {str(e)}")
            return False 