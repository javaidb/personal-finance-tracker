import pdfplumber
import re
import pandas as pd
from tqdm import tqdm
import json
import os

from src.modules.helper_fns import GeneralHelperFns

# Try to import config, but don't fail if it doesn't exist
try:
    from src.config import config
except ImportError:
    print("Warning: Could not import config, using defaults")
    # Create a simple empty config
    import types
    config = types.SimpleNamespace()
    config.rent_ranges = []

class PDFReader(GeneralHelperFns):

    def __init__(self, base_path=None):
        super().__init__()
        self.base_path = base_path
        self.df_raw = self.generate_fin_df()

    def process_raw_df(self):
        filtered_df = self.df_preprocessing(self.df_raw)
        filtered_df = self.recalibrate_amounts(filtered_df)
        filtered_df = self.combine_balances_across_accounts(filtered_df)
        filtered_df = self.tabulate_gap_balances(filtered_df)
        self.filtered_df = self.df_postprocessing(filtered_df)
        return self.filtered_df

    def extract_features_from_pdf(self, pdf_file, x_tolerance=2, init_y_top=440, reg_y_top=210):

        parent_dir = os.path.dirname(pdf_file)
        # Fix path handling to be OS-independent
        account_dir = os.path.basename(os.path.dirname(parent_dir))
        
        if account_dir in ["Chequing", "Savings"]:
            x_right = 600
            x_left = 750
            init_y_top = 400
            regular_page_box = (70, reg_y_top, x_right, x_left)
            initial_page_box = (70, init_y_top, x_right, x_left)
        elif account_dir == "Credit":
            x_right = 400
            x_left = 730
            reg_y_top = 100
            init_y_top = 600
            regular_page_box = (70, reg_y_top, x_right, x_left)
            initial_page_box = (70, init_y_top, x_right, x_left)

        text = ''
        # Open the PDF file
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                if page.page_number == 1:
                    rect = initial_page_box
                else:
                    rect = regular_page_box

                # Extract text only from the specified rectangle
                text += page.crop(rect).extract_text(x_tolerance=x_tolerance)

        # Process the extracted text
        lines = text.split('\n')
        return lines

    def __grab_pattern(self, account_type):
        if account_type == "Credit":
            return r'(\d{3})\s+(\w{3}\s+\d{1,2})\s+(\w{3}\s+\d{1,2})\s+(.+)\s+(\d+\.\d{2})(-)?'
        elif account_type in ["Chequing", "Savings"]:
            return r'(\w+ \d+)\s+(.*?)\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'


    def process_transactions_from_lines(self, pdf_lines, account_type):

        transactions = []

        pattern = self.__grab_pattern(account_type)

        for i, line in enumerate(pdf_lines):
            match = re.match(pattern, line)
            # print(line)
            if match:
                match_groups = list(match.groups())
                if account_type == "Credit":
                    ref_num = match_groups[0]
                    transaction_date = match_groups[1]
                    post_date = match_groups[2]
                    details = match_groups[3]
                    amount = match_groups[4]
                    # Condition where credit statement indicates this was a deposit, so we see a "-" at end of amount
                    if match_groups[5] == "-":
                        amount = str(float(amount) * -1)
                    # Create a dictionary for the transaction and add it to the list
                    transaction = {
                        'Reference #': ref_num,
                        'Transaction Date': transaction_date,
                        'Post Date': post_date,
                        'Details': details,
                        'Amount': amount,
                        'Transaction Type': '',
                    }
                elif account_type in ["Chequing", "Savings"]:
                    date = match_groups[0]
                    transaction_type = match_groups[1]
                    if match_groups[3] is None:
                        amount = None
                        balance = match_groups[2]
                    else:
                        amount = match_groups[2]
                        balance = match_groups[3]

                    # Create a dictionary for the transaction and add it to the list
                    transaction = {
                        'Transaction Date': date,
                        'Transaction Type': transaction_type,
                        'Amount': amount,
                        'Balance': balance,
                        'Details': pdf_lines[i+1],
                    }
                transactions.append(transaction)

        return transactions

    def __calculate_transaction_year(self, row):
        month_str, _ = row['Transaction Date'].split()
        month_str = month_str.lower()
        if (month_str in ['nov', 'dec']) and (row['Statement Month'].lower() == 'january'):
            return int(row['Statement Year']) - 1
        else:
            return int(row['Statement Year'])

    def generate_fin_df(self, account_types=None):
        overall_df = pd.DataFrame()
        if account_types is None:
            account_types = self.read_all_account_type_folder_names()
            # account_types = ['Chequing', 'Credit']
        for account_type in account_types:
            account_names = self.read_all_account_folder_names(account_type)
            for account_name in account_names:
                pdf_files = self.read_all_files(account_type, account_name)
                for pdf_file in tqdm(pdf_files, desc=f"Reading PDFs from '{account_name}' bucket"):
                    pdf_file_atts = self.grab_pdf_name_attributes(pdf_file)
                    # Use base_path if provided
                    if self.base_path:
                        pdf_file_path = os.path.join(self.base_path, "bank_statements", account_type, account_name, pdf_file)
                    else:
                        pdf_file_path = self.process_import_path(pdf_file, account_type, account_name)
                    lines = self.extract_features_from_pdf(pdf_file_path, x_tolerance=2, init_y_top=440, reg_y_top=210)
                    transactions= self.process_transactions_from_lines(lines, account_type)
                    temp_df = pd.DataFrame(transactions)
                    temp_df[['Statement Year']] = pdf_file_atts['year']
                    temp_df[['Statement Month']] = pdf_file_atts['month']
                    temp_df[['Account Type']] = account_type
                    temp_df[['Account Name']] = account_name
                    overall_df = pd.concat([temp_df, overall_df], ignore_index=True)

        overall_df['Transaction Year'] = overall_df.apply(self.__calculate_transaction_year, axis=1)
        overall_df['DateTime'] = overall_df['Transaction Date'] + ' ' + overall_df['Transaction Year'].astype(str)
        overall_df['DateTime'] = pd.to_datetime(overall_df['DateTime'])
        return overall_df



    def df_preprocessing(self, df_in):

        print(f"Pre-processing bank statements.")

        df = df_in.copy()
        df = self.__process_transaction_details(df)
        df = self.__classify_transactions(df)
        
        for col in ['Balance', 'Amount']:
            df[col] = pd.to_numeric(df[col].replace(',', '', regex=True), errors='coerce')

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
        Classifies transactions based on a databank.

        Args:
            df_ (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: A new DataFrame with an additional 'adjusted_amount' column.
        """

        imported_json = self.import_json().get('categories')
        imported_json = self.reset_json_matches(imported_json)

        def categorize_strings(row, categories = imported_json):
            list_of_strings = row['Processed Details']
            s = ' '.join(list_of_strings)
            s_lower = s.lower()
            found_categories = []
            matched_keywords = []
            pattern_indices = []
            for category, keyword_patterns in categories.items():
                for pattern_ind, keyword_pattern in enumerate(keyword_patterns.get('patterns')):
                    keyword = ' '.join(keyword_pattern.get('terms'))
                    if all(word.lower() in s_lower for word in keyword.split()):
                        found_categories.append(category)
                        matched_keywords.append(keyword)
                        pattern_indices.append(pattern_ind)
                        # Increment match tally
                        categories[category]['patterns'][pattern_ind]['matchCount'] += 1
                # if found_category:
                #     break

            # If multiple matches found (should ONLY be a case where there is one string is a subset of another, e.g. 'uber' vs 'uber eats'), grab one with most strings matched 
            if matched_keywords and len(matched_keywords) > 1:
                index = max(range(len(matched_keywords)), key=lambda i: len(matched_keywords[i].split()))
            else:
                index = 0
            # Now assign category/associated match keywords
            if not found_categories:
                found_category = "uncharacterized"
                matched_keyword= None
            else:
                found_category = found_categories[index]
                matched_keyword = matched_keywords[index]
                pattern_index = pattern_indices[index]
                # Increment match tallies
                categories[found_category]['totalMatches'] += 1
            
                assigned_matches = categories[found_category]['patterns'][pattern_index].setdefault('assignedDetailMatch', [])
                if list_of_strings not in assigned_matches:
                    assigned_matches.append(list_of_strings)
            categories_with_outcol = {"categories": categories}
            self.export_json(categories_with_outcol)
            return pd.Series([found_category, matched_keyword])

        df[['Classification', 'Matched Keyword']] = df.apply(categorize_strings, axis = 1)
        return df

    def recalibrate_amounts(self, df_in):
        """
        Adjusts the 'amount' column by applying a negative sign if the balance decreases
        and the absolute difference matches the amount within a threshold **For CHEQUING/SAVINGS; else awe assume it is negative.

        Args:
            df_in (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: A new DataFrame with an additional 'adjusted_amount' column.
        """

        print(f"Recalibrating amounts in bank statements.")

        df = df_in.copy()
        for account_type in df['Account Type'].unique().tolist():
            df_by_account_type = df.loc[(df['Account Type'] == account_type)]
            for account_name in df_by_account_type['Account Name'].unique().tolist():
                print(f"{account_type} {account_name}")
                df_by_account_name = df_by_account_type.loc[(df['Account Name'] == account_name)]
                df_by_account_name = self.sort_df(df_by_account_name)

                def modify_function(df):
                    df = df.sort_values(by='DateTime')
                    df = df.groupby('DateTime', group_keys=False).apply(lambda x: x.sort_index(), include_groups=False)
                    if account_type in ['Chequing', 'Savings']:
                        df['balance_diff'] = df['Balance'].diff().fillna(0)
                        df['Amount'] = df.apply(
                            lambda row: -1*row['Amount'] if row['balance_diff'] < 0 else row['Amount'],
                            axis=1
                        )
                        # df['Amount'] = df.apply(
                        #     lambda row: -1*row['Amount'] if row['balance_diff'] < 0 and abs(row['balance_diff']) - abs(row['Amount']) < 0.1 else row['Amount'],
                        #     axis=1
                        # )
                        df.drop(columns=['balance_diff'], inplace=True)
                    elif account_type == 'Credit':
                        df['Amount'] = df['Amount'] * -1
                    return df
                
                modified_rows = modify_function(df_by_account_name)

                df.update(modified_rows)   
        return df

    def combine_balances_across_accounts(self, df_in):
        merged_df = df_in.copy()
        merged_df = merged_df.sort_values(['DateTime', 'Account Type'])
        merged_df['balance_change'] = merged_df.groupby('Account Type')['Balance'].diff().fillna(merged_df['Balance'])
        merged_df['Balance'] = merged_df['balance_change'].cumsum()
        merged_df = merged_df.drop(columns=['balance_change'])
        return merged_df

    def tabulate_gap_balances(self, df_in):
        df = df_in.copy()
        df = self.sort_df(df).reset_index(drop=True)
        for i in range(1, len(df)):
            if pd.isna(df.loc[df.index[i], 'Balance']):
                # If balance is NaN, calculate it based on previous row
                previous_balance = df.loc[df.index[i-1], 'Balance']
                current_amount = df.loc[df.index[i], 'Amount']
                if pd.isna(previous_balance):
                    continue
                new_balance = previous_balance + current_amount
                df.loc[df.index[i], 'Balance'] = new_balance
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

        substring_exclusion_list = ['mb credit', 'mb transfer', 'opening balance', 'closing balance']
        fullstring_exclusion_list = ['from']
        # mask = ~df['Processed Details'].apply(
        #     lambda x: any(any(substring in item for item in x) for substring in substrings_to_remove)
        # )
        
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
        return self.sort_df(df_filtered)

    def __identify_rent_payments(self, df_in, rent_ranges):
        df = df_in.copy()

        # If no rent_ranges provided, return unchanged DataFrame
        if not rent_ranges:
            return df

        df = df.sort_values('DateTime')

        df['day_of_month'] = df['DateTime'].dt.day
        # print(df.head(4))
        rent_mask = ((df['day_of_month'] <= 5) | (df['day_of_month'] >= 25) & 
            (df['Details'].apply(lambda x: any('transfer' in word for word in x.lower().split())))
        )
        
        amount_mask = pd.Series(False, index=df.index)
        for rent_per_property in rent_ranges:
            amount_mask |= df['Amount'].abs().between(rent_per_property["min"], rent_per_property["max"])
        
        rent_mask &= amount_mask
        
        # Update classification to 'rent' for these rows
        df.loc[rent_mask, 'Classification'] = 'Rent'
        
        # Drop the helper column
        df = df.drop(columns=['day_of_month'])
        
        return df

    def __apply_custom_conditinos(self, df):
        
        """
        Adjusts the 'Amount' column in the DataFrame based on the 'Transaction Type' column.
        
        If 'Transaction Type' is 'Withdrawal', the corresponding 'Amount' is multiplied by -1.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame with 'Transaction Type' and 'Amount' columns.
        
        Returns:
        pd.DataFrame: Modified DataFrame with adjusted Amounts.
        """
        # if 'Transaction Type' not in df.columns or 'Amount' not in df.columns:
        #     raise ValueError("DataFrame must contain 'type' and 'amount' columns.")
        df['Amount'] = df['Amount'].where(df['Transaction Type'] != 'Withdrawal', df['Amount'].abs() * -1)
        df['Amount'] = df['Amount'].where(df['Transaction Type'] != 'Deposit', df['Amount'].abs())
        return df

    def import_json(self):
        if hasattr(self, 'base_path') and self.base_path:
            json_file_path = os.path.join(self.base_path, "cached_data", "databank.json")
        else:
            # Use os.path.join for OS-independence
            json_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'cached_data', 'databank.json')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        
        # Create a default databank.json if it doesn't exist
        if not os.path.exists(json_file_path):
            default_categories = {
                "categories": {
                    "Groceries": {
                        "totalMatches": 0,
                        "patterns": [
                            {
                                "terms": ["grocery"],
                                "dateAdded": "2023-01-01T00:00:00.000000Z",
                                "lastUpdated": "2023-01-01T00:00:00.000000Z",
                                "matchCount": 0
                            }
                        ]
                    },
                    "Dining": {
                        "totalMatches": 0,
                        "patterns": [
                            {
                                "terms": ["restaurant"],
                                "dateAdded": "2023-01-01T00:00:00.000000Z",
                                "lastUpdated": "2023-01-01T00:00:00.000000Z",
                                "matchCount": 0
                            }
                        ]
                    },
                    "Transport": {
                        "totalMatches": 0,
                        "patterns": [
                            {
                                "terms": ["gas", "fuel"],
                                "dateAdded": "2023-01-01T00:00:00.000000Z",
                                "lastUpdated": "2023-01-01T00:00:00.000000Z",
                                "matchCount": 0
                            }
                        ]
                    },
                    "Bills": {
                        "totalMatches": 0,
                        "patterns": [
                            {
                                "terms": ["bill", "utility"],
                                "dateAdded": "2023-01-01T00:00:00.000000Z",
                                "lastUpdated": "2023-01-01T00:00:00.000000Z",
                                "matchCount": 0
                            }
                        ]
                    },
                    "Rent": {
                        "totalMatches": 0,
                        "patterns": [
                            {
                                "terms": ["rent"],
                                "dateAdded": "2023-01-01T00:00:00.000000Z",
                                "lastUpdated": "2023-01-01T00:00:00.000000Z",
                                "matchCount": 0
                            }
                        ]
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
            json_file_path = os.path.join(self.base_path, "cached_data", "databank.json")
        else:
            # Use os.path.join for OS-independence
            json_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'cached_data', 'databank.json')
        
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