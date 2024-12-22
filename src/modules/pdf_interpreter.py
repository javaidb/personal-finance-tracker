import pdfplumber
import re
import pandas as pd
from tqdm.notebook import tqdm_notebook
import json

from src.modules.helper_fns import *

def extract_features_from_pdf(pdf_file, x_tolerance=2, init_y_top=440, reg_y_top=210):

    parent_dir = os.path.dirname(pdf_file)
    dir_before_file = parent_dir.split("/")[-2]
    if dir_before_file in ["Chequing", "Savings"]:
        x_right = 600
        x_left = 750
        init_y_top = 400
        regular_page_box = (70, reg_y_top, x_right, x_left)
        initial_page_box = (70, init_y_top, x_right, x_left)
    elif dir_before_file == "Credit":
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

def __grab_pattern(account_type):
    if account_type == "Credit":
        return r'(\d{3})\s+(\w{3}\s+\d{1,2})\s+(\w{3}\s+\d{1,2})\s+(.+)\s+(\d+.\d{2})'
    elif account_type in ["Chequing", "Savings"]:
        return r'(\w+ \d+)\s+(.*?)\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'


def process_transactions_from_lines(pdf_lines, account_type):

    transactions = []

    pattern = __grab_pattern(account_type)

    for i, line in enumerate(pdf_lines):
        match = re.match(pattern, line)
        # print(line)
        if match:
            match_groups = list(match.groups())
            if account_type == "Credit":
                ref_num = match.group(1)
                transaction_date = match.group(2)
                post_date = match.group(3)
                details = match.group(4)
                amount = match.group(5)
                # Create a dictionary for the transaction and add it to the list
                transaction = {
                    'Reference #': ref_num,
                    'Transaction Date': transaction_date,
                    'Post Date': post_date,
                    'Details': details,
                    'Amount': amount,
                    'Transaction Type': pdf_lines[i+1],
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

def __calculate_transaction_year(row):
    month_str, _ = row['Transaction Date'].split()
    month_str = month_str.lower()
    if (month_str in ['nov', 'dec']) and (row['Statement Month'].lower() == 'january'):
        return int(row['Statement Year']) - 1
    else:
        return int(row['Statement Year'])

def generate_fin_df(account_types=None):
    overall_df = pd.DataFrame()
    if account_types is None:
        account_types = read_all_account_type_folder_names()
        # account_types = ['Chequing', 'Credit']
    for account_type in account_types:
        account_names = read_all_account_folder_names(account_type)
        for account_name in account_names:
            pdf_files = read_all_files(account_type, account_name)
            for pdf_file in tqdm_notebook(pdf_files, desc=f"Reading PDFs from '{account_name}' bucket"):
                pdf_file_atts = grab_pdf_name_attributes(pdf_file)
                pdf_file_path = process_import_path(pdf_file, account_type, account_name)
                lines = extract_features_from_pdf(pdf_file_path, x_tolerance=2, init_y_top=440, reg_y_top=210)
                transactions= process_transactions_from_lines(lines, account_type)
                temp_df = pd.DataFrame(transactions)
                temp_df[['Statement Year']] = pdf_file_atts['year']
                temp_df[['Statement Month']] = pdf_file_atts['month']
                temp_df[['Account Type']] = account_type
                temp_df[['Account Name']] = account_name
                overall_df = pd.concat([temp_df, overall_df], ignore_index=True)

    overall_df['Transaction Year'] = overall_df.apply(__calculate_transaction_year, axis=1)
    overall_df['DateTime'] = overall_df['Transaction Date'] + ' ' + overall_df['Transaction Year'].astype(str)
    overall_df['DateTime'] = pd.to_datetime(overall_df['DateTime'])
    return overall_df

def df_preprocessing(df_in):
    df = df_in.copy()
    df = __process_transaction_details(df)
    df = __classify_transactions(df)
    
    for col in ['Balance', 'Amount']:
        df[col] = pd.to_numeric(df[col].replace(',', '', regex=True), errors='coerce')

    # substrings_to_avoid = ['MB', 'Tax', 'Opening Balance']
    # columns_to_filter = ['Details', 'Transaction Type']
    # pattern = '|'.join(substrings_to_avoid)
    # mask = df[columns_to_filter].apply(lambda x: x.str.contains(pattern, na=False)).any(axis=1)
    # df = df[~mask]

    return df

def __process_transaction_details(df):
    def process_row(row):
        # Function to process each string element
        def process_string(s):
            return re.sub(r'[^a-z\s&]', '', s.lower())
        
        # Split the string by space and process each element
        return [process_string(elem) for elem in row.split() if process_string(elem)]

    # Apply the function to a DataFrame column
    df['Processed Details'] = df['Details'].apply(process_row)

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

def __classify_transactions(df):
    """
    Classifies transactions based on a databank.

    Args:
        df_ (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with an additional 'adjusted_amount' column.
    """

    with open('../../cached_data/databank.json', 'r') as file:
        categories = json.load(file)

    def categorize_strings(list_of_strings):
        s = ' '.join(list_of_strings)
        s_lower = s.lower()
        found_category = None
        for category, keywords in categories.items():
            for keyword in keywords:
                if ' ' in keyword:
                    # Multi-word keyword
                    if all(word.lower() in s_lower for word in keyword.split()):
                        found_category = category
                        break
                else:
                    # Single-word keyword
                    if keyword.lower() in s_lower:
                        found_category = category
                        break
            if found_category:
                break
        if not found_category:
            found_category = "uncharacterized"
        return found_category

    df['Classification'] = df['Processed Details'].apply(categorize_strings)
    return df

def recalibrate_amounts(df_in):
    """
    Adjusts the 'amount' column by applying a negative sign if the balance decreases
    and the absolute difference matches the amount within a threshold **For CHEQUING/SAVINGS; else awe assume it is negative.

    Args:
        df_in (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with an additional 'adjusted_amount' column.
    """

    df = df_in.copy()
    for account_type in df['Account Type'].unique().tolist():
        df_by_account_type = df.loc[(df['Account Type'] == account_type)]
        for account_name in df_by_account_type['Account Name'].unique().tolist():
            print(f"{account_type} {account_name}")
            df_by_account_name = df_by_account_type.loc[(df['Account Name'] == account_name)]
            df_by_account_name = sort_df(df_by_account_name)

            def modify_function(df):
                df = df.sort_values(by='DateTime')
                df = df.groupby('DateTime', group_keys=False).apply(lambda x: x.sort_index(), include_groups=False)
                if account_type in ['Chequing', 'Savings']:
                    df['balance_diff'] = df['Balance'].diff().fillna(0)
                    df['Amount'] = df.apply(
                        lambda row: -1*row['Amount'] if row['balance_diff'] < 0 and abs(row['balance_diff']) - abs(row['Amount']) < 0.1 else row['Amount'],
                        axis=1
                    )
                    df.drop(columns=['balance_diff'], inplace=True)
                    # print(df.head(5))
                elif account_type == 'Credit':
                    df['Amount'] = df['Amount'] * -1
                return df
            
            modified_rows = modify_function(df_by_account_name)

            df.update(modified_rows)   
    return df

def combine_balances_across_accounts(df_in):
    merged_df = df_in.copy()
    merged_df = merged_df.sort_values(['DateTime', 'Account Type'])
    merged_df['balance_change'] = merged_df.groupby('Account Type')['Balance'].diff().fillna(merged_df['Balance'])
    merged_df['Balance'] = merged_df['balance_change'].cumsum()
    merged_df = merged_df.drop(columns=['balance_change'])
    return merged_df

def tabulate_gap_balances(df_in):
    df = df_in.copy()
    df = sort_df(df).reset_index(drop=True)
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

def df_postprocessing(df_in):
    df = df_in.copy()
    substrings_to_avoid = ['MB', 'Tax', 'Opening Balance']
    columns_to_filter = ['Details', 'Transaction Type']
    pattern = '|'.join(substrings_to_avoid)
    mask = df[columns_to_filter].apply(lambda x: x.str.contains(pattern, na=False)).any(axis=1)
    df = df[~mask]
    return sort_df(df)