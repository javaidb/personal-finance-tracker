import pdfplumber
import re
import pandas as pd
from tqdm.notebook import tqdm_notebook

from src.modules.helper_fns import *

def extract_features_from_pdf(pdf_file, x_tolerance=2, init_y_top=440, reg_y_top=210):

    parent_dir = os.path.dirname(pdf_file)
    dir_before_file = os.path.basename(parent_dir)
    if dir_before_file == "Chequing":
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

def __grab_pattern(account_name):
    if account_name == "Credit":
        return r'(\d{3})\s+(\w{3}\s+\d{1,2})\s+(\w{3}\s+\d{1,2})\s+(.+)\s+(\d+.\d{2})'
    elif account_name == "Chequing":
        return r'(\w{3}\s+\d{1,2})\s+([\w\s-]+)\s+([\d,]+\.\d+)(?:\s+([\d,]+\.\d+))?'


def process_transactions_from_lines(pdf_lines, account_name):

    transactions = []

    pattern = __grab_pattern(account_name)

    for i, line in enumerate(pdf_lines):
        match = re.match(pattern, line)
        # print(line)
        if match:
            match_groups = list(match.groups())
            if "Credit" in account_name:
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
            elif account_name == "Chequing":
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

def generate_fin_df(account_names=None):
    overall_df = pd.DataFrame()
    if account_names is None:
        # account_names = read_all_account_folder_names()
        account_names = ['Chequing', 'Credit']
    for account_name in account_names:
        pdf_files = read_all_files(account_name)
        for pdf_file in tqdm_notebook(pdf_files, desc=f"Reading PDFs from '{account_name}' bucket"):
            pdf_file_atts = grab_pdf_name_attributes(pdf_file)
            pdf_file_path = process_import_path(pdf_file, account_name)
            lines = extract_features_from_pdf(pdf_file_path, x_tolerance=2, init_y_top=440, reg_y_top=210)
            transactions= process_transactions_from_lines(lines, account_name)
            temp_df = pd.DataFrame(transactions)
            temp_df[['Statement Year']] = pdf_file_atts['year']
            temp_df[['Statement Month']] = pdf_file_atts['month']
            temp_df[['Account']] = account_name
            overall_df = pd.concat([temp_df, overall_df], ignore_index=True)

    overall_df['Transaction Year'] = overall_df.apply(__calculate_transaction_year, axis=1)
    overall_df['DateTime'] = overall_df['Transaction Date'] + ' ' + overall_df['Transaction Year'].astype(str)
    overall_df['DateTime'] = pd.to_datetime(overall_df['DateTime'])
    return overall_df