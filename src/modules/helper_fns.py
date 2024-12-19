import re
import calendar
import os

def __extract_month(input_string):
    months = '|'.join(f'.*{m}.*' for m in calendar.month_abbr[1:])
    month_match = re.match(f'.*({months}).*', input_string, re.IGNORECASE)
    if month_match:
        matched_text = month_match.group(1)
        for i, abbr in enumerate(calendar.month_abbr[1:], 1):
            if abbr.lower() in matched_text.lower():
                return calendar.month_name[i]

def grab_pdf_name_attributes(pdf_name):
    pdf_name = pdf_name.replace('.pdf', '')
    str_atts = re.split(r'[ _]', pdf_name)\
    
    result = {}
    for str_attribute in str_atts:
        month_extracted = __extract_month(str_attribute)
        if month_extracted is not None:
            result['month'] = month_extracted
        elif re.match(r'^\d{4}$', str_attribute) and 1000 <= int(str_attribute) <= 9999:
            result['year'] = str_attribute

    return result

def __process_export_cache_path(pdf_file, parent_account_name):
    pdf_atts = grab_pdf_name_attributes(pdf_file)
    full_path = "../../bank_statements/cached_data/" + parent_account_name + pdf_atts["month"] + "_" + pdf_atts['year'] + ".csv"
    return full_path

def process_import_path(pdf_file, parent_account_name):
    return "../../bank_statements/" + parent_account_name + "/" + pdf_file

def read_all_account_folder_names():
    return os.listdir("../../bank_statements/")

def read_all_files(account_name):
    pdf_files = os.listdir(f"../../bank_statements/{account_name}/")
    return [f for f in pdf_files if f.endswith('.pdf')]