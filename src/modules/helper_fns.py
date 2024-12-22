import re
import calendar
import os
import plotly.graph_objects as go

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
    full_path = f"../../cached_data/{parent_account_name}/{pdf_atts['month']}_{pdf_atts['year']}.csv"
    return full_path

def process_import_path(pdf_file, parent_account_type, parent_account_name):
    return f"../../bank_statements/{parent_account_type}/{parent_account_name}/{pdf_file}"

def read_all_account_type_folder_names():
    directory = f"../../bank_statements/"
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def read_all_account_folder_names(account_type):
    directory = f"../../bank_statements/{account_type}/"
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def read_all_files(account_type, account_name):
    pdf_files = os.listdir(f"../../bank_statements/{account_type}/{account_name}/")
    return [f for f in pdf_files if f.endswith('.pdf')]

def sort_df(df):
    df = df.sort_values(by='DateTime')
    return df.groupby('DateTime', group_keys=False).apply(lambda x: x.sort_index(), include_groups=True)

def plot_attribute_against_datetime(df, df_col='Balance'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['DateTime'], y=df[df_col], mode='markers', name=df_col))
    fig.update_layout(
        title=f'{df_col} Over Time',
        xaxis_title='Date',
        yaxis_title=df_col,
    )
    fig.show()
    return