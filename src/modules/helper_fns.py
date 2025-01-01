import re
import calendar
import os
import pandas as pd
import numpy as np
import ruptures as rpt
import json
from datetime import datetime, timezone

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.colors import qualitative

from scipy.ndimage import gaussian_filter1d

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

def __detect_change_points(y, penalty=10, min_size=10, jump=5, model="rbf"):
    # Convert y to a numpy array if it's not already
    y_array = np.array(y).reshape(-1, 1)  # reshape to 2D array
    algo = rpt.Pelt(model=model, jump=jump, min_size=min_size).fit(y_array)
    change_points = algo.predict(pen=penalty)
    return change_points

def __calculate_r2(y_true, y_pred):
    """Calculate R-squared"""
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

def __linearize_segments(numeric_time, y, change_points, min_segment_size=2):
    numeric_time = np.array(numeric_time)
    y = np.array(y)
    segments = []
    change_points = [0] + change_points + [len(y)]
    for start, end in zip(change_points, change_points[1:]):
        if end - start >= min_segment_size:
            
            x_segment = numeric_time[start:end]
            y_segment = y[start:end]
            coeffs = np.polyfit(x_segment, y_segment, 1)

            y_pred = np.polyval(coeffs, x_segment)
            r2 = __calculate_r2(y_segment, y_pred)

            segments.append((start, end, coeffs, r2))
    return segments

def update_json(imported_json, updated_entries):
    for update_category, update_keyword_patterns in updated_entries['categories'].items():
        current_category_patterns = imported_json['categories'][update_category]['patterns']
        current_keyword_list = [x['terms'] for x in current_category_patterns]
        for update_keyword_term in update_keyword_patterns:
            if update_keyword_term['terms'] not in current_keyword_list:
                current_category_patterns.extend(update_keyword_term)

def export_json(updated_json):
    json_file_path = '../../cached_data/databank.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(updated_json, json_file, indent=2)

def reset_json_matches(imported_json):
    for category, keyword_patterns in imported_json.get('categories').items():
        imported_json['categories'][category]['totalMatches'] = 0
        for pattern_ind, _ in enumerate(keyword_patterns.get('patterns')):
            imported_json['categories'][category]['patterns'][pattern_ind]['matchCount'] = 0
    return imported_json

def import_json():
    with open('../../cached_data/databank.json', 'r') as file:
        return json.load(file)

def add_entry_to_json(entry:list, category:str, json_dict:dict):
    current_category_patterns = json_dict.setdefault('categories', {}).setdefault(category, {}).setdefault('patterns',[])
    current_keyword_list = [x['terms'] for x in current_category_patterns]
    if not any(entry == keyword for keyword in current_keyword_list):
        current_time = datetime.now(timezone.utc)
        iso_time = current_time.isoformat(timespec='microseconds').replace('+00:00', 'Z')
        current_category_patterns.append({
            "terms": entry,
            "dateAdded": iso_time,
            "lastUpdated": iso_time,
        })
    else:
        print(f"Entry '{entry}' already exists in category '{category}', skipping.")

def plot_attribute_against_datetime(
    df_in,
    segment_settings={
        'penalty': 0.8,
        'min_size': 55,
        'jump': 50,
        'model': "l2"
    },
    view_segments = False):
    
    # ============================== INITIALIZATIONS FOR PLOTTING ==============================

    df_col_for_balance = 'Balance'
    concat_coeffs_col = "Regional Rates of Change (PELT)"
    segmented_col = "Linear Region Segmentations (PELT)"
    
    # ==========================================================================================

    df = df_in.copy()
    df = df[df[[df_col_for_balance]].notnull().all(axis=1)]
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values(by='DateTime')
    df.set_index('DateTime', inplace=True)

    df.reset_index(inplace=True)

    # ============================== REGION IDENTIFICATION/ISOLATION ==============================

    df['numeric_time'] = (df['DateTime'] - df['DateTime'].min()).dt.total_seconds()

    change_points = __detect_change_points(
        df['Balance'].values,
        penalty = segment_settings['penalty'],
        min_size = segment_settings['min_size'],
        jump = segment_settings['jump'],
        model = segment_settings['model']
    )
    segments = __linearize_segments(df['numeric_time'].values, df['Balance'].values, change_points)
    concat_y_segments = []
    concat_coeffs = []
    for start, end, coeffs, _ in segments:
        # print(_)
        concat_y_segments.extend(list(np.polyval(coeffs, df['numeric_time'][start:end])))
        concat_coeffs.extend([coeffs[0]*86400*7]*(end-start))

    # ==========================================================================================

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Accumulated Balance across Datetimes', "Rate of Change via segmentation by PELT algorithm"),vertical_spacing=0.1,
                        shared_xaxes=True)

    fig.add_trace(go.Scatter(
        x=df['DateTime'], 
        y=df[df_col_for_balance], 
        mode='markers', 
        marker=dict(color='#BB2525', size=6),
        name=df_col_for_balance), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['DateTime'], 
        y=concat_coeffs,
        mode='lines',
        line=dict(color='#0C0C54', width=2),
        name=concat_coeffs_col), row=2, col=1)

    if view_segments:
        fig.add_trace(go.Scatter(
            x=df['DateTime'],
            y=concat_y_segments,
            mode='lines',
            name=segmented_col,
            line=dict(color='#BCBF07', width=3)
        ), row=1, col=1)

        for cp in change_points:
            if cp < len(df):
                fig.add_vline(x=df['DateTime'].iloc[cp], line_dash="dash", line_color="green", row=1, col=1)


    fig.add_trace(go.Scatter(
        x=df['DateTime'],
        y=[max(0, y) for y in concat_coeffs],
        fill='tozeroy',
        fillcolor='rgba(0, 255, 0, 0.1)',
        line=dict(width=0),
        showlegend=False
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df['DateTime'],
        y=[min(0, y) for y in concat_coeffs],
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(width=0),
        showlegend=False
    ), row=2, col=1)

    fig.update_layout(
        title=f'{df_col_for_balance} Over Time',
        xaxis_title='Date',
        yaxis_title=df_col_for_balance,
        height=800,
    )
    fig.show()
    return

def plot_stacked(df_in):
    df = df_in.copy()
    timeframe = 'Month'
    df[timeframe] = pd.to_datetime(df['DateTime']).dt.to_period('M').astype(str)

    monthly_expenditure = df.groupby([timeframe, 'Classification'])['Amount'].sum().reset_index()
    
    extended_colors = qualitative.Plotly + qualitative.D3
    fig = px.bar(monthly_expenditure, 
                x=timeframe, 
                y='Amount', 
                color='Classification',
                color_discrete_sequence=extended_colors,
                title='Monthly Expenditure by Category',
                labels={'Amount': 'Total Expenditure', timeframe: timeframe},
                barmode='stack')

    fig.update_layout(
        barmode='relative',  # Stacks positive above and negative below the x-axis
        xaxis_title=timeframe,
        yaxis_title='Total Expenditure',
        legend_title='Classification',
        bargap=0.1
    )

    fig.show()