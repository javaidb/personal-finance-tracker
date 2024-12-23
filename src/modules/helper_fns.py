import re
import calendar
import os
import pandas as pd

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

def plot_attribute_against_datetime(
        df_in, 
        df_col='Balance',
        day_span={
            'moving_average': 7,
            'rate_of_change': 7
        }):

    moving_avg_col = f"moving_average_{day_span['moving_average']}D"
    rate_of_change_col = f"rate_of_change_{day_span['rate_of_change']}D"


    df = df_in.copy()
    df = df[df[[df_col]].notnull().all(axis=1)]
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values(by='DateTime')
    df.set_index('DateTime', inplace=True)
    df[moving_avg_col] = df[df_col].rolling(window=f"{day_span['moving_average']}D", min_periods=1).mean()
    df[f"smoothed_{moving_avg_col}"] = gaussian_filter1d(df[moving_avg_col], sigma=10)

    df[rate_of_change_col] = df[f"smoothed_{moving_avg_col}"].rolling(f"{day_span['rate_of_change']}D").apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
    df[rate_of_change_col] = gaussian_filter1d(df[rate_of_change_col], sigma=20)
    df.reset_index(inplace=True)

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Original Data', f"{day_span['rate_of_change']}-day Rate of Change"),vertical_spacing=0.1,
                        shared_xaxes=True)
    fig.add_trace(go.Scatter(x=df['DateTime'], y=df[df_col], mode='markers', name=df_col), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['DateTime'], y=df[f"smoothed_{moving_avg_col}"], mode='lines', name=f"{day_span['moving_average']}-day Moving Avg"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['DateTime'], y=df[rate_of_change_col], mode='lines', name=f"{day_span['rate_of_change']}-day Rate of Change"), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df['DateTime'],
        y=[max(0, y) for y in df[rate_of_change_col]],
        fill='tozeroy',
        fillcolor='rgba(0, 255, 0, 0.1)',
        line=dict(width=0),
        showlegend=False
    ), row=2, col=1)

    # Add shaded area for negative values (not in legend)
    fig.add_trace(go.Scatter(
        x=df['DateTime'],
        y=[min(0, y) for y in df[rate_of_change_col]],
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(width=0),
        showlegend=False
    ), row=2, col=1)

    fig.update_layout(
        title=f'{df_col} Over Time',
        xaxis_title='Date',
        yaxis_title=df_col,
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