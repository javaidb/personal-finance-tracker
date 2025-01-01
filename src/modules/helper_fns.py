import re
import calendar
import os
import pandas as pd
import numpy as np
import ruptures as rpt
import json
from datetime import datetime, timezone

import ipywidgets as widgets
from IPython.display import display

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.colors import qualitative

# from scipy.ndimage import gaussian_filter1d

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

def update_json(imported_json, candidate_word_updates, debug=False):
    for candidate_pattern in candidate_word_updates:
        update_category = candidate_pattern["category"]
        current_category_patterns = imported_json[update_category]['patterns']
        current_keyword_list = [x['terms'] for x in current_category_patterns]
        
        if debug:
            print(f"update_keyword_term: {candidate_pattern}")
            print(f"current_keyword_list: {current_keyword_list}")
        if candidate_pattern['terms'] not in current_keyword_list:
            # print("found!")
            filtered_keyword_term = {k: v for k, v in candidate_pattern.items() if k not in ['index', 'category']}
            current_category_patterns.append(filtered_keyword_term)
        if debug: print(f"new: {imported_json[update_category]['patterns']}")
    return imported_json

def export_json(updated_json, print_statement=False):
    json_file_path = '../../cached_data/databank.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(updated_json, json_file, indent=2)
    if print_statement: print(f"Exported updated JSON to '{json_file_path}'.")

def reset_json_matches(imported_json):
    for category, keyword_patterns in imported_json.items():
        imported_json[category]['totalMatches'] = 0
        for pattern_ind, _ in enumerate(keyword_patterns.get('patterns')):
            imported_json[category]['patterns'][pattern_ind]['matchCount'] = 0
    return imported_json

def import_json():
    json_file_path = '../../cached_data/databank.json'
    with open(json_file_path, 'r') as file:
        return json.load(file)

def add_entry_to_json(index_classifier:int, entry:list, category:str, candidate_cache:list, debug=False):
    if debug: print(f"candidate_cache: {candidate_cache}")
    # current_keyword_list = [x['terms'] for x in candidate_cache]
    # print(f"current_keyword_list: {current_keyword_list}")
    locate_next_matching_dict = any(d.get('terms') == entry and d.get('category') == category for d in candidate_cache)
    # next(keyword == entry for keyword in current_keyword_list)
    if debug: print(f"locate_next_matching_dict: {locate_next_matching_dict}")
    if locate_next_matching_dict is not None:
        current_time = datetime.now(timezone.utc)
        iso_time = current_time.isoformat(timespec='microseconds').replace('+00:00', 'Z')
        for candidate_word_update in candidate_cache:
            if candidate_word_update.get("index") == index_classifier:
                candidate_word_update.update({
                    "terms": entry,
                    "dateAdded": iso_time,
                    "lastUpdated": iso_time,
                    "index": index_classifier,
                    "category": category,
                    "matchCount": 1
                })
            return
        candidate_cache.append({
            "terms": entry,
            "dateAdded": iso_time,
            "lastUpdated": iso_time,
            "index": index_classifier,
            "category": category,
            "matchCount": 1
        })
        return
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


class CategoryUpdater:
    def __init__(self, df, debug=False, words_col='Processed Details', categories_col='Classification', matched_words_col='Matched Keyword'):
        self.df = df.copy()
        self.debug = debug
        self.categories = import_json().get('categories')
        self.words_col = words_col
        self.categories_col = categories_col
        self.matched_words_col = matched_words_col
        self.current_row = 0
        self.filtered_df = self.df.copy()
        self.candidate_cache_for_updates = []
        self.update_pie_chart(setup=True)
        self.setup_widgets()
        
    def setup_widgets(self):
        # print(self.categories)
        self.keyword_options_from_json = widgets.SelectMultiple(
            options=[],
            description='Words:',
            disabled=False
        )
        
        self.matched_words_text = widgets.Textarea(
            description='Matched Words:',
            disabled=False,
            layout=widgets.Layout(width='300px', height='100px')
        )
        
        self.category_select = widgets.Dropdown(
            options=list(self.categories.keys()),
            description='Category:',
            disabled=False,
        )
        self.category_select.observe(self.on_category_change, names='value')
        
        self.filter_select = widgets.Dropdown(
            options=['All'] + list(self.categories.keys()),
            description='Filter:',
            value='All'
        )
        self.filter_select.observe(self.on_filter_change, names='value')
        
        self.prev_button = widgets.Button(description="Previous")
        self.prev_button.on_click(self.prev_row)
        
        self.next_button = widgets.Button(description="Next")
        self.next_button.on_click(self.next_row)
        
        self.save_button_local = widgets.Button(
            description=" (local)",
            icon='save',
            layout=widgets.Layout(width='auto', height='auto'),
            style=dict(
                button_color='#E0E0E0',
                font_weight='normal'
        ))
        self.save_button_local.on_click(self.update_local_df)
        
        self.push_to_json_button = widgets.Button(
            description=" (JSON)",
            icon='arrow-up',
            layout=widgets.Layout(width='auto', height='auto'),
            style=dict(
                button_color='#E0E0E0',
                font_weight='normal'
        ))
        self.push_to_json_button.on_click(self.push_to_json)
        
        self.status_label = widgets.Label(value="")
        
        # self.pie_output = widgets.Output()

        # widgets.HBox([widgets.VBox([update_button, output]), info_widget])

        user_entry_content = widgets.VBox([
            self.filter_select, self.keyword_options_from_json, self.category_select, self.matched_words_text,
            widgets.HBox([self.prev_button, self.next_button, self.save_button_local]), 
            widgets.HBox([self.push_to_json_button]),
            self.status_label
        ])

        all_content = widgets.HBox([user_entry_content, self.fig])
        
        display(all_content)
        self.load_current_row()

    def update_local_df(self, b):
        """Save the edited matched words to a JSON file."""

        match_words_entered_textbox = [word.strip() for word in re.split(r'[,\s]+', self.matched_words_text.value)]
        original_index_prior_to_filter = self.filtered_df.index[self.current_row]

        keyword_options = [s.lower() for s in self.keyword_options_from_json.options]

        if self.debug:
            print(f"keyword_options: {keyword_options}")
            print(f"match_words_entered_textbox: {match_words_entered_textbox}")

        if all(any(entered_match.lower() in keyword_option for keyword_option in keyword_options) for entered_match in match_words_entered_textbox):
            if all(word != '' for word in match_words_entered_textbox):
                self.filtered_df.loc[original_index_prior_to_filter, self.matched_words_col] = self.matched_words_text.value
                
                add_entry_to_json(original_index_prior_to_filter, match_words_entered_textbox, self.category_select.value, self.candidate_cache_for_updates, self.debug)
                
                # print(self.candidate_cache_for_updates)

                self.status_label.value = f"Saved '{self.matched_words_text.value}' under '{self.category_select.value}' to row {original_index_prior_to_filter} of dataframe (not yet pushed to JSON)."
            else:
                self.status_label.value = f"{match_words_entered_textbox} is an invalid entry, cannot be empty, please try another entry."
        else:
            self.status_label.value = f"{match_words_entered_textbox} is an invalid entry, unable to locate substring for match in above word associations, please try another entry."
    
    def update_pie_chart(self, setup=False):
        
        status_counts = self.df['Classification'].value_counts(dropna=False)
        none_count = status_counts.get('uncharacterized', 0)
        other_count = len(self.df) - none_count
        
        labels = ['Uncharacterized', 'Characterized']
        values = [none_count, other_count]

        if setup:
            self.fig = go.FigureWidget(data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    textinfo='label+percent+text',
                    # textposition='inside',
                    insidetextorientation='radial',
                    marker=dict(colors=["#2F3136", "DCDDDE"]),
                    showlegend=False,
                    hole=0.3
            )])
            self.fig.update_layout(
                # title='Status Distribution',
                height=300,
                width=600,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title='Count',
                yaxis_title='Status'
            )

        with self.fig.batch_update():
            self.fig.data[0].values = values

    def load_current_row(self, retain_word_entry=False):
        if 0 <= self.current_row < len(self.filtered_df):
            words = self.filtered_df.iloc[self.current_row][self.words_col]
            category = self.filtered_df.iloc[self.current_row][self.categories_col]
            matched_words = self.filtered_df.iloc[self.current_row][self.matched_words_col]
            # print(matched_words)
            self.keyword_options_from_json.options = words
            if not retain_word_entry:
                if matched_words is None: matched_words = ''
                self.matched_words_text.value = matched_words
            self.category_select.value = category
            original_index = self.filtered_df.index[self.current_row]
            self.status_label.value = f"Row {original_index + 1} of {len(self.df)} - Current category: {category}"
            
            self.prev_button.disabled = (self.current_row == 0)
            self.next_button.disabled = (self.current_row == len(self.filtered_df) - 1)
        else:
            self.status_label.value = "No rows to display"
            self.prev_button.disabled = True
            self.next_button.disabled = True

        self.update_pie_chart()
            
    def on_category_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            new_category = change['new']
            original_index = self.filtered_df.index[self.current_row]
            old_category = self.df.at[original_index, self.categories_col]
            
            if new_category != old_category:
                self.df.at[original_index, self.categories_col] = new_category
                self.filtered_df.at[original_index, self.categories_col] = new_category
                
                ## Update JSON file
                # words = self.df.iloc[original_index][self.words_col]
                # self.update_json_category(words, old_category, new_category)
                
                print(f"Updated row {original_index + 1} to category: {new_category}")
                self.load_current_row(retain_word_entry=True)
        
    def push_to_json(self, new_updates):

        new_categories = self.candidate_cache_for_updates

        # Update new entries per category as per new_updates in imported_categories
        updated_json = update_json(self.categories, new_categories, self.debug)
        if self.debug:
            print(f"cache_to_update: {new_categories}")
            print(f"updated_json: {updated_json}")

        # Save updated categories to JSON file
        updated_json_with_outcol = {'categories': updated_json}
        export_json(updated_json_with_outcol, print_statement=True)
        
    def on_filter_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            filter_category = change['new']
            if filter_category == 'All':
                self.filtered_df = self.df.copy()
            else:
                self.filtered_df = self.df[self.df[self.categories_col] == filter_category].copy()
            self.current_row = 0
            self.load_current_row()
        
    def prev_row(self, b):
        self.current_row = max(0, self.current_row - 1)
        self.load_current_row()
        
    def next_row(self, b):
        self.current_row = min(len(self.filtered_df) - 1, self.current_row + 1)
        self.load_current_row()