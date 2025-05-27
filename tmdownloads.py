import numpy as np
import streamlit as st
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
from datetime import timedelta
import re

st.set_page_config(layout="wide", page_title='Trunk Main Analysis', initial_sidebar_state="expanded", page_icon="💧")
st.subheader('Trunk Main Analysis')
st.markdown('---' * 20)

tab2, tab3, tab4, tab5 = st.tabs([
    'Trunk Main Balance Calculation - Import',
    'Trunk Main Balance Calculation - All',
    'Predictive Modelling',
    'Chart analysis'])

# Load region mapping file
@st.cache_data
def load_region_map(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['File'] = df['TMA Ref'].astype(str) + '.csv'
        return dict(zip(df['File'], df['Region'])), df
    return {}, pd.DataFrame()

region_dict, region_df = load_region_map('my_tm_region.csv')

@st.cache_data
def load_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        return None

with tab2:
    data_dir = 'Trunkmain data'
    if not os.path.exists(data_dir):
        st.error(f"Directory '{data_dir}' not found. Please ensure the data directory exists.")
        st.stop()

    file_data_stored = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not file_data_stored:
        st.warning(f"No CSV files found in '{data_dir}' directory.")
        st.stop()

    st.sidebar.subheader("Analysis Parameters")

    # Region filtering
    if not region_df.empty:
        all_regions = sorted(region_df['Region'].unique())
        selected_regions = st.sidebar.multiselect("Filter by Region", all_regions, default=all_regions)
        file_data_stored = [f for f in file_data_stored if region_dict.get(f) in selected_regions]

    # Determine default date range
    sample_file = load_csv(os.path.join(data_dir, file_data_stored[0]))
    if sample_file is not None and 'Date of Reading' in sample_file.columns:
        sample_file['Date of Reading'] = pd.to_datetime(sample_file['Date of Reading'], dayfirst=True, errors='coerce')
        sample_dates = sample_file['Date of Reading'].dropna()
        default_start = sample_dates.min().date()
        default_end = sample_dates.max().date()
    else:
        default_start = datetime.date.today() - timedelta(days=365)
        default_end = datetime.date.today()

    # Sidebar filters
    start_date = pd.to_datetime(st.sidebar.date_input("Filter Start Date", value=default_start))
    end_date = pd.to_datetime(st.sidebar.date_input("Filter End Date", value=default_end))
    interval_days = st.sidebar.slider("Select Interval (Days)", min_value=30, max_value=720, step=30, value=30)

    computation_level = st.sidebar.radio(
        ':rainbow[Choose Computation Level]',
        ['Base Computation', 'Force Computation']
    )

    st.write(f':rainbow[Computation Type: {computation_level}]')
    st.write(f"Processing {len(file_data_stored)} files...")

    # Begin processing
    tm_calc_all_rows = []
    progress = st.progress(0)

    for idx, filename in enumerate(file_data_stored):
        df = load_csv(os.path.join(data_dir, filename))
        if df is None or 'Date of Reading' not in df.columns:
            continue

        df['Date of Reading'] = pd.to_datetime(df['Date of Reading'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date of Reading'])
        df = df[(df['Date of Reading'] >= start_date) & (df['Date of Reading'] <= end_date)]

        if df.empty:
            continue

        sub_tm_name = filename.replace('.csv', '')
        meter_cols = [
            col for col in df.columns
            if any(x in col.lower() for x in ['in', 'out', 'cus']) and 'balance' not in col.lower()
        ]
        if not meter_cols:
            continue

        summary = df[meter_cols].describe().T
        inflow_raw = summary[summary.index.str.contains('(in)', case=False)]['mean']
        outflow_raw = summary[
            summary.index.str.contains('cus', case=False) | summary.index.str.contains('(out)', case=False)
            ]['mean']

        if computation_level == 'Base Computation' and (inflow_raw.isna().any() or outflow_raw.isna().any()):
            balance = 'Meter is not working'
        else:
            # Identify inflow/outflow meter groups
            inflow_raw = summary[summary.index.str.contains('(in)', case=False)]['mean']
            outflow_raw = summary[
                summary.index.str.contains('cus', case=False) | summary.index.str.contains('out', case=False)
                ]['mean']

         #   Check for missing data BEFORE dropping NaNs
        if computation_level == 'Base Computation' and (inflow_raw.isna().any() or outflow_raw.isna().any()):
            balance = 'Meter is not working'
        else:
            # Clean + Calculate if allowed
            inflow = pd.to_numeric(inflow_raw, errors='coerce').dropna()
            outflow = pd.to_numeric(outflow_raw, errors='coerce').dropna() * -1
            balance = round(inflow.sum() + outflow.sum(),
                            3) if not inflow.empty and not outflow.empty else 'Meter is not working'

        tm_calc_all_rows.append({
            'scheme_name': sub_tm_name,
            'Balance': balance
        })

        progress.progress((idx + 1) / len(file_data_stored))

    tm_calc_all = pd.DataFrame(tm_calc_all_rows)

    # Display results
    if not tm_calc_all.empty:
        st.markdown('## 📊 Consolidated Results - All Trunk Mains')
        st.markdown(f'#### Summary of {len(tm_calc_all)} Trunk Main(s)')
        st.dataframe(tm_calc_all, use_container_width=True)

        numeric_balances = pd.to_numeric(tm_calc_all['Balance'], errors='coerce')
        working_count = (~numeric_balances.isna()).sum()
        total_count = len(tm_calc_all)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Trunk Mains", total_count)
        col2.metric("Working Trunk Main", working_count)
        col3.metric("Issues Detected", total_count - working_count)

        my_summarised = tm_calc_all[tm_calc_all['Balance'] != 'Meter is not working']
    else:
        my_summarised = pd.DataFrame()
        st.warning("No data was successfully processed from any files.")

with tab3:
    st.write('TM CHARTING')
    if not my_summarised.empty:
        my_summarised['Balance'] = pd.to_numeric(my_summarised['Balance'], errors='coerce')
        chart_data = my_summarised.sort_values(by='Balance', ascending=False)

        st.subheader('Bar Chart of Trunk Main Balances')
        st.bar_chart(chart_data.set_index('scheme_name')['Balance'])

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(chart_data['scheme_name'], chart_data['Balance'])
        ax.set_title('Trunk Main Balance by Scheme')
        ax.set_xlabel('Scheme Name')
        ax.set_ylabel('Balance')
        ax.tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        st.pyplot(fig)

        # CV Reliability Section
        st.markdown("---")
        st.subheader("🌐 Top 20 Most Reliable Trunk Mains by CV")

        reliability = []
        valid_tm_names = my_summarised['scheme_name'].tolist()

        for file in file_data_stored:
            tm_name = file.replace('.csv', '')
            if tm_name not in valid_tm_names:
                continue
            df = load_csv(os.path.join(data_dir, file))
            if df is not None:
                meter_cols = [col for col in df.columns if any(x in col.upper() for x in ['(CUS)', '(IN)', '(OUT)'])]
                if not meter_cols:
                    continue
                df_meters = df[meter_cols].copy()
                stats = df_meters.describe().T
                stats['Coefficient of Variation (%)'] = (stats['std'] / stats['mean']) * 100
                avg_cv = stats['Coefficient of Variation (%)'].mean()
                reliability.append({
                    'Trunk Main': tm_name,
                    'Average CV (%)': avg_cv,
                    'Region': region_dict.get(file, 'Unknown')
                })

        reliability_df = pd.DataFrame(reliability).sort_values(by='Average CV (%)').head(30)
        styled = reliability_df.style.background_gradient(subset=['Average CV (%)'], cmap='Greens')
        st.dataframe(styled, use_container_width=True)

        st.subheader("🌍 Visualisation of CV Reliability")
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        bars = ax2.barh(reliability_df['Trunk Main'], reliability_df['Average CV (%)'], color='seagreen')
        ax2.set_xlabel('Average Coefficient of Variation (%)')
        ax2.set_title('Top 20 Most Reliable Trunk Mains')
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%', va='center')
        st.pyplot(fig2)
    else:
        st.warning("No valid balances available to chart.")
    #########################################################################################
    # if not my_summarised.empty:
    #     my_summarised['Balance'] = pd.to_numeric(my_summarised['Balance'], errors='coerce')
    #     chart_data = my_summarised.sort_values(by='Balance', ascending=False)
    #
    #     st.subheader('Bar Chart of Trunk Main Balances')
    #     st.bar_chart(chart_data.set_index('scheme_name')['Balance'])
    #
    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     bars = ax.bar(chart_data['scheme_name'], chart_data['Balance'])
    #     ax.set_title('Trunk Main Balance by Scheme')
    #     ax.set_xlabel('Scheme Name')
    #     ax.set_ylabel('Balance')
    #     ax.tick_params(axis='x', rotation=45)
    #     for bar in bars:
    #         height = bar.get_height()
    #         ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
    #                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    #     st.pyplot(fig)
    #
    #     st.markdown("---")
    #     st.subheader("🌐 Top 20 Most Reliable Trunk Mains by CV")
    #
    #     reliability = []
    #     for file in file_data_stored:
    #         df = load_csv(os.path.join(data_dir, file))
    #         if df is not None:
    #             meter_cols = [col for col in df.columns if any(x in col.upper() for x in ['(CUS)', '(IN)', '(OUT)'])]
    #             df = df[meter_cols]
    #             stats = df.describe().T
    #             stats['Coefficient of Variation (%)'] = (stats['std'] / stats['mean']) * 100
    #             avg_cv = stats['Coefficient of Variation (%)'].mean()
    #             reliability.append({
    #                 'Trunk Main': file.replace('.csv', ''),
    #                 'Average CV (%)': avg_cv,
    #                 'Region': region_dict.get(file, 'Unknown')
    #             })
    #
    #     reliability_df = pd.DataFrame(reliability).sort_values(by='Average CV (%)').head(20)
    #     styled = reliability_df.style.background_gradient(subset=['Average CV (%)'], cmap='Greens')
    #     st.dataframe(styled, use_container_width=True)
    #
    #     st.subheader("🌍 Visualisation of CV Reliability")
    #     fig2, ax2 = plt.subplots(figsize=(10, 8))
    #     bars = ax2.barh(reliability_df['Trunk Main'], reliability_df['Average CV (%)'], color='seagreen')
    #     ax2.set_xlabel('Average Coefficient of Variation (%)')
    #     ax2.set_title('Top 20 Most Reliable Trunk Mains')
    #     for bar in bars:
    #         width = bar.get_width()
    #         ax2.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center')
    #     st.pyplot(fig2)
    # else:
    #     st.warning("No valid balances available to chart.")
