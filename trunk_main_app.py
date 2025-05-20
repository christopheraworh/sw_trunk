# import dis

import numpy as np
import streamlit as st
import pandas as pd
import time
import os
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta


st.set_page_config(layout="wide",
                   page_title='Trunk Main Analysis',
                   initial_sidebar_state="expanded",
                   page_icon="💧")
#########################################
st.subheader('Trunk Main Analysis')
st.markdown('---' * 20)

tab2, tab3, tab4, tab5 = st.tabs([ 'Trunk Main Balance Calculation - Import','Trunk Main Balance Calculation - All', 'Predictive Modelling','Chart analysis',])
with tab2:
    # Upload file
    uploaded_file = st.file_uploader("Upload flow data (.csv)", type=["csv"])

    if uploaded_file is not None:
        progress_bar = st.progress(0)
        try:
            # Read CSV file into a dataframe
            df = pd.read_csv(uploaded_file)
            for i in range(1,101):
                progress_bar.progress(i,f'Loaded {i}%')
                time.sleep(0.01)



            # Clean and parse datetime column
            df['Date of Reading'] = pd.to_datetime(df['Date of Reading'], dayfirst=True, errors='coerce')
            df = df.sort_values('Date of Reading').reset_index(drop=True)


            # Sidebar date filtering
            min_date = df['Date of Reading'].min()
            max_date = df['Date of Reading'].max()


            start_date = st.sidebar.date_input(
                "Filter Start Date",
                min_value=min_date.date(),
                max_value=max_date.date(),
                value=min_date.date(),
            )
            end_date = st.sidebar.date_input(
                "Filter End Date",
                min_value=min_date.date(),
                max_value=max_date.date(),
                value=max_date.date(),
            )

            st.sidebar.markdown('---' * 20)

            # Interval slider in the sidebar
            interval_days = st.sidebar.slider(
                "Select Interval (Days)",
                min_value=30,
                max_value=720,
                step=30,
                value=30,
            )

            # Apply date filter on dataframe
            filtered_df = df[
                (df['Date of Reading'] >= pd.to_datetime(start_date))
                & (df['Date of Reading'] <= pd.to_datetime(end_date))
                ]

            st.write('**Base Balance Calculation**')

            # Create a copy of the filtered DataFrame
            base_df = filtered_df.copy()
            base_df_columns = base_df.columns

            # Select relevant columns (with "in" / "out" or "date")
            col_names_selected = [
                col for col in base_df_columns
                if 'in' in col.lower() or 'out' in col.lower() or 'date' in col.lower()
            ]
            selected_df = base_df[col_names_selected]

            # Parse and extract date column
            date_column = [col for col in selected_df.columns if 'date' in col.lower()]
            if date_column:
                selected_df[date_column[0]] = pd.to_datetime(selected_df[date_column[0]])
                selected_df['date'] = selected_df[date_column[0]].dt.date

            # Initialize variables for summarization
            start_date_filter = pd.to_datetime(start_date)
            summarised_final_result = pd.DataFrame()

            # Summarize data based on intervals
            while start_date_filter <= pd.to_datetime(end_date):
                current_end_date = start_date_filter + timedelta(days=interval_days)
                filtering_data = selected_df[
                    (selected_df['date'] >= start_date_filter.date()) &
                    (selected_df['date'] <= current_end_date.date())
                    ]

                if not filtering_data.empty:
                    sub_data_summarised = filtering_data.describe()
                    mean_result = sub_data_summarised.loc['mean']

                    # Convert summary to a DataFrame
                    summary_df = pd.DataFrame(mean_result).T
                    summary_df['Period'] = f'{start_date_filter.date()} to {current_end_date.date()}'
                    summarised_final_result = pd.concat([summarised_final_result, summary_df])

                start_date_filter = current_end_date + timedelta(days=1)

            # Finalize summarised dataframe
            summarised_final_result = summarised_final_result.reset_index(drop=True)
            summarised_final_result.columns = [
                col.capitalize() if col != 'Period' else col
                for col in summarised_final_result.columns
            ]
            date_column = [i for i in summarised_final_result.columns if 'date' in i.lower()]
            summarised_final_result.drop(date_column, axis=1, inplace=True)


            ######### Calculating summary table
            inflow_column = [ i for i in summarised_final_result.columns if 'in' in i.lower()]
            outflow_column = [ i for i in summarised_final_result.columns if 'out' in i.lower()]

            inflow_balance = summarised_final_result[inflow_column].sum(axis=1)
            outflow_balance = summarised_final_result[outflow_column].sum(axis =1)
            data_balance = inflow_balance - outflow_balance
            period = summarised_final_result['Period']
            balance_df_summarised = pd.DataFrame({'period':period,'Inflow Balance': inflow_balance, 'Outflow Balance': outflow_balance,'Balance':data_balance})
            #####Data scalar Result
            inflow_average = round(inflow_balance.mean(), 2)
            outflow_average = round(outflow_balance.mean(), 2)
            balance_average = round(inflow_average - outflow_average,2)




            ########### Dashboard Setting
            ####### Display inflow, outflow, and balance in Streamlit columns ##########
            border1, border2, border3, border4 = st.columns(4)


            with border1:
                st.subheader('Trunk Main Uploaded')
                st.write(f'#### {uploaded_file.name.strip(".csv")}')

            with border2:
                st.subheader('Inflow (m3/hr)')
                st.write(f'### {inflow_average}')

            with border3:
                st.subheader('Outflow (m3/hr)')
                st.write(f'### {outflow_average}')

            with border4:
                st.subheader('Balance (m3/hr)')
                st.write(f'### {balance_average}')

            st.markdown('---' * 20)
            st.markdown(f'#### Balance Computed on a {interval_days} days')

            st.dataframe(summarised_final_result)

            st.markdown('---' * 20)
            st.markdown(f'#### Inflow, Outflow and Balance Summary')
            st.dataframe(balance_df_summarised)
        except Exception as e:
            st.warning('Error processing the file. Please make sure it contains valid flow meter data.')
            st.error(f'Details: {str(e)}')

with tab3:
    pass
#     st.write(f'#### Simulate Parameters')
#     file_data_stored = os.listdir('Trunkmain data')
#     main_filter = [i.strip('.csv') for i in file_data_stored]
#     # Importing the  Region area for the TM
#     file_region = pd.read_csv('tm_area.csv')
#     file_region['sheet_name'] =  file_region['sheet_name'].apply(lambda x:x.strip('Summary Report'))
#
#     # Storing the region in a dict for later use
#     file_dict = dict(zip(file_region['TMA Ref'], file_region['sheet_name']))
#
#
#     col_filter1, col_filter2, col_filter3 = st.columns(3, border= True)
#
#     with col_filter1:
#         selected_tm = st.multiselect('Select specific Trunk Main', main_filter)
#
#
#
#     st.markdown('---' * 20)
#     # Setting the min and max date
#     # Define the range
#     min_date = datetime.date(2020, 1, 1)
#     max_date = datetime.date(2025, 12, 31)
#
#     with col_filter3:
#         # Slider for date range
#         date_range = st.slider(
#             "Select a date range",
#             min_value=min_date,
#             max_value=max_date,
#             value=(min_date, max_date),
#             format="DD/MM/YYYY"
#         )
#     col4_filter, col5_filter, col6_filter = st.columns(3, border = True)
#     with col4_filter:
#         # Interval slider in the sidebar
#         interval_days_tm = st.slider(
#             "Select Interval (Days) for computation",
#             min_value=30,
#             max_value=720,
#             step=30,
#             value=30,
#         )
#
#     # Reading the file for analysis
# #0. Creating a master dataframe to store the final result
#     tm_calc_all = pd.DataFrame()
# # 1.  Loop through to read individual file
#     for filename in file_data_stored:
#         sub_tm_name = filename.strip('.csv')
#         path_file = os.path.join('Trunkmain data', filename)
#
#         sub_tm_all = pd.read_csv(path_file)
#
#         sub_tm_all_cols = sub_tm_all.columns
#         col_names_selected_tm = [
#             col for col in sub_tm_all_cols
#             if 'in' in col.lower() or 'out' in col.lower() or 'date' in col.lower()]
#         sub_tm_all =  sub_tm_all[col_names_selected_tm]
#
# #2. Identifying the date and calssifying to date time
#         date_tm_column = [col for col in sub_tm_all.columns if 'date' in col.lower()]
#
#         sub_tm_all[date_tm_column[0]] = pd.to_datetime(
#             sub_tm_all[date_tm_column[0]],
#             dayfirst=True,
#             errors='coerce'
#         )
#
# #4.  I want to create a column for just the date
#         sub_tm_all['date'] = sub_tm_all[date_tm_column[0]].dt.date
#         sub_tm_all.drop(date_tm_column[0], axis=1, inplace = True)
#
# #5.  Create a dataframe to store result that would store result from the while loop
#         sub_tm_while = pd.DataFrame()
#
# #6. # Now going through the while loop to create an average for filtered date intervals for teh sub_tm_all and store in the sub_dataframe above the while loop
#         start_date_tm  = date_range[0]
#         end_date_tm = date_range[-1]
#         while start_date_tm <= end_date_tm:
#             current_end_tm = start_date_tm + timedelta(days=interval_days_tm)
#             filtering_data_tm = sub_tm_all[
#                 (sub_tm_all['date'] >= start_date_tm) &
#                 (sub_tm_all['date'] <= current_end_tm)
#                 ]
#
#             if not filtering_data_tm.empty:
#                 mean_result_tm = filtering_data_tm.describe().loc['mean']
#                 summary_df_tm = pd.DataFrame(mean_result_tm).T
#                 summary_df_tm['Period'] = f'{start_date_tm} to {current_end_tm}'
#                 summary_df_tm['Scheme'] = sub_tm_name
#
#                 sub_tm_while = pd.concat([sub_tm_while, summary_df_tm])
#     #
#             start_date_tm = current_end_tm + timedelta(days=1)
# #7. Summarising the tm data for this analysis
#         inflow_column_tm = [i for i in sub_tm_while.columns if 'in' in i.lower()]
#         outflow_column_tm = [i for i in sub_tm_while.columns if 'out' in i.lower()]
#         inflow_balance_tm = sub_tm_while[inflow_column_tm].sum(axis=1)
#         outflow_balance_tm = sub_tm_while[outflow_column_tm].sum(axis=1)
#         data_balance_tm = inflow_balance_tm - outflow_balance_tm
#         period_tm = sub_tm_while['Period']
#         schemes_tm  = sub_tm_while['Scheme']
#         sub_balance_tm = pd.DataFrame({'Scheme':schemes_tm,
#                                        'Period':period_tm,
#                                        'Inflow':inflow_balance_tm,
#                                        'Outflow':outflow_balance_tm,
#                                        'Balance': data_balance_tm})
#         tm_calc_all=pd.concat([tm_calc_all,sub_balance_tm])
#
# # #8. # # Displaying the result of the calculation together with applied filters
#
#     final_df_tm =  tm_calc_all.copy()
#     final_df_tm['Region'] = final_df_tm['Scheme'].map(file_dict)
#     #Rearranging the columns
#     final_df_tm= final_df_tm[['Period','Scheme','Region','Inflow','Outflow','Balance']]
#     with col_filter2:
#         selected_region =st.multiselect('Select specific Region', list(final_df_tm['Region'].unique()))
#
#
#     # Final outlook of the result witrh filters applied
#     final_df_tm = final_df_tm[(final_df_tm['Region'].isin(selected_region))& final_df_tm['Scheme'].isin(selected_tm)]
#
#     # Calculating the average Balance for all the  Schemes
#
#     average_inflow_tm  =  final_df_tm['Inflow'].mean()
#     average_outflow_tm = final_df_tm['Outflow'].mean()
#     average_balance_tm = round(average_inflow_tm - average_outflow_tm,3)
#     final_df_tm = final_df_tm.reset_index()
#     final_df_tm.drop('index', axis =1, inplace = True)
#
#
#
#     with col5_filter:
#         st.write('Balance(m3/hr)')
#         st.write(f'### {average_balance_tm}')
#
#
#     with col6_filter:
#         st.write('No of Schemes Added to model')
#         st.write(f'### {len(main_filter)}')
#
#
#     st.write(f'#### Summary - TMA Balance Computation')
#
# # Computing overall balance
#     st.dataframe(final_df_tm)

############ Creating The Predictive Model for Predicting Unknown Meters
with tab4:
    pass
    # with st.form("my_form"):
    #     st.write("Enter the following Information")
    #
    #     trunk_main_name = st.text_input('Enter the name of the trunk main')
    #     pipe_diameter = st.number_input('Enter the Pipe diameter(mm)')
    #     tm_main_length = st.number_input('Enter the main length for the trunk main')
    #
    #
    #     st.form_submit_button('Submit Information Details for Processing')
    #     ################## Processing the information
    #
    # st.write(f' Trunk Main Name:  {trunk_main_name}')
    # st.write(f' Trunk Main Pipe Length: {pipe_diameter}')
    # st.write(f' Trunk Main Manin Length: {tm_main_length}')



with tab5:
    pass
    # #### Now summarising the already created dataframe to visualise the result
    # grouped_result = final_df_tm.groupby('Scheme')[['Inflow', 'Outflow']].mean()
    # grouped_dataframe = pd.DataFrame(grouped_result).reset_index()
    # grouped_dataframe['Balance'] = grouped_dataframe['Inflow'] - grouped_dataframe['Outflow']
    #
    # # Sort in descending order
    # grouped_dataframe = grouped_dataframe.sort_values(by='Balance', ascending=False)
    #
    #
    #
    # st.subheader('Bar Chart - TMA Balance Computed')
    #
    # import altair as alt
    #
    # # Create bar chart
    # chart = alt.Chart(grouped_dataframe).mark_bar(color="#c61a09").encode(
    #     x=alt.X('Scheme:N', sort='-y'),
    #     y=alt.Y('Balance:Q')
    # ).properties(width=700)
    #
    # # Add data labels
    # text = chart.mark_text(
    #     align='center',
    #     baseline='bottom',
    #     dy=-5
    # ).encode(
    #     text=alt.Text('Balance:Q', format='.2f')
    # )
    #
    # # Combine bar and text
    # st.altair_chart(chart + text, use_container_width=True)
    #


