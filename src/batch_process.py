import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tabulate import tabulate
import time
import csv
import sys
import datetime as dt
sys.path.append('src')
from functional import *

def run_batch_process():
    ### DEFINE DATAPATH ###
    data_path = 'data'

    ### LIST ALL FILES IN DATAPATH ###
    data_df = list_files_and_make_df(data_path)
    print(f"List of All Files...")

    ### ASK TO PRINT DATAFRAME ###
    print(f"Would you like to show the list of files? (y/n)")
    user_input = input()
    print(f"=====================")
    if user_input == 'y':
        print(tabulate(data_df, headers='keys', tablefmt='fancy_grid'))
        time.sleep(3)

    ### CONFIRMATION ABOUT RUN_LOG ###
    print(f"Results will be saved in run_log.csv")
    print(f"If you have previously run this program, the results will be appended to run_log.csv")
    print(f"=====================")
    print(f"Press enter to continue...")
    input()

    ### CREATE RUN_LOG ###
    log_csv_path = 'run_log.csv'
    if not os.path.exists(log_csv_path):
        print(f"run_log.csv does not exist. Creating run_log.csv...")
        print(f"=====================")
        # Create the csv file if it does not exist
        with open(log_csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'subject', 'noise_type', 'task', 'freq_type', 'sum_raw', 'avg_raw', 'sum_filtered', 'avg_filtered', 'rel_pow'])

    ### LOOP THROUGH ALL FILES ###
    with open(log_csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        
        if os.stat(log_csv_path).st_size == 0:
            writer.writerow(['run time', 'subject_id', 'noise_type', 'task', 'freq_type', 'sum_raw_value', 'avg_raw_value', 'sum_filtered', 'avg_filtered', 'rel_pow'])

        print(f"Starting to loop through all files...")
        for index, row in data_df.iterrows():
            print(f"Processing {row['path']}...")
            subject_id = str(row['subject'])
            noise_type = str(row['noise_type'])
            task = str(row['task'])
            signal_df = load_mnedf(row['path'])

            for freq_type in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                run_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                results = get_psd_feature(
                    dataframe = signal_df,
                    freq_type = freq_type,
                    return_psd = False
                )

                ### EXTRACT VALUE AND APPEND TO CSV ###
                sum_raw_value = results.get('sum_raw', 'N/A')
                avg_raw_value = results.get('avg_raw', 'N/A')
                sum_filtered = results.get('sum_filtered', 'N/A')
                avg_filtered = results.get('avg_filtered', 'N/A')
                rel_pow = results.get('rel_pow', 'N/A')

                ### WRITE TO CSV ###
                writer.writerow([run_time, subject_id, noise_type, task, freq_type, sum_raw_value, avg_raw_value, sum_filtered, avg_filtered, rel_pow])

        print(f"=====================")
        print(f"Finished processing all files...")