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
from utils import *

def run_custom_process():
    ### DEFINE DATAPATH ###
    data_path = 'data'

    ### LIST ALL FILES IN DATAPATH ###
    data_df = list_files_and_make_df(data_path)
    print(f"List of All Files...")
    print(tabulate(data_df, headers='keys', tablefmt='fancy_grid'))

    ### TYPE THE ROW NUMBER OF THE FILE TO PROCESS ###
    print(f"Type the row number of the file to process seperated by comma")
    print(f"Example: 0,1,2,3 or in a mixed format such as: 0-3,7-10,12-13,15,16")
    print(f"You can also type 'all' to process all files")
    user_input = input()
    print(f"===================== \n")
    if user_input == 'all':
        list_row_to_process = list(range(len(data_df)))
    else:
        list_row_to_process = parse_input(user_input)
    data_df = data_df.iloc[list_row_to_process]
    print(f"List of All Files...")
    print(tabulate(data_df, headers='keys', tablefmt='fancy_grid'))

    ### FILTER BY SUBJECT ID
    print(f"Type the subject id that you want to process separated by comma")
    print(f"Example: 0,1,2,3 or in a mixed format such as: 0-3,7-10,12-13,15,16")
    print(f"You can also type 'all' to process all subject id")
    user_input = input()
    print(f"===================== \n")
    if user_input != 'all':
        list_subject_id = parse_input(user_input)
        list_subject_id = [str(x) for x in list_subject_id]
        data_df = data_df[data_df['subject'].isin(list_subject_id)]
    print(f"Current list of files to process...")
    print(tabulate(data_df, headers='keys', tablefmt='fancy_grid'))

    ### FILTER 1: Noise Type ###
    print(f"Type the noise type that you want to process separated by comma, in lower case")
    print(f"Example: silent, white, brown")
    print(f"You can also type 'all' to process all noise types")
    user_input = input()
    print(f"===================== \n")
    if user_input == 'all':
        list_noise_type = ['silent', 'white', 'brown', 'pink']
    else:
        list_noise_type = parse_input_str(user_input)
        while not all([x in ['silent', 'white', 'brown', 'pink'] for x in list_noise_type]):
            print(f"Invalid input. Please input only 'silent', 'white', 'brown', 'pink', or 'all'.")
            user_input = input()
            if user_input == 'all':
                list_noise_type = ['silent', 'white', 'brown', 'pink']
                break
            list_noise_type = parse_input_str(user_input)
    # convert to upper case for first letter
    list_noise_type = [x[0].upper() + x[1:] for x in list_noise_type]
    data_df = data_df[data_df['noise_type'].isin(list_noise_type)]
    print(f"Current list of files to process...")
    print(tabulate(data_df, headers='keys', tablefmt='fancy_grid'))
    
    ### FILTER 2: Task ###
    print(f"Type the task that you want to process in lower case")
    print(f"Example: rest")
    print(f"You can also type 'all' to process all tasks")
    user_input = input()
    print(f"===================== \n")
    if user_input == 'all':
        list_task = ['rest', 'read']
    else:
        list_task = parse_input_str(user_input)
        while not all([x in ['rest', 'read'] for x in list_task]):
            print(f"Invalid input. Please input only 'rest', 'read', or 'all'.")
            user_input = input()
            if user_input == 'all':
                list_task = ['rest', 'read']
                break
            list_task = parse_input_str(user_input)
    # convert to upper case for first letter
    list_task = [x[0].upper() + x[1:] for x in list_task]
    data_df = data_df[data_df['task'].isin(list_task)]
    print(f"Current list of files to process...")
    print(tabulate(data_df, headers='keys', tablefmt='fancy_grid'))

    ### FILTER 3: Signal Type ###
    print(f"Type the signal type that you want to process in lower case")
    print(f"Example: alpha, beta")
    print(f"You can also type 'all' to process all signal types")
    user_input = input()
    print(f"===================== \n")
    if user_input == 'all':
        list_signal_type = ['alpha', 'beta', 'theta', 'delta', 'gamma']
    else:
        list_signal_type = parse_input_str(user_input)
        while not all([x in ['alpha', 'beta', 'theta', 'delta', 'gamma'] for x in list_signal_type]):
            print(f"Invalid input. Please input only 'alpha', 'beta', 'theta', 'delta', 'gamma', or 'all'.")
            user_input = input()
            if user_input == 'all':
                list_signal_type = ['alpha', 'beta', 'theta', 'delta', 'gamma']
                break
            list_signal_type = parse_input_str(user_input)

    ### FILTER 4: Which channels do you want to use ###
    print(f"Type the channels that you want to use separated by comma (MUST BE CAPITAL LETTER)")
    print(f"Option: 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'")
    print(f"You can also type 'all' to use all channels")
    user_input = input()
    print(f"===================== \n")
    if user_input == 'all':
        list_channel = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    else:
        list_channel = parse_input_str(user_input)
        while not all([x in ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'] for x in list_channel]):
            print(f"Invalid input. Please input only 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', or 'all'.")
            user_input = input()
            if user_input == 'all':
                list_channel = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
                break
            list_channel = parse_input_str(user_input)

    ### FILTER 5: Do you want to save the signal? ###
    print(f"Do you want to save the signal? (y/n)")
    user_input = input()
    print(f"===================== \n")
    if user_input == 'y':
        save_signal = True
    else:
        save_signal = False

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

            for idx, freq_type in enumerate(list_signal_type):
                run_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                results = get_psd_feature(
                    dataframe = signal_df,
                    freq_type = freq_type,
                    return_psd = save_signal,
                    select_channels = list_channel,
                    len_drop = 7680,
                    len_keep = 46080,
                )

                ### EXTRACT VALUE AND APPEND TO CSV ###
                sum_raw_value = results.get('sum_raw', 'N/A')
                avg_raw_value = results.get('avg_raw', 'N/A')
                sum_filtered = results.get('sum_filtered', 'N/A')
                avg_filtered = results.get('avg_filtered', 'N/A')
                rel_pow = results.get('rel_pow', 'N/A')

                ### WRITE TO CSV ###
                writer.writerow([run_time, subject_id, noise_type, task, freq_type, sum_raw_value, avg_raw_value, sum_filtered, avg_filtered, rel_pow])

                ### SAVE SIGNAL ###
                if save_signal:
                    psd_filtered = results.get('psd_filtered', 'N/A')
                    psd_filtered = psd_filtered.T
                    psd_filtered_freqs = results.get('psd_filtered_freqs', 'N/A')
                    psd_filtered = np.insert(psd_filtered, 0, psd_filtered_freqs, axis=1)
                    header = ['freqs'] + list_channel
                    file_name = f"{subject_id}_{noise_type}_{task}_{freq_type}.csv"
                    save_path = "out/signal"
                    np.savetxt(os.path.join(save_path, file_name), psd_filtered, delimiter=',', header=','.join(header), comments=save_path)

                    if idx == 0:
                        psd_raw = results.get('psd_raw', 'N/A')
                        psd_raw = psd_raw.T
                        psd_raw_freqs = results.get('psd_raw_freqs', 'N/A')
                        psd_raw = np.insert(psd_raw, 0, psd_raw_freqs, axis=1)
                        header = ['freqs'] + list_channel
                        file_name = f"{subject_id}_{noise_type}_{task}_raw.csv"
                        save_path = "out/signal"
                        np.savetxt(os.path.join(save_path, file_name), psd_raw, delimiter=',', header=','.join(header), comments=save_path)

        print(f"=====================")
        print(f"Finished processing all files...")
        print(f"Results are saved in run_log.csv")
        print(f"Signal are saved in out/")
        print(f"=====================")
    


    
