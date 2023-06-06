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
import json

def run_create_topomap():
    ### OPEN CONFIG ###
    with open('src/config.json', 'r') as f:
        config = json.load(f)

    ### DEFINE FREQUENCY BANDS ###
    if config["edf_config"]["frequency_mode"] == 5:
        freq_names = [d["name"] for d in config["edf_config"]["five_freq_split"]]
        bands = [(freq['low'], freq['high'], freq['name']) for freq in config['edf_config']['five_freq_split']]
    elif config["edf_config"]["frequency_mode"] == 7:
        freq_names = [d["name"] for d in config["edf_config"]["seven_freq_split"]]  
        bands = [(freq['low'], freq['high'], freq['name']) for freq in config['edf_config']['seven_freq_split']]

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
    print(f"Type the noise type that you want to process seperated by comma, in lower case")
    print(f"Example: silent, white, brown")
    print(f"You can also type 'all' to process all noise types")
    user_input = input()
    print(f"===================== \n")
    if user_input == 'all':
        list_noise_type = ['silent', 'white', 'brown', 'pink']
    else:
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
    # convert to upper case for first letter
    list_task = [x[0].upper() + x[1:] for x in list_task]
    data_df = data_df[data_df['task'].isin(list_task)]
    print(f"Current list of files to process...")
    print(tabulate(data_df, headers='keys', tablefmt='fancy_grid'))

    ### Custom Colormap ###
    print(f"Do you have any specific colormap? Please type the name of the colormap")
    print(f"Or Please type enter to use default colormap")
    print(f"Colormap Reference: https://matplotlib.org/stable/tutorials/colors/colormaps.html")
    user_input = input()
    print(f"===================== \n")
    if user_input == '':
        cmap = 'jet'
    else:
        cmap = user_input

    ### CONFIRMATION ABOUT RUN ###
    print(f"Topomap will be saved in out/topomap")
    print(f"Press enter to continue...")
    print(f"=====================")
    input()

    for idx, row in data_df.iterrows():
        print(f"Processing {row['path']}...")
        subject_id = str(row['subject'])
        noise_type = str(row['noise_type'])
        task = str(row['task'])
        signal_df = load_mnedf(row['path'])

        channel_list = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        signal_df = signal_df[channel_list]

        len_drop = 15361
        len_keep = 61441

        signal_df = signal_df.iloc[len_drop:]
        signal_df = signal_df.iloc[:len_keep]
        signal_df = signal_df.T

        info = mne.create_info(
            ch_names = channel_list,
            sfreq = 256,
            ch_types = 'eeg',
            verbose = False,
        )

        raw = mne.io.RawArray(
            data = signal_df,
            info = info,
            verbose = False,
        )

        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)

        # Compute PSD for all band
        psd_raw, freqs = mne.time_frequency.psd_array_multitaper(
            x = raw.get_data(),
            sfreq = 256,
            verbose = False,
        )
        
        bandpow = np.zeros((len(bands), psd_raw.shape[0]))

        for b, (fmin, fmax, name) in enumerate(bands):
            bandpow[b] = np.mean(psd_raw[:, (freqs >= fmin) & (freqs < fmax)], axis=1)

        rel_powers = bandpow / np.sum(bandpow, axis=0, keepdims=True)
        rel_powers = rel_powers.T

        # Normalize the data on the first dimension (across channels) between 0 and 1
        rel_powers_normalize = (rel_powers - np.min(rel_powers, axis=0)) / (np.max(rel_powers, axis=0) - np.min(rel_powers, axis=0))

        # Create topomap for each band
        fig, ax = plt.subplots(1, 7, figsize=(15, 4))
        for i, band in enumerate(bands):
            mne.viz.plot_topomap(rel_powers_normalize[:, i], raw.info, cmap=cmap, axes=ax[i], show=False)
            ax[i].set_title(band[2])
            # Plot the maximum value on the topomap and print the channel name on the x-axis
            max_ch = np.argmax(rel_powers[:, i])
            max_val = np.max(rel_powers[:, i])
            max_ch_name = raw.info['ch_names'][max_ch]
            ax[i].set_xlabel(f"Max Channel: {max_ch_name} \n Max Value: {max_val:.2f}")
        fig.suptitle(f"Topomap of {subject_id} {noise_type} {task} (Channels is Normalized)")
        fig.tight_layout()
        # show colorbar
        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # fig.colorbar(ax[0].images[0], cax=cbar_ax)


        save_path = "out/topomap"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f"{save_path}/{subject_id}_{noise_type}_{task}_normalized.png")
        plt.close()

        # Create topomap for each band (unnormalized)
        fig, ax = plt.subplots(1, 7, figsize=(15, 4))
        for i, band in enumerate(bands):
            mne.viz.plot_topomap(rel_powers[:, i], raw.info, cmap=cmap, axes=ax[i], show=False)
            ax[i].set_title(band[2])
            # Plot the maximum value on the topomap and print the channel name on the x-axis
            max_ch = np.argmax(rel_powers[:, i])
            max_val = np.max(rel_powers[:, i])
            max_ch_name = raw.info['ch_names'][max_ch]
            ax[i].set_xlabel(f"Max Channel: {max_ch_name} \n Max Value: {max_val:.2f}")
        fig.suptitle(f"Topomap of {subject_id} {noise_type} {task}")
        fig.tight_layout()
        # show colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(ax[0].images[0], cax=cbar_ax)


        save_path = "out/topomap"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f"{save_path}/{subject_id}_{noise_type}_{task}_raw.png")
        plt.close()

        # save rel_powers as csv
        save_path = "out/topomap"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # convert rel_powers to dataframe
        rel_powers_df = pd.DataFrame(rel_powers, columns=freq_names)
        # in the first column, create a list of channel names from `channel_list`
        rel_powers_df.insert(0, 'Channel', channel_list)
        # save the dataframe as csv
        rel_powers_df.to_csv(f"{save_path}/{subject_id}_{noise_type}_{task}.csv", index=False)



