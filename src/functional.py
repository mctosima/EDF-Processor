import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json

def list_files_and_make_df(data_path):
    file_list = sorted(os.listdir(data_path))
    data = []
    for item in file_list:
        split_item = item.split('_')
        id_no = split_item[0]
        noise_type = split_item[1]
        task = split_item[2]
        sex = split_item[3]
        full_path = os.path.join(data_path, item)
        
        # add to dataframe
        data.append({'subject': id_no, 'noise_type': noise_type, 'task': task, 'sex': sex, 'path': full_path})

    df = pd.DataFrame(data)
    # print(tabulate(df, headers='keys', tablefmt='fancy_grid'))
    return df

def load_mnedf(edf_path):
    signal_df = mne.io.read_raw_edf(edf_path, preload=True, verbose=False).to_data_frame()
    return signal_df

def get_psd_feature(
    dataframe,
    freq_type: str,
    fs: int = 256,
    len_drop: int = 15361,
    # len_keep: int = 61441,
    len_keep: int = 46081,
    plot_psd: bool = False,
    return_psd: bool = False,
    channel_drop: list = None,
    select_channels: list = None,
):
    
    # 1. Only get the signal from the dataframe
    df_signal = dataframe[['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']]
    
    # Drop channels if `channel_drop` is not None
    if channel_drop is not None:
        df_signal = df_signal.drop(columns=channel_drop)
        
    # Select certain channels based on `select_channels`
    if select_channels is not None:
        df_signal = df_signal[select_channels]

    
    
    # 2. Crop the signal
    df_signal = df_signal.iloc[len_drop:]
    df_signal = df_signal.iloc[:len_keep]
    
    # 3. Transpose the dataframe
    df_signal = df_signal.T
    
    # 4. Transform DF into MNE object
    info = mne.create_info(
        ch_names = df_signal.shape[0],
        sfreq = fs,
        ch_types = 'eeg',
        verbose= False,
    )
    
    raw = mne.io.RawArray(
        data = df_signal,
        info = info,
        verbose = False,
    )
    
    # 5. Get the PSD for RAW Signal
    psd_raw, freqs = mne.time_frequency.psd_array_multitaper(
        x = raw.get_data(),
        sfreq = fs,
        verbose = False,
    )
    
    # 6. Get the PSD for Selected Bands
    ### OPEN CONFIG ###
    with open('src/config.json', 'r') as f:
        config = json.load(f)

    ### DEFINE FREQUENCY BANDS ###
    if config["edf_config"]["frequency_mode"] == 5:
        freq_splits = config["edf_config"]["five_freq_split"]
    elif config["edf_config"]["frequency_mode"] == 7:
        freq_splits = config["edf_config"]["seven_freq_split"]
        
    freq_bands = {d["name"]: (d["low"], d["high"]) for d in freq_splits}
    
    # obtain frequency range based on the parameter passed to `freq_type`
    freq_range = freq_bands[freq_type]
    raw_copy = raw.copy()
    raw_copy.filter(freq_range[0], freq_range[1], fir_design = 'firwin', verbose=False)
    
    psd_filered, freqs_filtered = mne.time_frequency.psd_array_multitaper(
        x = raw_copy.get_data(),
        sfreq = fs,
        verbose = False,
    )
    
            
    # 7. Count the features
    sum_raw = np.sum(psd_raw)
    avg_raw = np.average(psd_raw)
    
    sum_filtered = np.sum(psd_filered)
    avg_filtered = np.average(psd_filered)
    
    rel_pow = sum_filtered / sum_raw
    
    # 8. Return
    
    output = {
        'sum_raw': sum_raw,
        'avg_raw': avg_raw,
        'sum_filtered': sum_filtered,
        'avg_filtered': avg_filtered,
        'rel_pow': rel_pow,
    }
    
    if plot_psd:
        # Plot the PSD
        plt.figure(figsize=(10, 5))
        plt.plot(freqs_filtered, psd_filered.mean(0), color='r')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (dB)')
        plt.title('PSD of filtered signal')
        xlim = (freq_range[0]-2, freq_range[1]+2)
        plt.xlim(xlim)        
        plt.show()
        
    if return_psd:
        output['psd_raw'] = psd_raw
        output['psd_raw_freqs'] = freqs
        output['psd_filtered'] = psd_filered
        output['psd_filtered_freqs'] = freqs_filtered
    
    return output