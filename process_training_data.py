'''
Initial file for pulling and processing training data from PPG-DaLiA dataset
'''

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def load_data(session, data_dict):

    def unpack(data):
        # data = data.reshape(-1, 1)
        data = np.squeeze(data)
        return data

    # load relevant data into dictionary
    with open(f'ppg+dalia/{session}/{session}.pkl', 'rb') as file:

        print(f'loading {session}')
        data = pickle.load(file, encoding='latin1')

        data_dict[session]['ppg'] = unpack(data['signal']['wrist']['BVP'])
        data_dict[session]['acc'] = unpack(data['signal']['wrist']['ACC'])
        data_dict[session]['label'] = unpack(data['label'])        # ground truth EEG

    print(data_dict)


    return data_dict



def process_for_cnn(ppg_data, x_data, y_data, z_data, ecg_ground_truth, fs_ppg, fs_acc, num_ppg, num_acc):

    def calculate_fft(data):
        result = np.fft.fft(data)  # take full FFT not just the positive part
        return result

    def trim_fft(data, num_segments):
        # only keep 0-4Hz
        trim_index = 256 * num_segments +1         # make dynamic when you set classes
        result = data[:trim_index]
        return result


    # get the original signal into segments + take FFT
    def segment_fft(data, fs, num_data):
        segment_size_seconds = 8
        segment_step_seconds = 2
        segment_size = segment_size_seconds * fs       # better to keep in terms of indices not time
        segment_step = segment_step_seconds * fs
        num_windows = ((num_data - segment_size) // segment_step) +1                  # easy to visualise this formula

        num_segments = 1
        segments = []


        for i in range(num_windows):
            start_idx = i * segment_step
            end_idx = start_idx + (segment_size * num_segments)
            # print(start_idx)
            # print(end_idx)
            segment = data[start_idx:end_idx]
            segments.append(segment)    # append each segment (row) into the matrix of all segments

        # get rows of segments into array format
        segments = np.array(segments)

        # zero padding to original signal increaed frequency resolution before FFT, based on wanting 0-4Hz
        desired_freq = 4
        zeros_to_add = int((256*fs*num_segments)/desired_freq - (fs*segment_size_seconds*num_segments))  # see laptop notes + cancel out terms for derivation
        #print(f'Zeros to add:  {zeros_to_add}')
        segments_padded = []

        for i in range(segments.shape[0]):
            segment = segments[i]
            num_zeros = np.zeros(zeros_to_add)
            padded_row = np.concatenate((segment, num_zeros))
            segments_padded.append(padded_row)

        segments_padded = np.array(segments_padded)
        segments_fft = []
        segments_normalised = []

        # calculate FFT independently on each segment
        for i in range(segments_padded.shape[0]):
            segment_padded = segments_padded[i]
            segment_fft = calculate_fft(segment_padded)
            segments_fft.append(segment_fft)

        segments_fft = np.array(segments_fft)
        #print(f'FFT shape:  {segments_fft.shape}')

        # final processing
        for i in range(segments_fft.shape[0]):
            segment_fft = segments_fft[i]
            # cut down to 0-4Hz
            segment_fft = trim_fft(segment_fft, num_segments)
            # z-normalization individually on each new trimmed segment (makes sense)
            mean = np.mean(segment_fft, axis=0)
            std_dev = np.std(segment_fft, axis=0)
            segment_normalised = (segment_fft - mean) / std_dev
            segments_normalised.append(segment_normalised)


        result = np.array(segments_normalised)
        #print(f'Normalised shape:  {result.shape}')

        return result

    ppg_input = segment_fft(ppg_data, fs_ppg, num_ppg)
    x_input = segment_fft(x_data, fs_acc, num_acc)
    y_input = segment_fft(y_data, fs_acc, num_acc)
    z_input = segment_fft(z_data, fs_acc, num_acc)

    # with open('ppg_input.csv', 'w', newline='') as csvfile:
    #     csv.writer(csvfile).writerows(ppg_input)

    ##### MAKE SURE YOU UNDERSTAND THE DIMENSIONS INSIDE OUT!!!!! when pressed, can you visualise what's happening?
    # Great to put in your illustration / ReadMe + TEST your knowledge
    input_all_channels = np.stack([ppg_input, x_input, y_input, z_input], axis=0)
    # convert any NaN from the FFT to zero
    input_all_channels = np.nan_to_num(input_all_channels, nan=0)

    # reshape data (Conv1D)
    input_all_channels = np.transpose(input_all_channels, (1, 2, 0))


    # convert any NaN from the FFT to zero
    input_all_channels = np.nan_to_num(input_all_channels, nan=0)
    print(f'All channels shape:  {input_all_channels.shape}')

    # label the data 1-to-1
    label_ecg_data = ecg_ground_truth
    #print(label_ecg_data.shape)

    # NaN / blank tester (error handling)
    assert not np.any(np.isnan(input_all_channels)), "NaN."

    return input_all_channels, label_ecg_data

def main():

    # create dictionary to hold all data
    sessions = [f'S{i}' for i in range(1,16)]
    data_dict = {f'{session}': {} for session in sessions}

    for session in sessions:
        data_dict = load_data(session, data_dict)



if __name__ == '__main__':
    main()
