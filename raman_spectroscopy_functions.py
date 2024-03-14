import numpy as np
import pandas as pd
from matplotlib import colormaps
import scipy.signal as ss
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import medfilt

# subtract min value from signal
def subtract_min(signal):
    subtracted_signal = signal - min(signal)
    return subtracted_signal

# subtract polynomial fitted baseline from signal
def subtract_poly(x, signal, polydeg):
        coeffs = np.polyfit(x, signal, polydeg)
        polyfunc = np.poly1d(coeffs)
        subtracted_signal = signal - polyfunc(x)
        return subtracted_signal

# smooth signal (Savitzky-Golay)
def smoothing(signal, window, deg):
    smoothed_signal = ss.savgol_filter(signal, window, deg)
    return smoothed_signal

# cosmic spike detection (reference: Whitaker and Hayes, 2018)
def cosmic_spike_detection(x, signal, mod_z_factor, mod_z_limit, sample_no):
    # calculate difference of signal from its predecessor
    diff_signal = np.diff(signal, append=[signal.iloc[-1]])
    # find median absolute deviation
    median = np.median(diff_signal)
    mad = np.median([np.abs(diff_signal - median)])
    # find modified z score
    mod_z_scores = mod_z_factor * (diff_signal - median) / mad
    # find maximum value and its index
    abs_max = np.max(np.abs(mod_z_scores))
    abs_max_idx = np.argmax(np.abs(mod_z_scores))
    
    # if maximum exceed threshold, despike plot
    if abs_max > mod_z_limit:
        plt.plot(x, signal, label='Before De-spiking')
        # median filter of segemented signal
        filtered = medfilt(signal[abs_max_idx-5:abs_max_idx+5])
        # replace value
        for y_idx, y_new in enumerate(filtered):
            signal.iloc[abs_max_idx-5+y_idx] = y_new
        plt.plot(x, signal, label='After De-spiking')
        plt.axvline(x=x[abs_max_idx], alpha=0.3, color='yellow')
        plt.legend()
        plt.title(abs_max)
        plt.savefig(os.path.join(os.getcwd(), 'results', 'CS Sample%i' % sample_no))
        plt.close()
    
    return signal  

# find peaks and their respective properties in dataset
def peak_finder(x, df, key_label_col, col_names):
    keys = ['peak_position', 'peak_intensity', 'peak_width']
    dic = dict()
    for row in df.index:
        label = df.at[row, key_label_col]
        dic[label] = dict.fromkeys(keys, None)
        signal = df.loc[row, col_names]
        peaks, properties = ss.find_peaks(signal, height=0, width=0)
        dic[label]['peak_position'] = [x[posn] for posn in peaks]
        dic[label]['peak_intensity'] = properties['peak_heights']
        dic[label]['peak_width'] = properties['widths']
        dic[label]['peak_ratio'] = properties['peak_heights']/max(properties['peak_heights'])
    return dic


# create list of colors with varying colors of a specified length
def create_colors(var_len):
    spacing = np.linspace(0, 1, var_len)
    color_map = colormaps.get_cmap('Spectral')
    color_list = color_map(spacing)
    return color_list

# principal component analysis (PCA): unsupervised clustering
def principal_component_analysis(df, no_of_pc, col_names, pca_label):
    # prepare data for PCA
    pca_data = df.loc[:, col_names].values
    norm_pca_data = StandardScaler().fit_transform(pca_data)
    
    # intialize PCA function based on specific PC
    pca_function = PCA(n_components=no_of_pc)
    pc = pca_function.fit_transform(norm_pca_data)
    
    # extract results from PCA
    pca_df_columns = ['PC%i' %pc_no for pc_no in range(1, no_of_pc+1)]
    pca_df = pd.DataFrame(data = pc, columns = pca_df_columns, index=pca_label)
    pca_var_ratio = 100*pca_function.explained_variance_ratio_
    pc_components = pca_function.components_

    return pca_df, pca_var_ratio, pc_components