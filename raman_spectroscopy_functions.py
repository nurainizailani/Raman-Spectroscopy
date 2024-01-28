import numpy as np
import pandas as pd
from matplotlib import colormaps
import scipy.signal as ss
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

# cosmic spike detection (Whitaker and Hayes, 2018)
def cosmic_spike_detection(signal, mod_z_factor, mod_z_limit):
    median = np.median(signal)
    mad = np.median([np.abs(signal - median)])
    mod_z_scores = mod_z_factor * (signal - median) / mad
    if max(mod_z_scores) > mod_z_limit:
        # plt.plot(rs, signal)
        # plt.axvline(x=rs[np.argmax(mod_z_scores)], alpha=0.3, color='yellow')
        # plt.show()
        # plt.close()
        despiked_signal = signal # check outcome
    else:
        despiked_signal = signal
    
    return despiked_signal  

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