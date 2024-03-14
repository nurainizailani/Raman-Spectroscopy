import pandas as pd
import numpy as np
import os
import openpyxl
import matplotlib.pyplot as plt
from termcolor import cprint

from raman_spectroscopy_functions import cosmic_spike_detection, subtract_min, subtract_poly,smoothing, peak_finder, create_colors, principal_component_analysis

def main():

    ## INITIALISE VARIABLES
    # define variables
    filename = r"Kaewseekhao B et al Dataset of serum proteomic spectra from tuberculosis patients detected by Raman spectroscopy and surface-enhanced Raman spectroscopy RS532.csv"
    label_col = "Sample names"
    group_label_col = "Sample names"
    no_of_pc = 2
    labels_to_drop = [] 
    # HC: healthy control, ATB: active TB, EC: exposed w/o TB, LTBI: latent TB
    smoothing_window = 5
    smoothing_degree = 3

    # define user controls
    drop_rows = False
    drop_specific_index = True
    save_plot = True
    regroup_for_pca = True

    # initialise path
    savepath = os.path.join(os.getcwd(), "results")
    if not os.path.exists(savepath):
        os.mkdir(savepath)


    ## LOAD SOURCE FILE
    # read file if path exists
    filepath = os.path.join(os.getcwd(), "data", filename)
    if os.path.exists(filepath):
        raw_dataset = pd.read_csv(filepath)
    else:
        cprint("File path does not exist: %s" %filepath, 'red')
        breakpoint()
    print("Loading source file...")


    ## FORMAT DATASET
    print("Formatting dataset...")

    # intialise list of Raman Shift
    rs_col_names = raw_dataset.columns[1:]

    # fill empty cells with sample names
    for row in raw_dataset.index:
        if pd.isnull(raw_dataset.at[row, label_col]):
            raw_dataset.at[row, label_col] = raw_dataset.at[row-1, label_col]     
    
    # drop rows without keywords
    if drop_rows:
        for label in labels_to_drop:
            raw_dataset = raw_dataset[raw_dataset[label_col].str.contains(label)==False]


    ## DATA CLEANING & PRE-PROCESSING
    print("Starting data cleaning...")

    # extract Raman Shift
    rs = rs_col_names.astype('float64')

    # drop rows based on z-score as an indicator of SNR
    z_score_dataset = raw_dataset.loc[:,rs_col_names].std(axis=1)/raw_dataset.loc[:,rs_col_names].mean(axis=1)
    clean_dataset = raw_dataset[z_score_dataset >= 0.35]
    
    # drop row of anomaly
    if drop_specific_index:
        clean_dataset = clean_dataset.drop([152, 541, 542, 576, 1671, 1676, 1713, 1718, 1739, 1781])
    
    print("Starting signal preprocessing...")
    
    # prepare empty dataframe and copy label column
    pp_dataset = pd.DataFrame(index=clean_dataset.index, columns=clean_dataset.columns)
    if group_label_col in pp_dataset.columns:
        pp_dataset[group_label_col] = clean_dataset[group_label_col]

    # iterate over each sample row
    for row_no, row in enumerate(clean_dataset.index):

        print('>preprocessing data point %i of %i' %(row_no+1, len(clean_dataset.index)))
        subject = clean_dataset.loc[row, group_label_col]
        raw_signal = clean_dataset.loc[row, rs_col_names].astype('float64')
        # notify if data lengths mismatch
        if len(rs) != len(raw_signal):
            cprint("Data length mismatch!", "red")
            breakpoint()

        # detect cosmic spike 
        raw_signal = cosmic_spike_detection(rs, raw_signal, 0.675, 6.2, row_no)
        
        # check if data is normalized
        if min(raw_signal) != 0 and max(raw_signal) != 1:
            cprint("Data is normalized", "orange")
            breakpoint()

        # baseline correction: removal of DC and AC noise from baseline
        dc_sub_signal = subtract_min(raw_signal)
        ac_sub_signal = subtract_poly(rs, dc_sub_signal, 1)
        norm_signal = subtract_min(ac_sub_signal)

        # smooth signal 
        # note: requires fine tuning
        smooth_signal = smoothing(norm_signal, smoothing_window, smoothing_degree)
        
        # save to dataframe
        pp_dataset.loc[row, rs_col_names] = smooth_signal
        if np.isnan(smooth_signal).any():
            breakpoint()

    # pp_dataset.to_csv(os.path.join('results','preprocessed data.csv'))


    ## DATA VISUALIZATION
    print("Starting data visualization...")

    # average signal by group label
    avg_dataset = pp_dataset.groupby(group_label_col).mean()
    min_dataset = pp_dataset.groupby(group_label_col).min()
    max_dataset = pp_dataset.groupby(group_label_col).max() 
    std_dataset = pp_dataset.groupby(group_label_col).std()
    low_std_dataset = std_dataset - avg_dataset
    upp_std_dataset = std_dataset + avg_dataset

    #initialize list of colors for specified length
    colors = create_colors(len(avg_dataset.index))

    # create overlay of averaged signals, with min and max boundaries and shaded std 
    plt.figure(figsize=(9,4))
    for key_label, c in zip(avg_dataset.index, colors):
        plt.plot(rs, avg_dataset.loc[key_label], label=key_label, color=c)
        plt.plot(rs, min_dataset.loc[key_label], linestyle='--', color=c, alpha=0.7)
        plt.plot(rs, max_dataset.loc[key_label], linestyle='--', color=c, alpha=0.7)
        # plt.fill_between(x=rs, y1=upp_std_dataset.loc[key_label].astype('float'), 
        #                 y2=low_std_dataset.loc[key_label].astype('float'), 
        #                 alpha=0.5, color=c)
    plt.xlabel('Raman Shift')
    plt.ylabel('Intensity (a.u.)')
    plt.xlim(min(rs), max(rs))
    plt.legend(fontsize=3, ncols=6)
    if save_plot:
        plt.savefig(os.path.join(savepath, 'signal overlay %s.png' %group_label_col), transparent=True, dpi=300)
    else:
        plt.show()
    plt.close()

    # create array of averaged signals, with min and max boundaries and shaded std
    subplot_rows = 2
    subplot_cols = 1

    while len(avg_dataset.index) >= subplot_rows*subplot_cols:
        if subplot_rows/subplot_cols != 2:
            subplot_rows += 1
        else:
            subplot_cols += 1

    f, axes = plt.subplots(nrows=subplot_rows, ncols=subplot_cols, figsize=(subplot_rows, subplot_cols), sharex=True, sharey=True)

    for ax, key_label, c in zip(axes.ravel(), avg_dataset.index, colors):
        ax.plot(rs, avg_dataset.loc[key_label], color=c)
        ax.plot(rs, min_dataset.loc[key_label], linestyle='--', color=c, alpha=0.7)
        ax.plot(rs, max_dataset.loc[key_label], linestyle='--', color=c, alpha=0.7)
        ax.fill_between(x=rs, y1=upp_std_dataset.loc[key_label].astype('float'), 
                        y2=low_std_dataset.loc[key_label].astype('float'), 
                        alpha=0.5, color=c)
        # ax.set_xlabel('Raman Shift')
        # ax.set_ylabel('Intensity (a.u.)')
        ax.set_title(key_label, fontsize=3)
    f.supxlabel('Raman Shift')
    f.supylabel('Intensity (a.u.)')
    if save_plot:
        plt.savefig(os.path.join(savepath, 'signal array %s.png' %group_label_col), transparent=True, dpi=300)
    else:
        plt.show()
    plt.close()

    # principal component analysis: unsupervised clustering
    # intialize list with labels for PCA
    if regroup_for_pca:
        pca_label = [label[:-4] for label in avg_dataset.index]
    else:
        pca_label = avg_dataset.index

    pca_df, pca_var_ratio, pc_components = principal_component_analysis(avg_dataset, no_of_pc, rs_col_names, pca_label)

    # plot PCA scatter plot of PC1 and PC2
    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('PC1 ({}%)'.format(round(pca_var_ratio[0],2)))
    plt.ylabel('PC2 ({}%)'.format(round(pca_var_ratio[1], 2)))
    plt.title("Principal Component Analysis")

    targets = np.unique(pca_label) 
    color_list = create_colors(len(targets))
    for target, color in zip(targets, color_list):
        target_pca_df = pca_df[pca_df.index == target]
        # indicesToKeep = pca_label == target
        plt.scatter(target_pca_df.loc[:, 'PC1'], target_pca_df.loc[:, 'PC2'], 
                    color = color, s = 50)
    plt.legend(targets)
    if save_plot:
        plt.savefig(os.path.join(savepath, 'PCA scatter plot.png'), transparent=True, dpi=300)
    else:
        plt.show()
    plt.close()

    # plot PC1 and PC2 individually with reference to averaged signal
    f, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)

    for key_label, c in zip(targets, color_list):
        # target_avg = raw_dataset[raw_dataset.loc[group_label_col].str.contains(key_label)].avg()
        target_avg = avg_dataset[avg_dataset.index.str.contains(key_label)].mean()
        ax1.plot(rs, target_avg, label=key_label, color=c)
    
    ax1.legend()
    ax1.set_title('Average Signal')
    ax2.plot(rs, pc_components[0])
    ax2.set_title('PC1 ({}%)'.format(round(pca_var_ratio[0],2)))
    ax3.plot(rs, pc_components[1])
    ax3.set_title('PC2 ({}%)'.format(round(pca_var_ratio[1],2)))

    if save_plot:
        plt.savefig(os.path.join(savepath, 'PCA plot.png'), transparent=True, dpi=300)
    else:
        plt.show()
    plt.close()

    cprint("COMPLETED", "green")
    return

if __name__ == '__main__':
    main()