import pandas as pd
import numpy as np
import os
import openpyxl
import matplotlib.pyplot as plt
from termcolor import cprint

from raman_spectroscopy_functions import cosmic_spike_detection, subtract_min, subtract_poly,smoothing, peak_finder, create_colors, principal_component_analysis

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
save_plot = True
regroup_for_pca = True

# initialise path
savepath = os.path.join(os.getcwd(), "Raman-Spectroscopy", "results")
if not os.path.exists(savepath):
    os.mkdir(savepath)


## LOAD SOURCE FILE
# read file if path exists
filepath = os.path.join(os.getcwd(), "Raman-Spectroscopy", "data", filename)
if os.path.exists(filepath):
    raw_dataset = pd.read_csv(filepath)
else:
    cprint("File path does not exist: %s" %filepath, 'red')
    breakpoint()

## FORMAT DATASET
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


## PRE-PROCESSING
# extract Raman Shift
rs = rs_col_names.astype('float64')

# prepare empty dataframe and copy label column
pp_dataset = pd.DataFrame(index=raw_dataset.index, columns=raw_dataset.columns)
if group_label_col in pp_dataset.columns:
    pp_dataset[group_label_col] = raw_dataset[group_label_col]

array = []
# iterate over each sample row
for row in raw_dataset.index:
    subject = raw_dataset.loc[row, group_label_col]
    raw_signal = raw_dataset.loc[row, rs_col_names].astype('float64')
    # notify if data lengths mismatch
    if len(rs) != len(raw_signal):
        cprint('Data length mismatch!', "red")
        breakpoint()

    # detect cosmic spike 
    # note: requires fine tuning
    raw_signal = cosmic_spike_detection(raw_signal, 0.6745, 25)
    
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

    array.append(smooth_signal)

df = pd.DataFrame(array, index=raw_dataset.index, columns=rs_col_names)

pass

## DATA VISUALIZATION
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
for key_label, c in zip(avg_dataset.index, colors):
    plt.plot(rs, avg_dataset.loc[key_label], label=key_label, color=c)
    plt.plot(rs, min_dataset.loc[key_label], linestyle='--', color=c, alpha=0.7)
    plt.plot(rs, max_dataset.loc[key_label], linestyle='--', color=c, alpha=0.7)
    plt.fill_between(x=rs, y1=upp_std_dataset.loc[key_label].astype('float'), 
                     y2=low_std_dataset.loc[key_label].astype('float'), 
                     alpha=0.5, color=c)
plt.xlabel('Raman Shift')
plt.ylabel('Intensity (a.u.)')
plt.legend()
if save_plot:
    plt.savefig(os.path.join(savepath, 'signal overlay %s.png' %group_label_col), transparent=True)
plt.close()

# create array of averaged signals, with min and max boundaries and shaded std
subplot_rows = 2
subplot_cols = 1

while len(avg_dataset.index) >= subplot_rows*subplot_cols:
    if subplot_rows%3 == 0 and subplot_rows/subplot_cols ==  3:
        subplot_cols += 1
    else:
        subplot_rows += 1

f, axes = plt.subplots(nrows=subplot_rows, ncols=subplot_cols)

for ax, key_label, c in zip(axes.ravel(), avg_dataset.index, colors):
    ax.plot(rs, avg_dataset.loc[key_label], color=c)
    ax.plot(rs, min_dataset.loc[key_label], linestyle='--', color=c, alpha=0.7)
    ax.plot(rs, max_dataset.loc[key_label], linestyle='--', color=c, alpha=0.7)
    ax.fill_between(x=rs, y1=upp_std_dataset.loc[key_label].astype('float'), 
                     y2=low_std_dataset.loc[key_label].astype('float'), 
                     alpha=0.5, color=c)
    ax.set_xlabel('Raman Shift')
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_title(key_label)
if save_plot:
    plt.savefig(os.path.join(savepath, 'signal array %s.png' %group_label_col), transparent=True)
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
    plt.savefig(os.path.join(savepath, 'PCA scatter plot.png'), transparent=True)
plt.close()

# plot PC1 and PC2 individually with reference to averaged signal
f, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
for key_label, c in zip(avg_dataset.index, color_list):
    ax1.plot(rs, avg_dataset.loc[key_label], label=key_label, color=c)
ax1.legend()
ax1.set_title('Average Signal')
ax2.plot(rs, pc_components[0])
ax2.set_title('PC1 ({}%)'.format(round(pca_var_ratio[0],2)))
ax3.plot(rs, pc_components[1])
ax3.set_title('PC2 ({}%)'.format(round(pca_var_ratio[1],2)))

if save_plot:
    plt.savefig(os.path.join(savepath, 'PCA plot.png'), transparent=True)
plt.close()
