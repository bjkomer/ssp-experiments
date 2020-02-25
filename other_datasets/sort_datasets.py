# Sort out which encodings work the best on which datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from constants import full_continuous
import sys

# df = pd.read_csv('encoding_exp_results_v2.csv')
# # df = pd.read_csv('encoding_exp_results.csv')
#
# # remove the bad scales
# df = df[df['Scale'] != 5.0]
# df = df[df['Scale'] != 2.0]
# df = df[df['Scale'] != 1.0]
# df = df[df['Scale'] != 0.5]
#
# meta_df = pd.read_csv('metadata.csv')
#
# df = df.merge(meta_df, on='Dataset')
#
# ssp_best = []
# normal_best = []
# tied = []
#
# for dataset_name in full_continuous:
#     acc_ssp = df[(df['Dataset'] == dataset_name) & (df['Encoding'] == 'SSP Normalized')]['Accuracy'].mean()
#     acc_normal = df[(df['Dataset'] == dataset_name) & (df['Encoding'] == 'Normalized')]['Accuracy'].mean()
#
#     if acc_ssp > acc_normal:
#         ssp_best.append(dataset_name)
#     elif acc_normal > acc_ssp:
#         normal_best.append(dataset_name)
#     else:
#         tied.append(dataset_name)
#
#
# # get metadata for the best results
# meta_ssp_df = meta_df[meta_df['Dataset'].isin(ssp_best)]
# meta_normal_df = meta_df[meta_df['Dataset'].isin(normal_best)]
#
# print("")
# print("SSP best: {}".format(ssp_best))
# print("Number of datasets: {}".format(len(ssp_best)))
# print("Avg # of features: {}".format(meta_ssp_df['Float Features'].mean()))
# print("Avg imbalance metric: {}".format(meta_ssp_df['Imbalance Metric'].mean()))
# print("Avg # of samples: {}".format(meta_ssp_df['Number of Samples'].mean()))
# print("Avg overall Acc: {}".format(df[df['Encoding'] == 'SSP Normalized']['Accuracy'].mean()))
# print("")
# print("Normal best: {}".format(normal_best))
# print("Number of datasets: {}".format(len(normal_best)))
# print("Avg # of features: {}".format(meta_normal_df['Float Features'].mean()))
# print("Avg imbalance metric: {}".format(meta_normal_df['Imbalance Metric'].mean()))
# print("Avg # of samples: {}".format(meta_normal_df['Number of Samples'].mean()))
# print("Avg overall Acc: {}".format(df[df['Encoding'] == 'Normalized']['Accuracy'].mean()))
# print("")
# print("Tied: {}".format(tied))

if len(sys.argv) > 1:
    # one or more files given, load them all into one dataframe
    fnames = sys.argv[1:]
    df = pd.DataFrame()
    for fname in fnames:
        df = df.append(pd.read_csv(fname))

meta_df = pd.read_csv('metadata.csv')
df = df.merge(meta_df, on='Dataset')

encodings = df['Encoding'].unique()

n_encodings = len(encodings)
n_datasets = len(full_continuous)

# initialize dictionary with empty lists
best_dict = {}
for encoding in encodings:
    best_dict[encoding] = []

accs = np.zeros((n_encodings, n_datasets))

for di, dataset_name in enumerate(full_continuous):
    for ei, encoding in enumerate(encodings):
        accs[ei, di] = df[(df['Dataset'] == dataset_name) & (df['Encoding'] == encoding)]['Accuracy'].mean()

    ind_best = np.argmax(accs[:, di])
    best_dict[encodings[ind_best]].append(dataset_name)

for encoding in encodings:
    result_df = meta_df[meta_df['Dataset'].isin(best_dict[encoding])]
    print("")
    print("{} best: {}".format(encoding, best_dict[encoding]))
    print("Number of datasets: {}".format(len(best_dict[encoding])))
    print("Avg # of features: {}".format(result_df['Float Features'].mean()))
    print("Avg imbalance metric: {}".format(result_df['Imbalance Metric'].mean()))
    print("Avg # of samples: {}".format(result_df['Number of Samples'].mean()))
    print("Avg overall Acc: {}".format(df[df['Encoding'] == encoding]['Accuracy'].mean()))
    print("")
