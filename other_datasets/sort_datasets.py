# Sort out which encodings work the best on which datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from constants import full_continuous

df = pd.read_csv('encoding_exp_results_v2.csv')
# df = pd.read_csv('encoding_exp_results.csv')

# remove the bad scales
df = df[df['Scale'] != 5.0]
df = df[df['Scale'] != 2.0]
df = df[df['Scale'] != 1.0]
df = df[df['Scale'] != 0.5]

meta_df = pd.read_csv('metadata.csv')

df = df.merge(meta_df, on='Dataset')

ssp_best = []
normal_best = []
tied = []

for dataset_name in full_continuous:
    acc_ssp = df[(df['Dataset'] == dataset_name) & (df['Encoding'] == 'SSP Normalized')]['Accuracy'].mean()
    acc_normal = df[(df['Dataset'] == dataset_name) & (df['Encoding'] == 'Normalized')]['Accuracy'].mean()

    if acc_ssp > acc_normal:
        ssp_best.append(dataset_name)
    elif acc_normal > acc_ssp:
        normal_best.append(dataset_name)
    else:
        tied.append(dataset_name)


# get metadata for the best results
meta_ssp_df = meta_df[meta_df['Dataset'].isin(ssp_best)]
meta_normal_df = meta_df[meta_df['Dataset'].isin(normal_best)]

print("")
print("SSP best: {}".format(ssp_best))
print("Number of datasets: {}".format(len(ssp_best)))
print("Avg # of features: {}".format(meta_ssp_df['Float Features'].mean()))
print("Avg imbalance metric: {}".format(meta_ssp_df['Imbalance Metric'].mean()))
print("Avg # of samples: {}".format(meta_ssp_df['Number of Samples'].mean()))
print("Avg overall Acc: {}".format(df[df['Encoding'] == 'SSP Normalized']['Accuracy'].mean()))
print("")
print("Normal best: {}".format(normal_best))
print("Number of datasets: {}".format(len(normal_best)))
print("Avg # of features: {}".format(meta_normal_df['Float Features'].mean()))
print("Avg imbalance metric: {}".format(meta_normal_df['Imbalance Metric'].mean()))
print("Avg # of samples: {}".format(meta_normal_df['Number of Samples'].mean()))
print("Avg overall Acc: {}".format(df[df['Encoding'] == 'Normalized']['Accuracy'].mean()))
print("")
print("Tied: {}".format(tied))