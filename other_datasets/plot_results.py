import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# {
#     'Dim': 0,
#     'Seed': seed,
#     'Scale': 0,
#     'Encoding': 'Normalized',
#     'Dataset': classification_dataset,
#     'Model': 'MLP - {}'.format(hidden_layer_sizes),
#     'Accuracy': acc,
#     'Solver': solver,
# },

fname = sys.argv[1]

df = pd.read_csv(fname)

meta_df = pd.read_csv('metadata.csv')

df = df.merge(meta_df, on='Dataset')

# df_ssp = df[df['Encoding'] == 'SSP Normalized']
# df_normal = df[df['Encoding'] == 'Normalized']

plt.figure()
sns.barplot(data=df, x='Encoding', y='Accuracy')
plt.figure()
sns.barplot(data=df, x='Encoding', y='Accuracy', hue='Dataset')


plt.figure()
sns.barplot(data=df, x='Encoding', y='Accuracy', hue='Dim')

plt.figure()
sns.barplot(data=df, x='Dim', y='Accuracy', hue='Scale')

plt.figure()
sns.barplot(data=df, x='Encoding', y='Accuracy', hue='Scale')

plt.figure()
sns.barplot(data=df, x='Encoding', y='Accuracy', hue='Model')


plt.figure()
sns.barplot(data=df, x='Float Features', y='Accuracy', hue='Encoding')
plt.figure()
sns.barplot(data=df, x='Imbalance Metric', y='Accuracy', hue='Encoding')
plt.figure()
sns.barplot(data=df, x='Number of Samples', y='Accuracy', hue='Encoding')


# for plot_df in [df_ssp, df_normal]:
#
#     # plt.figure()
#     # sns.scatterplot(data=plot_df, x='Float Features', y='Accuracy', hue='Encoding')
#     # plt.figure()
#     # sns.scatterplot(data=plot_df, x='Imbalance Metric', y='Accuracy', hue='Encoding')
#     # plt.figure()
#     # sns.scatterplot(data=plot_df, x='Number of Samples', y='Accuracy', hue='Encoding')
#
#
#     plt.figure()
#     sns.barplot(data=plot_df, x='Float Features', y='Accuracy', hue='Encoding')
#     plt.figure()
#     sns.barplot(data=plot_df, x='Imbalance Metric', y='Accuracy', hue='Encoding')
#     plt.figure()
#     sns.barplot(data=plot_df, x='Number of Samples', y='Accuracy', hue='Encoding')

plt.show()
