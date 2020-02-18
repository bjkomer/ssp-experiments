import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# df = pd.read_csv('encoding_exp_results_v2.csv')
# df = pd.read_csv('encoding_exp_results.csv')
# df = pd.read_csv('encoding_exp_all_results.csv')
df = pd.read_csv('encoding_exp_all_results_100iters.csv')

meta_df = pd.read_csv('metadata.csv')

df = df.merge(meta_df, on='Dataset')

print(df['Model'])

df_ssp = df[df['Encoding'] == 'SSP Normalized']
df_normal = df[df['Encoding'] == 'Normalized']

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
