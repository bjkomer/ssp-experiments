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

df = pd.read_csv('encoding_exp_results.csv')

plt.figure()
sns.barplot(data=df, x='Encoding', y='Accuracy')
# plt.figure()
# sns.barplot(data=df, x='Encoding', y='Accuracy', hue='Dim')

plt.show()
