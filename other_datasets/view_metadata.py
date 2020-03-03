import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv('metadata_regression.csv')
df = pd.read_csv('metadata.csv')

# print(len(df))
print(df['Number of Samples'].mean())
print(df['Float Features'].mean())

print(df['Number of Samples'].max())
print(df['Number of Samples'].min())
print(df['Number of Samples'].median())

# print(df)
# print(df[df['Type'] == 'Classification']['Number of Samples'].mean())
# print(df[df['Type'] == 'Regression']['Number of Samples'].mean())
#
# print(df[df['Type'] == 'Classification']['Float Features'].mean())
# print(df[df['Type'] == 'Regression']['Float Features'].mean())

# print(df[(df['Type'] == 'Regression') & (df['Number of Samples'] < 5000)]['Dataset'])
