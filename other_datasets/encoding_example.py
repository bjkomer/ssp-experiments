from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from functools import partial
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb

from constants import some_continuous, full_continuous, small_continuous
from feature_encoding import encode_dataset

from pmlb import fetch_data, classification_dataset_names

logit_test_scores = []
gnb_test_scores = []
logit_enc_test_scores = []
gnb_enc_test_scores = []
logit_scaled_test_scores = []
gnb_scaled_test_scores = []
logit_enc_scaled_test_scores = []
gnb_enc_scaled_test_scores = []

# datasets = full_continuous
datasets = small_continuous

n_datasets = len(datasets)


classifiers = [
    MLPClassifier,
    LogisticRegression,
    GaussianNB,
]

classifier_names = [
    'MLPClassifier',
    'LogisticRegression',
    'GaussianNB',
]

preprocs = [
    '',
    'Enc',
    'Scaled',
    'Enc Scaled'
]

combination_names = []
# generate a dictionary of empty lists to be filled later
test_scores = {}
for name in classifier_names:
    for pre in preprocs:
        test_scores[name + pre] = []
        combination_names.append(name + pre)


for i, classification_dataset in enumerate(datasets):

    # temporarily skipping the bigger datasets to save time prototyping
    if (classification_dataset == 'shuttle') or (classification_dataset == 'magic'):
        continue

    print('\x1b[2K\r {} of {}. {}'.format(i + 1, n_datasets, classification_dataset), end="\r")
    X, y = fetch_data(classification_dataset, return_X_y=True)

    # making features 0 mean and unit variance
    scaler = StandardScaler()

    train_X, test_X, train_y, test_y = train_test_split(X, y)

    scaler.fit(train_X)

    train_X_scaled = scaler.transform(train_X)
    test_X_scaled = scaler.transform(test_X)

    train_X_enc = encode_dataset(train_X, dim=256, seed=13, scale=1.0)
    test_X_enc = encode_dataset(test_X, dim=256, seed=13, scale=1.0)

    train_X_enc_scaled = encode_dataset(train_X_scaled, dim=256, seed=13, scale=1.0)
    test_X_enc_scaled = encode_dataset(test_X_scaled, dim=256, seed=13, scale=1.0)

    # print(X.shape)

    for train, test, pre in zip([train_X, train_X_enc, train_X_scaled, train_X_enc_scaled], [test_X, test_X_enc, test_X_scaled, test_X_enc_scaled], preprocs):
        for i, classifier in enumerate(classifiers):
            clf = classifier()

            clf.fit(train, train_y)

            test_scores[classifier_names[i] + pre].append(clf.score(test, test_y))

sb.boxplot(
    data=[
        test_scores[name] for name in combination_names
    ],
    notch=True
)
plt.xticks(
    list(range(len(combination_names))),
    combination_names
)
plt.ylabel('Test Accuracy')

plt.show()
