# list the names of the datasets that only include continuous features
from pmlb import fetch_data, classification_dataset_names, regression_dataset_names
from pmlb.write_metadata import count_features_type

# only continuous features
full_continuous = []

# some continuous features
some_continuous = []

# 10 or less continuous features, only continuous
small_continuous = []

n_datasets = len(classification_dataset_names)

# for i, classification_dataset in enumerate(['banana', 'iris', 'titanic']):
for i, classification_dataset in enumerate(classification_dataset_names):
    print('\x1b[2K\r {} of {}. {}'.format(i+1, n_datasets, classification_dataset), end="\r")
    df = fetch_data(classification_dataset, return_X_y=False)
    # feat = count_features_type(df.ix[:, df.columns != 'class'])
    feat = count_features_type(df.ix[:, df.columns != 'target'])
    n_binary = feat[0]
    n_integer = feat[1]
    n_float = feat[2]

    # if classification_dataset == 'banana':
    #     print('banana:')
    #     print(feat)
    #     print(df)
    # if classification_dataset == 'titanic':
    #     print('titanic:')
    #     print(feat)
    #     print(df)
    # if classification_dataset == 'iris':
    #     print('iris:')
    #     print(feat)
    #     print(df)

    if n_float > 0:
        some_continuous.append(classification_dataset)
        if n_binary == 0 and n_integer == 0:
            full_continuous.append(classification_dataset)

            if n_float <= 10:
                small_continuous.append(classification_dataset)


print(some_continuous)
print(len(some_continuous))
print(full_continuous)
print(len(full_continuous))
print(small_continuous)
print(len(small_continuous))

some_continuous = ['Hill_Valley_with_noise', 'Hill_Valley_without_noise', 'adult', 'analcatdata_aids', 'analcatdata_asbestos', 'analcatdata_authorship', 'analcatdata_bankruptcy', 'analcatdata_creditscore', 'analcatdata_cyyoung8092', 'analcatdata_cyyoung9302', 'analcatdata_germangss', 'analcatdata_happiness', 'analcatdata_japansolvent', 'analcatdata_lawsuit', 'ann-thyroid', 'appendicitis', 'australian', 'auto', 'backache', 'banana', 'biomed', 'breast', 'breast-cancer-wisconsin', 'breast-w', 'buggyCrx', 'bupa', 'cars', 'cars1', 'churn', 'clean1', 'clean2', 'cleve', 'cleveland', 'cloud', 'cmc', 'collins', 'confidence', 'credit-a', 'credit-g', 'crx', 'diabetes', 'ecoli', 'flare', 'german', 'glass', 'glass2', 'haberman', 'hayes-roth', 'heart-c', 'heart-h', 'heart-statlog', 'hepatitis', 'hungarian', 'ionosphere', 'iris', 'irish', 'kddcup', 'letter', 'liver-disorder', 'lupus', 'magic', 'mfeat-factors', 'mfeat-fourier', 'mfeat-karhunen', 'mfeat-morphological', 'mfeat-pixel', 'mfeat-zernike', 'movement_libras', 'new-thyroid', 'optdigits', 'page-blocks', 'pendigits', 'phoneme', 'pima', 'prnn_crabs', 'prnn_fglass', 'prnn_synth', 'profb', 'ring', 'saheart', 'satimage', 'schizo', 'segmentation', 'shuttle', 'sleep', 'sonar', 'spambase', 'spectf', 'tae', 'texture', 'titanic', 'tokyo1', 'twonorm', 'vehicle', 'vowel', 'waveform-21', 'waveform-40', 'wdbc', 'wine-quality-red', 'wine-quality-white', 'wine-recognition', 'yeast']
full_continuous = ['Hill_Valley_with_noise', 'Hill_Valley_without_noise', 'analcatdata_authorship', 'appendicitis', 'banana', 'breast-cancer-wisconsin', 'bupa', 'cars1', 'confidence', 'diabetes', 'ecoli', 'glass', 'glass2', 'hayes-roth', 'heart-statlog', 'iris', 'letter', 'liver-disorder', 'lupus', 'magic', 'mfeat-factors', 'mfeat-fourier', 'mfeat-karhunen', 'mfeat-morphological', 'mfeat-pixel', 'mfeat-zernike', 'movement_libras', 'new-thyroid', 'optdigits', 'page-blocks', 'pendigits', 'phoneme', 'pima', 'prnn_fglass', 'prnn_synth', 'ring', 'satimage', 'segmentation', 'shuttle', 'sleep', 'sonar', 'spectf', 'texture', 'titanic', 'tokyo1', 'twonorm', 'vehicle', 'waveform-21', 'waveform-40', 'wdbc', 'wine-quality-red', 'wine-quality-white', 'yeast']
