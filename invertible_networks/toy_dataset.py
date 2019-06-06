import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data


class ToyDataGen(object):

    def __init__(self, mode_labels=(0, 0, 0, 0, 1, 2, 3, 3), radius=3, std=.2):

        self.mode_labels = mode_labels
        self.std = std

        angles = np.linspace(np.pi/8, 2*np.pi+np.pi/8, 9)[:-1]
        self.center_locations = np.vstack([radius*np.cos(angles), radius*np.sin(angles)]).T

        assert self.center_locations.shape == (8, 2)

    def sample(self):

        # choose mode
        mode_index = np.random.randint(low=0, high=len(self.mode_labels))
        label = self.mode_labels[mode_index]

        # sample gaussian point from within mode
        loc = np.random.normal(loc=self.center_locations[mode_index, :], scale=self.std)

        # return location and label
        return loc, label

    def sample_batch(self, n_samples):

        locations = np.zeros((n_samples, 2))
        # labels = np.zeros((n_samples, 1))
        labels = np.zeros((n_samples,))  # important for the pytorch criterion that this is one dimensional

        for i in range(n_samples):
            loc, label = self.sample()
            locations[i, :] = loc
            labels[i] = label

        return locations, labels

        # TODO: vectorize this
        # # choose mode
        # mode_index = np.random.randint(low=0, high=len(self.mode_labels), size=n_samples)
        # label = self.mode_labels[mode_index]
        #
        # # sample gaussian point from within mode
        # loc = np.random.normal(loc=self.center_locations[mode_index, :], scale=self.std, size=n_samples)
        #
        # # return location and label
        # return loc, label


class ToyDataset(data.Dataset):

    def __init__(self, n_samples, mode_labels=(0, 0, 0, 0, 1, 2, 3, 3), radius=3, std=.2):

        datagen = ToyDataGen(mode_labels=mode_labels, radius=radius, std=std)

        locations, labels = datagen.sample_batch(n_samples)

        self.locations = locations.astype(np.float32)

        self.labels = labels.astype(np.long)

        # # Convert labels to one-hot encoding
        # n_classes = max(mode_labels) + 1
        # self.labels = np.zeros((n_samples, n_classes)).astype(np.long)
        # for i in range(n_samples):
        #     self.labels[i, int(labels[i])] = 1

    def __getitem__(self, index):
        return self.locations[index], self.labels[index]

    def __len__(self):
        return self.locations.shape[0]


def plot_data(locations, labels, ax):

    r = np.zeros((len(labels), 1))
    g = np.zeros((len(labels), 1))
    b = np.zeros((len(labels), 1))
    for i in range(len(labels)):
        if labels[i] == 0:
            r[i, 0] = 0
            g[i, 0] = 0
            b[i, 0] = 1
        elif labels[i] == 1:
            r[i, 0] = 1
            g[i, 0] = 0
            b[i, 0] = 0
        elif labels[i] == 2:
            r[i, 0] = 0
            g[i, 0] = 1
            b[i, 0] = 0
        elif labels[i] == 3:
            r[i, 0] = 1
            g[i, 0] = 0
            b[i, 0] = 1
        elif labels[i] == 4:
            r[i, 0] = 0
            g[i, 0] = 1
            b[i, 0] = 1
        elif labels[i] == 5:
            r[i, 0] = 1
            g[i, 0] = 1
            b[i, 0] = 0
        elif labels[i] == 6:
            r[i, 0] = .5
            g[i, 0] = 1
            b[i, 0] = .5
        elif labels[i] == 7:
            r[i, 0] = 1
            g[i, 0] = .5
            b[i, 0] = 0

    colours = np.hstack([r, g, b])
    ax.scatter(locations[:, 0], locations[:, 1], c=colours)


if __name__ == "__main__":

    dataset = ToyDataGen()

    n_samples = 1000
    locations = np.zeros((n_samples, 2))
    labels = np.zeros((n_samples, 1))

    for i in range(n_samples):
        loc, label = dataset.sample()
        locations[i, :] = loc
        labels[i] = label

    locations, labels = dataset.sample_batch(n_samples)

    fix, ax = plt.subplots()

    plot_data(locations, labels, ax)

    plt.show()
