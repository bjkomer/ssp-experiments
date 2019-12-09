import matplotlib.pyplot as plt
import numpy as np

n_images = 20

fname = 'data/cifar-10-batches-py/test_batch'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = unpickle(fname)

# print(data.keys())
# print(data[b'data'].shape)
#
# img = data[b'data'][1, :]
# img = img.reshape((3, 32, 32))
# img = np.swapaxes(img, 0, 1)
# img = np.swapaxes(img, 1, 2)
#
# plt.imshow(img)
# plt.show()

images = np.zeros((n_images, 32, 32, 3))

for i in range(n_images):
    img = data[b'data'][i, :]
    img = img.reshape((3, 32, 32))
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 1, 2)

    # convert from between 0 and 256 to -1 and 1
    images[i, :, :, :] = (img / 128) - 1

    # plt.imshow(img)
    # plt.show()

np.savez('data/{}images_cifar.npz'.format(n_images), images=images)
