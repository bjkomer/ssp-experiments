import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

n_images = 20

categories = ['other', 'flower', 'dog', 'bird', 'berry']
n_categories = len(categories)

res = 256

path = 'data/Linnaeus{}/test/{}/{}_{}.jpg'

# img=mpimg.imread(path + '400_256.jpg')
#
# plt.imshow(img)
# plt.show()

images = np.zeros((n_images, res, res, 3))


for i in range(n_images):
    category = categories[i % n_categories]

    # integer division
    n = int((i / n_categories) + 1)

    # convert from between 0 and 256 to -1 and 1
    images[i, :, :, :] = (mpimg.imread(path.format(res, category, n, res)) / 128) - 1

    # print(images[i, :, :, :])

np.savez('data/{}images_linnaeus.npz'.format(n_images), images=images)
