from scipy.special import diric
import numpy as np
import matplotlib.pyplot as plt


def normalized_diric(x, n):
    # return np.sin(x * n / 2) / (n * np.sin(x / 2))
    return np.sin(x * n) / (n * np.sin(x))
    # return np.sin(x * n / 2) / (np.sin(x / 2))
    # return np.sin(x*n) / x*n


def normalized_sinc(x, n):
    return np.sin(x * n) / (x*n)


dim = 256
# dim = 9
dim = 17
res = 512
limit = 100
xs = np.linspace(-limit, limit, res)

# plt.plot(xs, diric(xs, dim))
plt.plot(xs, normalized_diric(xs/dim, dim))
plt.plot(xs, normalized_sinc(xs/dim, dim))

plt.show()

