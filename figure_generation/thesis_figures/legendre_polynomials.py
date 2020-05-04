import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import legendre
import numpy as np

# half_dim = int(dim // 2)
#
# # dim must be evenly divisible by 2 for this encoding
# assert half_dim * 2 == dim
#
# # set up legendre polynomial functions
# poly = []
# for i in range(half_dim):
#     poly.append(legendre(i + 1))
#
# domain = limit_high - limit_low
#
# def encoding_func(x, y):
#
#     # shift x and y to be between -1 and 1 before going through the polynomials
#     xn = ((x - limit_low) / domain) * 2 - 1
#     yn = ((y - limit_low) / domain) * 2 - 1
#     ret = np.zeros((dim,))
#     for i in range(half_dim):
#         ret[i] = poly[i](xn)
#         ret[i + half_dim] = poly[i](yn)
#
#     return ret

res = 256
xs = np.linspace(-1, 1, res)

n_poly = 7

values = np.zeros((res, n_poly))

for k in range(n_poly):
    poly = legendre(k+1)
    for i, x in enumerate(xs):
        values[i, k] = poly(x)

fig, ax = plt.subplots(figsize=(5, 2))

ax.plot(xs, values)

legend = ['order {}'.format(i+1) for i in range(n_poly)]

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

ax.legend(legend, bbox_to_anchor=(1.0, 1.00))

sns.despine()

plt.show()
