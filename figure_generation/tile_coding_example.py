import matplotlib.pyplot as plt
import numpy as np

# TODO: improve this file so that the colours are clearer.
# Might be easiest to just layer images with the right colours on top of each other instead of just building one

res = 128#256

img = np.ones((res, res, 3))

bin_size = 16

n_bins = int(res / bin_size)

print("n_bins: {}".format(n_bins))

offsets = np.array([
    [7, 9],
    [1, 14],
    [11, 3],
])


loc = [40, 50]

linspaces = np.zeros((3, 2, n_bins + 1))

# indexes into the linspaces that the locations fall under
inds = np.zeros((3, 2))

for t in range(3):  # tiling
    for c in range(2):  # x or y
        # the -bin_size/2 is to make the lining up of the grid lines make sense
        # so the filled in square is between lines rather than centered
        linspaces[t, c, :] = np.linspace(-bin_size/2 + offsets[t, c], -bin_size/2 + res + offsets[t, c], n_bins + 1)
        inds[t, c] = np.argmin(np.abs(linspaces[t, c, :] - loc[c]))


alpha_lines = False

if alpha_lines:

    for x in range(res):
        for y in range(res):
            for t in range(3):  # tilings/channels
                # if the point lies on a grid line in the x or y direction
                if (x % bin_size == offsets[t, 0]) or (y % bin_size == offsets[t, 1]):
                    img[x, y, t] = 0

                match_x = np.argmin(np.abs(linspaces[t, 0, :] - x)) == inds[t, 0]
                match_y = np.argmin(np.abs(linspaces[t, 1, :] - y)) == inds[t, 1]

                # if the square should be filled in
                if match_x and match_y:
                    img[x, y, t] = 0

                # The true location
                if x == loc[0] and y == loc[1]:
                    img[x, y, t] = .5

else:

    for t in range(3):
        loc_indx = np.argmin(np.abs(linspaces[t, 0, :] - loc[0]))
        loc_indy = np.argmin(np.abs(linspaces[t, 1, :] - loc[1]))
        img[int(linspaces[t, 0, loc_indx] - bin_size/2): int(linspaces[t, 0, loc_indx] + bin_size/2)+1, int(linspaces[t, 1, loc_indy] - bin_size/2):int(linspaces[t, 1, loc_indy] + bin_size/2)+1, t] = 0

    # # First colour in the squares
    # for x in range(res):
    #     for y in range(res):
    #         for t in range(3):  # tilings/channels
    #
    #             indx = np.argmin(np.abs(linspaces[t, 0, :] - x))
    #             indy = np.argmin(np.abs(linspaces[t, 1, :] - y))
    #
    #             match_x = indx == inds[t, 0]
    #             match_y = indy == inds[t, 1]
    #
    #             # # if the square should be filled in
    #             # if match_x and match_y:
    #             #     img[x, y, t] = 0
    #
    #
    #             # fill in the outer lines to make the image look nicer
    #             if ((x % bin_size == offsets[t, 0]) or (y % bin_size == offsets[t, 1])):# and linspaces[t, 0, :][indx - 1] == x:
    #                 # if (int(linspaces[t, 0, :][indx] + bin_size / 2) == x) or (int(linspaces[t, 1, :][indy] + bin_size / 2) == y):
    #                 # if (int(linspaces[t, 0, :][indx] - bin_size /2 + 1) == x) or (int(linspaces[t, 1, :][indy] - bin_size/2 + 1) == y):
    #                 if True:
    #                     print("hskdhajshf")
    #                     # print(linspaces[t, 0, :][indx - 1], x)
    #                     img[x, y, t] = 0


    # Now draw the grid lines, but don't let them overlap
    for x in range(res):
        for y in range(res):
            for t in range(3):  # tilings/channels
                # if the point lies on a grid line in the x or y direction
                if (x % bin_size == offsets[t, 0]) or (y % bin_size == offsets[t, 1]):
                    if img[x, y, :].min() > .5:
                        img[x, y, t] = 0
                    # elif img[x, y, t] == .5:
                    #     img[x, y, t] = 0.1


    # The true location
    img[loc[0], loc[1], :] = .5#0


plt.imshow(img)

plt.show()
