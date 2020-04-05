height = 4
width = 4

connections = []

# the 8 possible knight move offsets
offsets = [
    [1, 2],
    [-1, 2],
    [1, -2],
    [-1, -2],
    [2, 1],
    [-2, 1],
    [2, -1],
    [-2, -1]
]

# list of nodes within the space marked as impassable
blocked_nodes = [
    (4, 4),
    (4, 3),
    (3, 4),
    (1, 1),
    # (2, 1), #
    (4, 2),
]

# for every square on the board, create a connection to all squares one knight move away
for w in range(1, width + 1):
    for h in range(1, height + 1):
        if (w, h) in blocked_nodes:
            continue
        for offset in offsets:
            wn = w + offset[0]
            hn = h + offset[1]
            if (wn >= 1) and (wn <= width) and (hn >= 1) and (hn <= height) and (wn, hn) not in blocked_nodes:
                connections.append("{}{} {}{}".format(w, h, w + offset[0], h + offset[1]))

for c in connections:
    print(c)

