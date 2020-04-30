import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  Human data from the paper, from webplot digitizer

# pairs of 'distance (cm)' and 'reaction time (s)'
human_data = np.array([
    [1.9633252906339114, 1.0769072164948454],
    [3.3955692037727574, 1.0538144329896908],
    [3.9981574906777806, 1.1494845360824744],
    [4.600394823426189, 1.088453608247423],
    [5.0162754990129415, 1.179175257731959],
    [6.487299846457557, 1.2715463917525776],
    [7.010221539811361, 1.3160824742268042],
    [7.45242377714411, 1.3193814432989694],
    [7.956393946040798, 1.3820618556701032],
    [8.935556042991884, 1.457938144329897],
    [9.430050449660012, 1.4496907216494845],
    [9.918227681509103, 1.5008247422680414],
    [10.541697740732616, 1.4002061855670105],
    [11.607720991445493, 1.4595876288659795],
    [12.673393288001755, 1.5222680412371137],
    [14.347795569203772, 1.6228865979381446],
    [14.82035534108357, 1.6608247422680413],
    [15.861811800833516, 1.631134020618557],
    [16.318052204430796, 1.8224742268041236],
    [18.204957227462167, 1.6855670103092786],
    [18.797192366747094, 1.8785567010309279],
])

# TODO: link to the final data
fname = "/home/ctnuser/spatial-cognition/prototyping/map_traversal/exp_data_kosslyn.npy"

data = np.load(fname)

# model of the delay in reporting unrelated to the task
# calculated by aligning the 0 distance value (no delay due to task)
reaction_time_offset = .962

model_data = np.zeros((data.shape[0], 2))
model_data[:, 0] = data[:, 3]
model_data[:, 1] = data[:, 2] + reaction_time_offset

fig, ax = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)

palette = sns.color_palette(n_colors=2)

ax.scatter(human_data[:, 0], human_data[:, 1], color=palette[0])
ax.scatter(model_data[:, 0], model_data[:, 1], color=palette[1])

ax.legend(['Human', 'Model'])

# trendlines
p_human = np.poly1d(np.polyfit(human_data[:, 0], human_data[:, 1], 1))
p_model = np.poly1d(np.polyfit(model_data[:, 0], model_data[:, 1], 1))
xs = np.linspace(0, 20, 32)
ax.plot(xs, p_human(xs), color=palette[0], linestyle='dashed')
ax.plot(xs, p_model(xs), color=palette[1], linestyle='dashed')

ax.set_xlabel("Distance (cm)", fontsize=14)
ax.set_ylabel("Reaction Time (s)", fontsize=14)



sns.despine()

plt.show()
