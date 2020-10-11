import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

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


human_data_fig2 = np.array([
    [2.0966442953020135, 1.432754417203123],
    [3.514093959731544, 1.1894200318002897],
    [4.051006711409396, 1.2722579992020155],
    [4.566442953020134, 1.4062136812705825],
    [5.060402684563758, 1.3911096157167275],
    [6.628187919463087, 1.5714951495625975],
    [7.100671140939597, 1.7075975298202155],
    [7.5731543624161075, 1.5647291913555619],
    [8.00268456375839, 1.5411326619938903],
    [9.097986577181207, 1.6940227366114233],
    [9.613422818791946, 1.8812171054590499],
    [10.730201342281877, 1.6614277972642224],
    [10.236241610738254, 1.9693446401029044],
    [11.739597315436242, 1.6855145513122087],
    [13.210738255033558, 1.8616795793310033],
    [15.025503355704696, 1.7683196465046482],
    [14.467114093959733, 1.900574549049267],
    [15.991946308724831, 2.0085726195935045],
    [16.528859060402684, 1.989192308379436],
    [18.50469798657718, 1.924516951221691],
    [19.02013422818792, 2.048889669670028],
])

human_data_fig2_jump = np.array([
    [1.9892617449664431, 0.7896739577068062],
    [3.4926174496644293, 0.8912919611965009],
    [4.008053691275167, 0.709009843797455],
    [4.555704697986577, 0.788649202312965],
    [5.028187919463086, 0.7426752737861999],
    [6.660402684563759, 0.8314645402952545],
    [7.132885906040269, 0.7109564502778061],
    [7.63758389261745, 0.7895481857756232],
    [8.024161073825503, 0.7553210697760284],
    [9.065771812080538, 0.6580107549293428],
    [9.548993288590605, 0.8143395484835314],
    [10.064429530201341, 0.7726075641812022],
    [10.558389261744967, 0.8118069591419879],
    [11.546308724832215, 0.7528499371735855],
    [13.028187919463086, 0.771424164646892],
    [14.488590604026847, 0.7357035069645019],
    [15.068456375838924, 0.7099174026190578],
    [16.013422818791945, 0.7478719413064321],
    [16.550335570469798, 0.8903372379007048],
    [18.558389261744964, 0.8320376601180306],
    [19.02013422818792, 0.8158816838670107],
])




# first figure
reaction_time_offset = .962

# second figure
# reaction_time_offset = 1.282
# human_data = human_data_fig2

# instant detection offset for second figure
#.7713

if len(sys.argv) <= 1:
    # # TODO: link to the final data
    # fname = "/home/ctnuser/spatial-cognition/prototyping/map_traversal/exp_data_kosslyn.npy"
    attractor = True
    if not attractor:
        # old version with cleanup and no attractor
        fname = "/home/ctnuser/ssp_navigation_sandbox/kosslyn_experiment/output/kosslyn_ssp_cconv_{}seed_tpi1.0_thresh0.4_vel{}_npd{}.npy"
        eps = 0.002
        reaction_time = []
        distance = []
        seeds = [1, 2, 3, 4, 5]
        vel = '0.125'
        npd = 25
        distance_scaling = 2
        center_offset = 0
    else:
        # fname = "/home/ctnuser/ssp_navigation_sandbox/kosslyn_experiment/output/attractor_kosslyn_ssp_cconv_{}seed_tpi1.5_thresh0.4_vel{}_npd{}.npy"
        fname = "/home/ctnuser/ssp_navigation_sandbox/kosslyn_experiment/output/attractor_kosslyn_ssp_cconv_{}seed_tpi2.0_thresh0.45_vel{}_npd{}.npy"
        eps = 0.002
        reaction_time = []
        distance = []
        # seeds = [1, 2, 3, 4, 5]
        # seeds = [1, 2, 3, 5]
        # seeds = [6, 7, 8, 9]
        # seeds = [1, 2, 3, 5, 16, 17, 18, 19, 20]
        # seeds = [23, 24, 25, 26, 28, 29]
        # all the runs that didn't crash
        seeds = [1, 2, 3, 5, 16, 17, 18, 19, 20, 23, 24, 25, 26, 28, 29]
        vel = '0.5'
        npd = 50
        distance_scaling = 4.2 #4#3.5  # arbitrary scaling parameter
        center_offset = 1.2  # account for the radius of the objects in distance calculation
    for seed in seeds:

        data = np.load(fname.format(seed, vel, npd))

        inds = (data[:, 2] > eps) & (data[:, 3] > center_offset)
        reaction_time += list(data[inds, 2])
        distance += list(data[inds, 3])

    model_data = np.zeros((len(reaction_time), 2))
    model_data[:, 0] = distance
    model_data[:, 0] -= center_offset
    model_data[:, 0] *= distance_scaling
    model_data[:, 1] = reaction_time
    model_data[:, 1] += reaction_time_offset

    # cut off data past where human is measured for the plot

    inds = model_data[:, 0] < 19
    model_data = model_data[inds, :]

else:
    fname = sys.argv[1]

    data = np.load(fname)

    # model of the delay in reporting unrelated to the task
    # calculated by aligning the 0 distance value (no delay due to task)

    distance_scaling = 2

    model_data = np.zeros((data.shape[0], 2))
    model_data[:, 0] = data[:, 3] * distance_scaling
    model_data[:, 1] = data[:, 2] + reaction_time_offset

# remove any corrupted data
model_data = model_data[model_data[:, 0] != 0, :]
eps = 0.01
# model_data = model_data[model_data[:, 1] > reaction_time_offset + eps, :]

fig, ax = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)

palette = sns.color_palette(n_colors=2)

ax.scatter(model_data[:, 0], model_data[:, 1], color=palette[1], s=5)
ax.scatter(human_data[:, 0], human_data[:, 1], color=palette[0])

ax.legend(['Model', 'Human'])

# trendlines
p_human = np.poly1d(np.polyfit(human_data[:, 0], human_data[:, 1], 1))
p_model = np.poly1d(np.polyfit(model_data[:, 0], model_data[:, 1], 1))
xs = np.linspace(0, 20, 32)
ax.plot(xs, p_model(xs), color=palette[1], linestyle='-')
ax.plot(xs, p_human(xs), color=palette[0], linestyle='--')


ax.set_xlabel("Distance (cm)", fontsize=14)
ax.set_ylabel("Reaction Time (s)", fontsize=14)


# Calculate r^2 value for human and model
error_human = np.zeros((human_data.shape[0]))
mean_human = np.mean(human_data[:, 1])
for i in range(human_data.shape[0]):
    error_human[i] = p_human(human_data[i, 0]) - human_data[i, 1]
ss_res = np.sum(error_human**2)
ss_tot = np.sum((human_data[:, 1] - mean_human)**2)

r2 = 1 - (ss_res / ss_tot)

print("r2 human: {}".format(r2))

error_model = np.zeros((model_data.shape[0]))
mean_model = np.mean(model_data[:, 1])
for i in range(model_data.shape[0]):
    error_model[i] = p_model(model_data[i, 0]) - model_data[i, 1]
ss_res = np.sum(error_model**2)
ss_tot = np.sum((model_data[:, 1] - mean_model)**2)

r2 = 1 - (ss_res / ss_tot)

print("r2 model: {}".format(r2))

sns.despine()

plt.show()
