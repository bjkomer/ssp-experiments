# Create a random agent in a dmlab environment and view their trajectories

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# softlinked from: ../../lab/rl/env_interface.py
from env_interface import EnvInterface

def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


level_script = 'contributed/dmlab30/explore_goal_locations_small'

episode_length = 1000
# obs = ('agent_pos', 'goal_pos', 'vision')
obs = ('agent_pos', 'vision')
seed = 13
img_height = 64
img_width = 64

env = EnvInterface(
    width=img_width,
    height=img_height,
    obs=obs,
    episode_length=episode_length,
    level=level_script,
    seed=seed
)


# agent = RandomTrajectoryAgent(obs_index_dict=env.obs_index_dict)


plt.ion()
fig, ax = plt.subplots(1, 2)
img = [ax[0].imshow(np.zeros((img_width, img_height, 3))), ax[1].imshow(np.zeros((img_width, img_height, 3)))]
plt.show()


num_episodes = 10
time_steps = 10000#100
returns = np.zeros((num_episodes,))
for e in range(num_episodes):
    # obs = env.reset(goal_distance=params['goal_distance'])
    obs = env.reset()
    for s in range(episode_length):
        # env.render()
        # env._render_extras()

        img[0].set_data(env.env.observations()['RGB_INTERLEAVED'])
        img[1].set_data(env.env.observations()['DEBUG.CAMERA.TOP_DOWN'].T)
        plt.draw()
        # plt.pause(0.001)
        mypause(0.001)

        # action = agent.act(obs)
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action)
        # print(obs)
        returns[e] += reward
        # if reward != 0:
        #    print(reward)
        # time.sleep(dt)
        # ignoring done flag and not using fixed episodes, which effectively means there is no goal
        # if done:
        #     break

print(returns)
