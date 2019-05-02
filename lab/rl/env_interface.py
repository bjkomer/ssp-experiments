import numpy as np
import deepmind_lab
import gym
import gym.spaces


def _action(*entries):
    return np.array(entries, dtype=np.intc)


# ACTIONS = {
#     'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
#     'look_right': _action(20, 0, 0, 0, 0, 0, 0),
#     'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
#     'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
#     'forward': _action(0, 0, 0, 1, 0, 0, 0),
#     'backward': _action(0, 0, 0, -1, 0, 0, 0),
# }

ACTIONS = [
    _action(-20, 0, 0, 0, 0, 0, 0),
    _action(20, 0, 0, 0, 0, 0, 0),
    _action(0, 0, -1, 0, 0, 0, 0),
    _action(0, 0, 1, 0, 0, 0, 0),
    _action(0, 0, 0, 1, 0, 0, 0),
    _action(0, 0, 0, -1, 0, 0, 0),
]


class EnvInterface(gym.Env):

    def __init__(self, episode_length=1000, width=64, height=64, fps=60,
                 level='contributed/dmlab30/explore_goal_locations_small',
                 record=None, demo=None, demofiles=None, video=None,
                 num_steps=1, seed=1):
        """
        :param episode_length: episode length
        :param width: width of visual image
        :param height: height of visual image
        :param fps: fps of simulator
        :param level: level script
        :param record:
        :param demo:
        :param demofiles:
        :param video:
        :param num_steps: number of simulator steps for each env.step() call. action repeat
        :param seed: seed for level to use
        """

        # Guessing parameters of the maze
        # https://github.com/deepmind/lab/blob/master/game_scripts/factories/explore/factory.lua
        # position ranges from about 0 to 1000 in the environment
        self.block_size = 100
        self.n_blocks = 11
        self.maze_size = self.n_blocks * self.block_size

        # Number of simulator steps per .step() call
        self.num_steps = num_steps

        # Seed for the maze layout (also goal order)
        self.seed = seed

        self.episode_length = episode_length

        config = {
            'fps': str(fps),
            'width': str(width),
            'height': str(height)
        }
        if record:
            config['record'] = record
        if demo:
            config['demo'] = demo
        if demofiles:
            config['demofiles'] = demofiles
        if video:
            config['video'] = video

        obs_list = [
            'RGB_INTERLEAVED',
            'DEBUG.POS.TRANS',
            'DEBUG.POS.ROT',
            'DEBUG.CAMERA.TOP_DOWN',
            'DEBUG.MAZE.LAYOUT',
            'DEBUG.MAZE.VARIATION',
            'DEBUG.CAMERA.PLAYER_VIEW_NO_RETICLE',
        ]

        self.obs_first_person = 'RGB_INTERLEAVED'
        # self.obs_first_person = 'DEBUG.CAMERA.PLAYER_VIEW_NO_RETICLE'

        self.env = deepmind_lab.Lab(level, obs_list, config=config)

        self.steps_so_far = 0

        # actions are: look_left, look_right, strafe_left, strafe_right, forward, and backward
        self.action_space = gym.spaces.Discrete(6)

        # observations are x,y,th and 3 channel image of width by height
        self.observation_space = gym.spaces.Box(
            low=-1*np.ones((3 + 3 * width * height,)),
            high=np.ones((3 + 3 * width * height,))
        )

    def reset(self):
        self.steps_so_far = 0

        self.env.reset(seed=self.seed)
        raw_obs = self.env.observations()

        return self.parse_raw_obs(raw_obs)

    def step(self, action):
        """
        Apply action to the environment and move forward by one time-step
        :param action: integer corresponding to one of the discrete actions available
        :return: obs, reward, done, info
        """

        reward = self.env.step(ACTIONS[action], num_steps=self.num_steps)
        raw_obs = self.env.observations()

        self.steps_so_far += 1

        done = self.steps_so_far >= self.episode_length

        return self.parse_raw_obs(raw_obs), reward, done, {}

    def parse_raw_obs(self, raw_obs):
        """
        take the raw observations from the simulator, convert them into an observation vector
        :param raw_obs: obs returned from self.env.step()
        :return: observation vector
        """

        pos = raw_obs['DEBUG.POS.TRANS']
        # Convert to between -1 and 1
        pos_x = ((pos[0] / self.maze_size) - .5) * 2
        pos_y = ((pos[1] / self.maze_size) - .5) * 2

        # angle between -180 and 180 degrees
        # convert it to between -1 and 1
        ang = raw_obs['DEBUG.POS.ROT'][1] / 180.

        img = raw_obs[self.obs_first_person]

        # TODO: have option to convert position to ssp here

        obs = np.concatenate([pos_x, pos_y, ang, img.flatten()])
        return obs
