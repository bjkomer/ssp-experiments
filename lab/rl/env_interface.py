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

    def __init__(self,
                 obs=('agent_pos', 'goal_pos', 'vision'),
                 use_vision=True, use_pos=True,
                 episode_length=1000, width=64, height=64, fps=60,
                 level='contributed/dmlab30/explore_goal_locations_small',
                 record=None, demo=None, demofiles=None, video=None,
                 num_steps=1, seed=1):
        """
        :param obs: a tuple of string values representing what observations to return
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

        self.obs = obs

        # If true, get vision as an observation
        self.use_vision = use_vision

        # If true, get x,y,th as an observation
        self.use_pos = use_pos

        # Number of simulator steps per .step() call
        self.num_steps = num_steps

        # Seed for the maze layout (also goal order)
        # TODO: will need a different seed for maze layout and goal order
        self._seed = seed

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

        # TODO: possibly remove some obs that are not being used during training to speed things up
        obs_list = [
            'RGB_INTERLEAVED',
            'DEBUG.POS.TRANS',
            'DEBUG.POS.ROT',
            'DEBUG.CAMERA.TOP_DOWN',
            'DEBUG.MAZE.LAYOUT',
            'DEBUG.MAZE.VARIATION',
            'DEBUG.CAMERA.PLAYER_VIEW_NO_RETICLE',
            'DEBUG.GOAL.POS',
        ]

        self.obs_first_person = 'RGB_INTERLEAVED'
        # self.obs_first_person = 'DEBUG.CAMERA.PLAYER_VIEW_NO_RETICLE'

        self.env = deepmind_lab.Lab(level, obs_list, config=config)

        self.steps_so_far = 0

        # actions are: look_left, look_right, strafe_left, strafe_right, forward, and backward
        self.action_space = gym.spaces.Discrete(6)

        self.obs_dim = 0
        if 'agent_pos' in self.obs:
            self.obs_dim += 3
        if 'goal_pos' in self.obs:
            self.obs_dim += 2
        if 'vision' in self.obs:
            self.obs_dim += 3 * width * height

        # observations are x,y,th and 3 channel image of width by height
        self.observation_space = gym.spaces.Box(
            low=-1*np.ones((self.obs_dim,)),
            high=np.ones((self.obs_dim,))
        )

    def reset(self):
        self.steps_so_far = 0

        self.env.reset(seed=self._seed)
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

        if 'agent_pos' in self.obs:
            pos = raw_obs['DEBUG.POS.TRANS']
            # Convert to between -1 and 1
            pos_x = ((pos[0] / self.maze_size) - .5) * 2
            pos_y = ((pos[1] / self.maze_size) - .5) * 2

            # angle between -180 and 180 degrees
            # convert it to between -1 and 1
            ang = raw_obs['DEBUG.POS.ROT'][1] / 180.

            # TODO: have option to convert position to ssp here

        if 'goal_pos' in self.obs:
            goal_pos = raw_obs['DEBUG.GOAL.POS']
            # Convert to between -1 and 1
            goal_pos_x = ((goal_pos[0] / self.maze_size) - .5) * 2
            goal_pos_y = ((goal_pos[1] / self.maze_size) - .5) * 2

            # TODO: have option to convert position to ssp here

        if 'vision' in self.obs:
            img = raw_obs[self.obs_first_person]

        if 'vision' in self.obs and 'agent_pos' in self.obs and 'goal_pos' in self.obs:
            obs = np.concatenate([np.array([pos_x, pos_y, ang, goal_pos_x, goal_pos_y]), img.flatten()])
        elif 'vision' in self.obs and 'agent_pos' in self.obs:
            obs = np.concatenate([np.array([pos_x, pos_y, ang]), img.flatten()])
        elif 'agent_pos' in self.obs:
            obs = np.array([pos_x, pos_y, ang])
        elif 'vision' in self.obs:
            # obs = img.flatten().copy()
            obs = np.transpose(img.copy(), (2, 0, 1))
        else:
            # TODO: not all possibilities implemented yet
            raise NotImplementedError("Must use at least one of vision or pose for observations")
        return obs

    def seed(self, seed):
        self._seed = seed
