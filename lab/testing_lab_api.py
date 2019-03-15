import deepmind_lab
import numpy as np
import six
import random
import argparse
import matplotlib.pyplot as plt

"""
Levels to use:
contributed/dmlab30/explore_goal_locations_small
contributed/dmlab30/explore_goal_locations_large
contributed/dmlab30/explore_obstructed_goals_small
contributed/dmlab30/explore_obstructed_goals_large
"""

def _action(*entries):
    return np.array(entries, dtype=np.intc)


class DiscretizedRandomAgent(object):
  """Simple agent for DeepMind Lab."""

  ACTIONS = {
      'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
      'look_right': _action(20, 0, 0, 0, 0, 0, 0),
      'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
      'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
      'forward': _action(0, 0, 0, 1, 0, 0, 0),
      'backward': _action(0, 0, 0, -1, 0, 0, 0),
  }

  ACTION_LIST = list(six.viewvalues(ACTIONS))

  rewards = 0

  def step(self, reward, unused_image):
    """Gets an image state and a reward, returns an action."""
    self.rewards += reward
    # action = random.choice(DiscretizedRandomAgent.ACTION_LIST)
    # print("action", action)
    # return action
    return random.choice(DiscretizedRandomAgent.ACTION_LIST)

class DiscretizedDerivativeRandomAgent(object):
  """Simple agent for DeepMind Lab."""

  ACTIONS = {
      'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
      'look_right': _action(20, 0, 0, 0, 0, 0, 0),
      'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
      'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
      'forward': _action(0, 0, 0, 1, 0, 0, 0),
      'backward': _action(0, 0, 0, -1, 0, 0, 0),
  }

  ACTION_LIST = list(six.viewvalues(ACTIONS))

  rewards = 0

  action = np.zeros((7,), dtype=np.intc)

  dt = 1#0.01

  def step(self, reward, unused_image):
    """Gets an image state and a reward, returns an action."""
    self.rewards += reward
    action = random.choice(DiscretizedRandomAgent.ACTION_LIST)
    self.action += action#*self.dt

    return self.action


class SpringAgent(object):
    """A random agent using spring-like forces for its action evolution."""

    def __init__(self, action_spec):
        self.action_spec = action_spec
        print('Starting random spring agent. Action spec:', action_spec)

        self.omega = np.array([
            0.1,  # look left-right
            0.1,  # look up-down
            0.1,  # strafe left-right
            0.1,  # forward-backward
            0.0,  # fire
            0.0,  # jumping
            0.0  # crouching
        ])

        self.velocity_scaling = np.array([2.5, 2.5, 0.01, 0.01, 1, 1, 1])

        self.indices = {a['name']: i for i, a in enumerate(self.action_spec)}
        self.mins = np.array([a['min'] for a in self.action_spec])
        self.maxs = np.array([a['max'] for a in self.action_spec])
        self.reset()

        self.rewards = 0

    def critically_damped_derivative(self, t, omega, displacement, velocity):
        r"""Critical damping for movement.
        I.e., x(t) = (A + Bt) \exp(-\omega t) with A = x(0), B = x'(0) + \omega x(0)
        See
          https://en.wikipedia.org/wiki/Damping#Critical_damping_.28.CE.B6_.3D_1.29
        for details.
        Args:
          t: A float representing time.
          omega: The undamped natural frequency.
          displacement: The initial displacement at, x(0) in the above equation.
          velocity: The initial velocity, x'(0) in the above equation
        Returns:
           The velocity x'(t).
        """
        a = displacement
        b = velocity + omega * displacement
        return (b - omega * t * (a + t * b)) * np.exp(-omega * t)

    def step(self, reward, unused_frame):
        """Gets an image state and a reward, returns an action."""
        self.rewards += reward

        action = (self.maxs - self.mins) * np.random.random_sample(
            size=[len(self.action_spec)]) + self.mins

        # Compute the 'velocity' 1 time unit after a critical damped force
        # dragged us towards the random `action`, given our current velocity.
        self.velocity = self.critically_damped_derivative(1, self.omega, action,
                                                          self.velocity)

        # Since walk and strafe are binary, we need some additional memory to
        # smoothen the movement. Adding half of action from the last step works.
        self.action = self.velocity / self.velocity_scaling + 0.5 * self.action

        # Fire with p = 0.01 at each step
        self.action[self.indices['FIRE']] = int(np.random.random() > 0.99)

        # Jump/crouch with p = 0.005 at each step
        self.action[self.indices['JUMP']] = int(np.random.random() > 0.995)
        self.action[self.indices['CROUCH']] = int(np.random.random() > 0.995)

        # Clip to the valid range and convert to the right dtype
        return self.clip_action(self.action)

    def clip_action(self, action):
        return np.clip(action, self.mins, self.maxs).astype(np.intc)

    def reset(self):
        self.velocity = np.zeros([len(self.action_spec)])
        self.action = np.zeros([len(self.action_spec)])


def run(length, width, height, fps, level, record, demo, demofiles, video):
    """Spins up an environment and runs the random agent."""
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

    obs_first_person = 'RGB_INTERLEAVED'
    # obs_first_person = 'DEBUG.CAMERA.PLAYER_VIEW_NO_RETICLE'

    env = deepmind_lab.Lab(level, obs_list, config=config)

    env.reset()

    # agent = DiscretizedRandomAgent()
    # agent = DiscretizedDerivativeRandomAgent()
    agent = SpringAgent(action_spec=env.action_spec())

    reward = 0

    plt.ion()
    fig, ax = plt.subplots(1, 2)
    obs = env.observations()
    print(obs['DEBUG.CAMERA.TOP_DOWN'].T.shape)
    print(obs[obs_first_person].shape)
    # print(obs['DEBUG.MAZE.LAYOUT'])
    # print("")
    # print(obs['DEBUG.MAZE.VARIATION'])
    # assert False
    img = [ax[0].imshow(obs[obs_first_person]), ax[1].imshow(obs['DEBUG.CAMERA.TOP_DOWN'].T)]
    plt.show()

    for _ in six.moves.range(length):
        if not env.is_running():
            print('Environment stopped early')
            env.reset()
            agent.reset()

        # fig.canvas.draw()
        obs = env.observations()
        action = agent.step(reward, obs[obs_first_person])
        reward = env.step(action, num_steps=1)

        print(obs['DEBUG.POS.TRANS'])
        img[0].set_data(obs[obs_first_person])
        img[1].set_data(obs['DEBUG.CAMERA.TOP_DOWN'].T)
        plt.draw()
        plt.pause(0.001)

    # plt.imshow(obs['RGB_INTERLEAVED'])
    # plt.show()

    print('Finished after %i steps. Total reward received is %f'
        % (length, agent.rewards))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--length', type=int, default=1000,
                      help='Number of steps to run the agent')
  parser.add_argument('--width', type=int, default=64,
                      help='Horizontal size of the observations')
  parser.add_argument('--height', type=int, default=64,
                      help='Vertical size of the observations')
  parser.add_argument('--fps', type=int, default=60,
                      help='Number of frames per second')
  parser.add_argument('--runfiles_path', type=str, default=None,
                      help='Set the runfiles path to find DeepMind Lab data')
  parser.add_argument('--level_script', type=str,
                      # default='tests/empty_room_test',
                      default='contributed/dmlab30/explore_goal_locations_small',
                      help='The environment level script to load')
  parser.add_argument('--record', type=str, default=None,
                      help='Record the run to a demo file')
  parser.add_argument('--demo', type=str, default=None,
                      help='Play back a recorded demo file')
  parser.add_argument('--demofiles', type=str, default=None,
                      help='Directory for demo files')
  parser.add_argument('--video', type=str, default=None,
                      help='Record the demo run as a video')

  args = parser.parse_args()
  if args.runfiles_path:
    deepmind_lab.set_runfiles_path(args.runfiles_path)
  run(args.length, args.width, args.height, args.fps, args.level_script,
      args.record, args.demo, args.demofiles, args.video)
