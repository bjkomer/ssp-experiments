import gym
import tensorflow as tf
import numpy as np
from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.ppo1.pposgd_simple import learn as learn_ppo
from functools import partial
import json

def run_ppo(env, fname, max_timesteps=100000, timesteps_per_batch=2048):

    def gather_results(locals_dict, globals_dict, results):
        results[0].append(locals_dict['timesteps_so_far'])
        results[1].append(np.mean(locals_dict['rewbuffer']))

    results = [[], []]
    results_callback = partial(gather_results, results=results)

    params = {}
    params["max_timesteps"] = max_timesteps
    params["timesteps_per_actorbatch"] = timesteps_per_batch
    params["gamma"] = 0.999
    params["lam"] = 0.995
    # params["num_hid_layers"] = 1
    # params["hid_size"] = 64
    params["clip_param"] = 0.2
    params["entcoeff"] = 0.0
    params["optim_epochs"] = 11
    params["optim_stepsize"] = 0.000892
    params["optim_batchsize"] = 1024
    params["schedule"] = 'linear'

    json.dump(params, open(fname + "/ppo_params.json", "w"))

    params["callback"] = results_callback

    sess = U.make_session(num_cpu=1)
    sess.__enter__()

    #with tf.Session(config=tf.ConfigProto()) as sess:

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(
            name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64,#params["hid_size"],
            num_hid_layers=1,#params["num_hid_layers"]
        )

    learn_ppo(
        env,
        policy_fn,
        **params
    )

    env.close()

    with open("{0}/results.csv".format(fname), 'wb') as f:
        f.write(b'time_steps, average_returns\n')
        # remove the first invalid row when saving
        np.savetxt(f, np.array(results).T[1:,:], delimiter=",")

    saver = tf.train.Saver()
    saver.save(sess, "{0}/model".format(fname))