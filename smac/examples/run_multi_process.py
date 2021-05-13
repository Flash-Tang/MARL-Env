from smac.env import StarCraft2Env
import numpy as np

from mlp import Mlp

import multiprocessing
from multiprocessing import Pool
import subprocess

import time


def act(agent_id, obs, avail_actions):
    avail_actions_ind = np.nonzero(avail_actions)[0]

    nn_model = Mlp(avail_actions_ind.size)
    action_index = np.argmax(nn_model(np.expand_dims(np.array(obs), axis=0)))
    action = avail_actions_ind[action_index]

    actions[agent_id] = action


def main():
    n_episodes = 10

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            obs = env.get_obs()
            avail_actions = env.get_avail_actions()

            subprocess
            p = Pool(4)

            start_time = time.time()
            for agent_id in range(n_agents):
                p.apply_async(act, args=(agent_id, obs[agent_id], avail_actions[agent_id]))

            p.close()
            p.join()

            print(f'all threads run {time.time() - start_time}seconds')

            reward, terminated, _ = env.step(actions)
            episode_reward += reward

        print("Total reward in episode {} = {}".format(e, episode_reward))

    env.close()


if __name__ == "__main__":
    env = StarCraft2Env(map_name="val_from_25m", debug=True)

    env_info = env.get_env_info()
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    actions = [1] * n_agents

    main()
