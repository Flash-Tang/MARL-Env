from smac.env import StarCraft2Env
import numpy as np

from smac.examples.mlp import Mlp

import threading
from threading import Thread
import time


class AgentThread(Thread):
    def __init__(self, agent_id, obs, avail_actions):
        super(AgentThread, self).__init__()
        self.agent_id = agent_id
        self.obs = obs
        self.avail_actions = avail_actions

    def run(self):
        start_time = time.time()

        # print(f'thread {threading.current_thread().name} starts')

        # avail_actions = env.get_avail_agent_actions(self.agent_id)
        avail_actions_ind = np.nonzero(self.avail_actions)[0]

        # obs = env.get_obs_agent(self.agent_id)
        nn_model = Mlp(avail_actions_ind.size)
        action_index = np.argmax(nn_model(np.expand_dims(np.array(self.obs), axis=0)))
        self.action = avail_actions_ind[action_index]

        # self.action = 4

        run_time = time.time() - start_time
        # if run_time > 4:
        # print(f'thread {threading.current_thread().name} runs {time.time() - start_time}')
        # print(f'thread {threading.current_thread().name} terminates')

    def get_action(self):
        return self.agent_id, self.action


def main():
    n_episodes = 10

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            obs = env.get_obs()
            avail_actions = env.get_avail_actions()

            threads = []

            start_time = time.time()

            for agent_id in range(n_agents):
                t = AgentThread(agent_id, obs[agent_id], avail_actions[agent_id])
                t.start()
                threads.append(t)
                # t.join()

            for t in threads:
                t.join()

            print(f'all threads run {time.time() - start_time}seconds')

            actions = [0] * n_agents
            for t in threads:
                agent, action = t.get_action()
                actions[agent] = action


            reward, terminated, _ = env.step(actions)
            episode_reward += reward

        print("Total reward in episode {} = {}".format(e, episode_reward))

    env.close()


if __name__ == "__main__":
    env = StarCraft2Env(map_name="3m", debug=True)

    env_info = env.get_env_info()
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    main()
