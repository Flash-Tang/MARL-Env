from smac.env import StarCraft2Env
import numpy as np

from mininet.topo import Topo, SingleSwitchTopo
from mininet.net import Mininet
from mininet.log import lg, info
from mininet.cli import CLI

from threading import Thread
from mlp import Mlp

import time


class simpleMultiLinkTopo(Topo):
    "Simple topology with multiple links"

    # pylint: disable=arguments-differ
    def build(self, **_kwargs):
        for i in range(n_agents):
            hosts[i] = self.addHost(f'h{i}')

        for i in range(0, n_agents, 3):
            self.addLink(hosts[i], hosts[i + 1])
            self.addLink(hosts[i], hosts[i + 2])
            self.addLink(hosts[i + 1], hosts[i + 2])


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
        # print(f'thread {threading.current_thread().name} runs {time.time() - start_time}')
        # print(f'thread {threading.current_thread().name} terminates')

    def get_action(self):
        return self.agent_id, self.action


def main():
    lg.setLogLevel('info')
    topo = simpleMultiLinkTopo()
    net = Mininet(topo=topo)
    net.start()

    n_episodes = 1

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0

        step = 0

        for agent_id in range(0, n_agents, 3):
            h1 = net.get(hosts[agent_id])
            h2 = net.get(hosts[agent_id + 1])
            h3 = net.get(hosts[agent_id + 2])

            h1.cmd(f'python3 ~/PycharmProjects/MARL-Env/smac/examples/node_recv.py -i {h3.IP()} &')
            h2.cmd(f'python3 ~/PycharmProjects/MARL-Env/smac/examples/node_recv.py -i {h3.IP()} &')

        for _ in range(1):
            obs = env.get_obs()
            avail_actions = env.get_avail_actions()

            threads = []

            start_time = time.time()

            for agent_id in range(0, n_agents, 3):
                h1 = net.get(hosts[agent_id])
                h2 = net.get(hosts[agent_id + 1])
                h3 = net.get(hosts[agent_id + 2])

                # CLI(net)
                h3.cmd(f'python3 ~/PycharmProjects/smac/smac/examples/node_send.py -i1 {h1.IP()} -i2 {h2.IP()}')

            print('---------step---------')
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

            step += 1

        print("Total reward in episode {} = {}".format(e, episode_reward))

    env.close()
    net.stop()


if __name__ == "__main__":
    env = StarCraft2Env(map_name="3m", debug=True)

    env_info = env.get_env_info()
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    hosts = [''] * n_agents

    main()
