import time
import threading
from threading import Thread

from smac.examples.mlp import Mlp
import numpy as np


def threadFunc(threadName):
    print("\r\n%s start" % threadName)
    time.sleep(5)
    print("\r\n%s end" % threadName)
    pass


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

    def get_action(self):
        return self.agent_id, self.action


start = time.time()
threads = []

obs = np.random.random(size=(5050,)).astype(np.float32)
avail_actions = [1] * 16
for agent in range(1000):
    # thread = threading.Thread(target=threadFunc, args=("Thread%s" % index,))
    thread = AgentThread(agent, obs, avail_actions)
    thread.start()
    threads.append(thread)

for t in threads:
    t.join()

print("thread finished , cost %s s" % (time.time() - start))
