import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
from PPO import Agent
import numpy as np
from pathplanning_env import pathplanning
import time
import matplotlib.pyplot as plt

env = pathplanning()

agent = Agent()
agent.load()

max_average_rewards = -1e6
average_rewards = 0
sum_rewards = 0

for count in range(10000):
        
    s = env.reset()
    rewards = 0
    plot = np.zeros((2, 1000))
        
    for i in range(1000):
        s = np.array(s)

        plot[0,i] = s[0]
        plot[1,i] = s[1]

        a = agent.choose_action(torch.tensor(s, dtype=torch.float))
        s_, r, done, _ = env.step(a)
            
        rewards += r
            
        agent.push_data((s, a, r, s_, done))
        s = s_
           
        if done:
            break
            
    sum_rewards += rewards
    agent.update()

    if count > 0 and count % 10 == 0:
        
        average_rewards = sum_rewards/10
        sum_rewards = 0
        print(count-9, '-', count, 'average_rewards:', average_rewards, 'max_average_rewards:', max_average_rewards, end='\r')

        if max_average_rewards < average_rewards:
            max_average_rewards = average_rewards
            agent.save()

        if max_average_rewards>-1050:
            plt.plot(plot[0], plot[1], color='red', label='USV 0')
            goal = plt.Circle((0, 0), 1, color='green', fill=False)
            plt.gca().add_patch(goal)
            plt.axis('equal')
            plt.show()
            time.sleep(np.inf)