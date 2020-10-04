#!/usr/bin/env python3
import numpy as np
import gym
import gym_connect 
from connect_4_agent.node_connect_4 import NodeConnect4
from MCTS.mcts import MonteCarloTreeSearch

env = gym.make('connect-v0') 
state = env.reset()

is_done = False
while not is_done:
    action = env.get_random_action()
    next_state, reward, is_done = env.step(action)
print("Winning player : ",next_state['player'].name)
print(next_state['board'])

# root = NodeConnect4(state = state)
# mcts = MonteCarloTreeSearch(root)
# best_node = mcts.best_action(simulations_number=10000)