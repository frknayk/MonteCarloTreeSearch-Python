#!/usr/bin/env python3
import numpy as np
import gym
import gym_connect 
from connect_4_agent.node_connect_4 import NodeConnect4
from MCTS.mcts import MonteCarloTreeSearch

env = gym.make('connect-v0')

is_done = False
while not is_done:
    root = NodeConnect4(gym_env = env.copy())
    mcts = MonteCarloTreeSearch(root)
    best_child, best_child_action = mcts.best_action(simulations_number=500)
    next_state, reward, is_done = env.step(best_child_action)
    print(next_state['board'])

print("Winning player : ",next_state['player'].name)