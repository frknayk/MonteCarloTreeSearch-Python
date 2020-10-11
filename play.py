#!/usr/bin/env python3
import numpy as np
import gym
import gym_connect 
from gym_connect.envs.connect_env import PLAYER
from connect_4_agent.node_connect_4 import NodeConnect4
from MCTS.mcts import MonteCarloTreeSearch

env = gym.make('connect-v0')

# Training
root = NodeConnect4(gym_env = env.copy())
mcts = MonteCarloTreeSearch(root)
best_child, best_child_action = mcts.best_action(simulations_number=500)


is_done = False
# while not is_done:
    
#     selected_action = None
    
#     if env.PLAYER is PLAYER.FIRST:
#         root = NodeConnect4(gym_env = env.copy())
#         mcts.root = root
#         best_child, best_child_action = mcts.best_action(simulations_number=100)
#         selected_action = best_child_action
#     else:
#         selected_action = env.get_action_from_terminal()
    
#     next_state, reward, is_done = env.step(selected_action)

#     if next_state is None:
#         debug = True
#     else:
#         print(next_state['board'])

# print("Winning player : ",env.PLAYER)