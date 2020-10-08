#!/usr/bin/env python3
import numpy as np
import gym
import gym_connect 
from connect_4_agent.node_connect_4 import NodeConnect4
from MCTS.mcts import MonteCarloTreeSearch

env = gym.make('connect-v0') 
root = NodeConnect4(gym_env = env)
mcts = MonteCarloTreeSearch(root)
best_node = mcts.best_action(simulations_number=10000)