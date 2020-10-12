#!/usr/bin/env python3
import gym
from gym_connect.envs.connect_env import PLAYER
from gym_connect.envs.enums.results_enum import RESULTS
from MCTS.mcts import MonteCarloTreeSearch
from MCTS.node import Node

env_train = gym.make('connect-v0')
root = Node(gym_env=env_train, game_status=RESULTS.NOT_FINISHED, move=None, parent=None)
mcts_agent = MonteCarloTreeSearch(root)
mcts_agent.train(250)
# Save trained agents under /TrainedAgents folder
mcts_agent.save()
