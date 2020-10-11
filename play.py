#!/usr/bin/env python3
import gym
from gym_connect.envs.connect_env import PLAYER
from gym_connect.envs.enums.results_enum import RESULTS
from MCTS.mcts import MonteCarloTreeSearch
from MCTS.node import Node

env_train = gym.make('connect-v0')
root = Node(gym_env=env_train, game_status=RESULTS.NOT_FINISHED, move=None, parent=None)
mcts_agent = MonteCarloTreeSearch(root)
mcts_agent.train(100)

env_test = gym.make('connect-v0')
is_done = False
while not is_done:
    selected_action = None
    if env_test.PLAYER is PLAYER.FIRST:
        mcts_agent.root.env = env_test.copy()
        mcts_agent.train_mcts_online(mcts_agent.root,1000)
        _, selected_action = mcts_agent.root.select_move()
    else:
        selected_action = env_test.get_action_from_terminal()
        mcts_agent.root.env = env_test.copy()
        mcts_agent.train_mcts_online(mcts_agent.root,1000)

    next_state, reward, is_done = env_test.step(selected_action)

    if next_state is None:
        debug = True
    else:
        print(next_state['board'])

print("Winning player : ",env_test.PLAYER)
