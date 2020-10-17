#!/usr/bin/env python3
import gym
from gym_connect.envs.connect_env import PLAYER
from gym_connect.envs.enums.results_enum import RESULTS
from MCTS.mcts import MonteCarloTreeSearch
from MCTS.node import Node

env_train = gym.make('connect-v0')
root = Node(gym_env=env_train, game_status=RESULTS.NOT_FINISHED, move=None, parent=None)
mcts_agent = MonteCarloTreeSearch(root)
mcts_agent.train(50)

env_test = gym.make('connect-v0')
is_done = False
node = mcts_agent.root
while not is_done:
    selected_action = None
    
    # if env_test.PLAYER is PLAYER.FIRST:
    #     selected_action = env_test.get_action_from_terminal()
    #     new_node = node.get_children_with_move(selected_action)
    #     node = mcts_agent.train_mcts_online(node, 2000).get_children_with_move(selected_action)
    # else:
    #     node = mcts_agent.train_mcts_online(node, 2000)
    #     _, selected_action = mcts_agent.root.select_move()

    next_state, reward, is_done = env_test.step(selected_action)

    if next_state is not None:
        print(next_state['board'])

print("Winning player : ",env_test.PLAYER)