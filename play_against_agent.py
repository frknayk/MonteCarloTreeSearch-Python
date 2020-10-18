#!/usr/bin/env python3
import gym
from gym_connect.envs.connect_env import PLAYER
from gym_connect.envs.enums.results_enum import RESULTS
from MCTS.mcts import MonteCarloTreeSearch
from MCTS.node import Node

env_train = gym.make('connect-v0')
root = Node(gym_env=env_train, game_status=RESULTS.NOT_FINISHED, move=None, parent=None)
mcts_agent = MonteCarloTreeSearch(root)
mcts_agent.load('TrainedAgents/mcts_brain_200.pickle')


is_done = False
root_node = mcts_agent.root
env_test = gym.make('connect-v0')

# Enable debug mode for rendering mode
from gym_connect.envs.enums.run_mode import MODE
env_test.mode_game = MODE.RENDER_DEBUG
env_test.set_renderer()

while not is_done:
    
    env_test.render()

    if env_test.PLAYER is PLAYER.FIRST:
        selected_action = env_test.get_action_from_terminal()
        # selected_action = env_test.get_random_action()
        _, _, is_done = env_test.step(selected_action)
        # If it is an unexplored state
        node_selected = root_node.get_children_with_move(selected_action)
        is_eq =  node_selected.env.state == env_test.state
        if False in is_eq:
            print("WRONG STATE ASSIGNMENT !")
            break
        root_node = node_selected
    else:
        # selected_action = env_test.get_random_action()
        selected_action = root_node.select_move()
        # If its an unexplored state
        if selected_action is None:
            mcts_agent = MonteCarloTreeSearch(root_node)
            mcts_agent.train(training_step=1)
            root_node = mcts_agent.root.copy()
            selected_action = root_node.select_move()
        _, _, is_done = env_test.step(selected_action)
        node_selected = root_node.get_children_with_move(selected_action)
        is_eq =  node_selected.env.state == env_test.state
        if False in is_eq:
            print("WRONG STATE ASSIGNMENT !")
            break
        root_node = node_selected
    
    print(root_node.env.state)

print("Winning player : ",root_node.env.PLAYER)