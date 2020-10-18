#!/usr/bin/env python3
import random
from gym.logger import debug
import numpy as np
from tqdm import tqdm
from MCTS.node import Node
from gym_connect.envs.enums.player import PLAYER
from gym_connect.envs.enums.results_enum import RESULTS
import pickle

class MonteCarloTreeSearch(object):

    def __init__(self, node):
        # Root node of the tree
        self.root = node

    def select(self, root):
        """Select highest uct"""
        node = root
        while node.children is not None:
            ucts = [child.get_uct() for child in node.children]
            if None in ucts:
                node = random.choice(node.children)
            else:
                node = node.children[np.argmax(ucts)]
        return node

    def expand(self, node):
        is_node_expanded = False

        # Get possible moves in that state
        moves = node.env.get_valid_actions()

        # Nodes to be expanded
        new_nodes = []
        
        if node.game_status is RESULTS.NOT_FINISHED:
            is_node_expanded = True

            for move in moves:
                # Do not perform any action on original one
                node_current = node.copy()
                _, _, is_done = node_current.env.step(move)
                game_status = RESULTS.NOT_FINISHED
                if is_done:
                    game_status = RESULTS.WON
                    if node_current.env.PLAYER is PLAYER.SECOND:
                        game_status = RESULTS.LOST

                new_nodes.append( 
                    Node(
                        gym_env=node_current.env.copy(),
                        game_status=game_status,
                        move=move,
                        parent=node
                        ) )

            node.set_children(new_nodes)

        return node, is_node_expanded

    def simulate(self, node, is_node_expanded):
        """
        Should override current game tree
        to backpropagation step
        """
        final_game_status = node.game_status
        if is_node_expanded:
            is_done = False
            child_selected = node.copy()
            while not is_done:
                # moves = child_selected.env.get_valid_actions()
                # move_random = random.choice(moves)
                result, move_random = child_selected.random_play_improved(child_selected)
                env_copied = child_selected.env.copy()
                _, _, is_done = env_copied.step(move_random)
                game_status = self.get_game_status(env_copied.PLAYER, is_done)
                child_selected = Node(
                    gym_env=env_copied.copy(),
                    game_status=game_status,
                    move=move_random,
                    parent=child_selected.copy())
                final_game_status = game_status
        return node, final_game_status

    def backpropagation(self, node, game_status):
        parent = node
        while parent is not None:
            parent.games += 1
            if game_status is RESULTS.WON:
                parent.win += 1
            parent = parent.parent

    def run(self, root):
        node_selected = self.select(root)
        if node_selected.game_status is RESULTS.NOT_FINISHED:
            node_selected, is_node_expanded = self.expand(node_selected)
            node_selected, game_status = self.simulate(node_selected, is_node_expanded)
            self.backpropagation(node_selected,game_status)

    def train(self, training_step):
        for _ in tqdm(range(0, training_step),desc='MCTS simulation'):
            self.run(self.root)

    def node_children_info(self, node):
        if node.children is not None:
            for child in node.children:
                print(" ----------------- ")
                print("move         :",child.move)
                print("win          :",child.win)
                print("games        :",child.games)
                print("children     :",child.children)
                print("env          :",child.env)
                print("game_status  :",child.game_status)

    def get_game_status(self, player, is_done):
        if is_done:
            if player is PLAYER.SECOND:
                return RESULTS.LOST
            return RESULTS.WON
        return RESULTS.NOT_FINISHED

    def save(self):
        total_play = 0
        for child in self.root.children:
            total_play += child.games
        agent_name = 'TrainedAgents/mcts_brain_' + str(total_play) +  '.pickle' 
        with open(agent_name, 'wb') as handle:
            pickle.dump(self.root, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Agent is sucesfully saved !")

    def load(self, agent_path):
        agent_name = agent_path
        root_loaded = None
        with open(agent_name, 'rb') as handle:
            root_loaded  = pickle.load(handle)
        if root_loaded is not None:
            self.root = root_loaded
            print("Agent brain is loaded succesfully !")
        else:
            print("Agent brain could not be loaded")
