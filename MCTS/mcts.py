#!/usr/bin/env python3
import random
from gym.logger import debug
import numpy as np
from tqdm import tqdm

from .node import Node

from .mcts_tools import random_play_improved
from gym_connect.envs.enums.player import PLAYER
from gym_connect.envs.enums.results_enum import RESULTS

class MonteCarloTreeSearch(object):

    def __init__(self, node):
        # Root node of the tree
        self.root = node
    
    def selection(self, root):
        """
        The selection strategy is applied recursively until
        an unknown position is reached

        Args:
            root (Node): Root node of the tree

        Returns:
            Node 
        """
        # Node to be selected
        node = root
        # selection
        while node.children is not None:
            # Select highest uct
            ucts = [child.get_uct() for child in node.children]
            if None in ucts:
                node = random.choice(node.children)
            else:
                node = node.children[np.argmax(ucts)]
        return node

    def playout(self, node):
        """One simulated game is played"""
        is_node_expanded = False

        # Only expand node when the selected node 
        # is not the terminator node
        if node.game_status is RESULTS.NOT_FINISHED:
            # To control if the leaf expanded 
            # needed at the simulation step
            is_node_expanded = True

            # Get possible moves in that state
            # moves = valid_move(node.state)
            moves = node.env.get_valid_actions()

            # Get possible states to set them as node later
            states = []
            for move in moves:
                node_current = node.copy()
                # game_status : 0 if game continues
                _, _, is_done = node_current.env.step(move)
                
                # TODO: get this from gym-env directly
                game_status = RESULTS.NOT_FINISHED
                if is_done:
                    if node_current.env.PLAYER is PLAYER.FIRST:
                        game_status = RESULTS.WON

                states.append( ( (node_current, game_status), move) )

            # Expand : Set new states as nodes
            new_nodes = []
            for next_env_info, move in states:
                next_node,game_status = next_env_info
                new_node = Node(gym_env=next_node.env.copy(), game_status=game_status, move=move, parent=node)
                new_nodes.append( new_node )

            node.set_children(new_nodes)

        return node, is_node_expanded

    def expansion(self, node, is_node_expanded):
        game_status = node.game_status
        if is_node_expanded:        
            # Get game_status nodes
            winning_game_status_nodes = []
            for child_node in node.children :
                # If the child node is winning node
                if child_node.game_status is RESULTS.WON:
                    winning_game_status_nodes.append(child_node)
            # If there is a game_status node, 
            if len(winning_game_status_nodes) > 0:
                node = winning_game_status_nodes[0]
                game_status = node.game_status
            else:
                # Expand random child node 
                # And check for winning/losing move
                node = random.choice(node.children)
                game_status = random_play_improved(node.copy())
            
        return game_status, node
    
    def backpropagation(self, game_status, selected_node):
        parent = selected_node
        while parent is not None:
            parent.games += 1
            # If game is done and the winning is MCTS
            if game_status is RESULTS.WON:
                parent.win += 1
            parent = parent.parent

    def mcts_one_step(self, root):
        node_selected = self.selection(root)
        if node_selected.game_status is RESULTS.NOT_FINISHED:
            node_selected, is_node_expanded = self.playout(node_selected)
            game_status, node_selected = self.expansion(node_selected,is_node_expanded)
            self.backpropagation(game_status, node_selected)

    def train(self, training_step):
        for _ in tqdm(range(0, training_step),desc='MCTS simulation'):
            self.mcts_one_step(self.root)

    def train_mcts_online(self,node, training_time):
        self.root = node
        import time
        start = int(round(time.time() * 1000))
        current = start
        # while time resources left, search for best move
        while (current - start) < training_time:
            self.mcts_one_step(self.root)
            current = int(round(time.time() * 1000))
        return self.root
