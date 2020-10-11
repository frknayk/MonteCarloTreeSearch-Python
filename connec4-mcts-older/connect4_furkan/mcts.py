#!/usr/bin/env python3
import random
import numpy as np
from tqdm import tqdm

from .node import Node

# TODO:change architecture ! MCTS must be high level class 
from .connect4 import get_player_to_play, valid_move
from .connect4 import play, can_play
from .mcts_tools import random_play_improved


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
        if node.winner == 0:
            # To control if the leaf expanded 
            # needed at the simulation step
            is_node_expanded = True

            # Get possible moves in that state
            moves = valid_move(node.state)

            # Get possible states to set them as node later
            states = []
            for move in moves:
                # game_status : 0 if game continues
                next_state, game_status = play(node.state, move)
                states.append( ( (next_state, game_status), move) )

            # Expand : Set new states as nodes
            new_nodes = []
            for next_state, move in states:
                next_state,game_status = next_state
                new_node = Node(state=next_state, winning=game_status, move=move, parent=node)
                new_nodes.append( new_node )

            node.set_children(new_nodes)

        return node, is_node_expanded

    def expansion(self, node, is_node_expanded):
        game_status = node.winner
        if is_node_expanded:        
            # Get winner nodes
            winner_nodes = []
            for child_node in node.children :
                if child_node.winner:
                    winner_nodes.append(child_node)
            # If there is a winner node, 
            if len(winner_nodes) > 0:
                node = winner_nodes[0]
                game_status = node.winner
            else:
                node = random.choice(node.children)
                game_status = random_play_improved(node.state)
        return game_status, node
    
    def backpropagation(self, game_status, selected_node):
        parent = selected_node
        while parent is not None:
            parent.games += 1
            if game_status != 0 and get_player_to_play(parent.state) != game_status:
                parent.win += 1
            parent = parent.parent
    
    def mcts_one_step(self, root):
        node_selected = self.selection(root)
        if len(valid_move(node_selected.state)) > 0:
            node_selected, is_node_expanded = self.playout(node_selected)
            game_status, node_selected = self.expansion(node_selected,is_node_expanded)
            self.backpropagation(game_status, node_selected)
        return root

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

    def depreceated_expand_simulate(self, node):
        victorious = None

        moves = valid_move(node.state)

        # If game continues expand and simulate
        if node.winner == 0:

            # states = [(play(node.state, move), move) for move in moves]
            # node.set_children([Node(state_winning[0], state_winning[1], move=move, parent=node) for state_winning, move in states])
            # winner_nodes = [n for n in node.children if n.winner]

            ########### EXPAND #########
            # Get possible states to set them as node later
            states = []
            for move in moves:
                # game_status : 0 if game continues
                next_state, game_status = play(node.state, move)
                states.append( ( (next_state, game_status), move) )
            
            # Expand : Set new states as nodes
            new_nodes = []
            for next_state, move in states:
                next_state,game_status = next_state
                new_node = Node(state=next_state, winning=game_status, move=move, parent=node)
                new_nodes.append( new_node )    
            node.set_children(new_nodes)

            ########### SIMULATE #########            
            # Get winner nodes
            winner_nodes = []
            for child_node in node.children :
                if child_node.winner:
                    winner_nodes.append(child_node)
            # If there is a winner node, 
            if len(winner_nodes) > 0:
                node = winner_nodes[0]
                victorious = node.winner
            else:
                node = random.choice(node.children)
                victorious = random_play_improved(node.state)
        else:
            victorious = node.winner
        
        return victorious, node
