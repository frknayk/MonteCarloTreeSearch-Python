#!/usr/bin/env python3
import random
import numpy as np
from tqdm import tqdm
from MCTS.node import Node
from gym_connect.envs.enums.player import PLAYER
from gym_connect.envs.enums.results_enum import RESULTS

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
        # Get possible moves in that state
        moves = node.env.get_valid_actions()

        # Nodes to be expanded
        new_nodes = []

        for move in moves:
            # Do not perform any action on original one
            node_current = node.copy()
            _, _, is_done = node_current.env.step(move)
            game_status = RESULTS.NOT_FINISHED
            if is_done:
                game_status = RESULTS.WON
                if node_current.env.PLAYER is PLAYER.SECOND:
                    game_status = RESULTS.LOST

            new_nodes.append( Node(
                gym_env=node_current.env.copy(),
                game_status=game_status,
                move=move,
                parent=node_current) )

        node.set_children(new_nodes)

        return node

    def simulate(self, node):
        """
        Should override current game tree
        to backpropagation step
        """
        is_done = False
        child_selected = node.copy()
        while not is_done:
            moves = child_selected.env.get_valid_actions()
            move_random = random.choice(moves)
            env_copied = child_selected.env.copy()
            _, _, is_done = env_copied.step(move_random)
            child_selected = Node(
                gym_env=env_copied.copy(),
                game_status=self.get_game_status(env_copied.PLAYER, is_done),
                move=move_random,
                parent=child_selected.copy())
        node = child_selected.copy()
        return node

    def backpropagation(self, node):
        parent = node
        while parent is not None:
            parent.games += 1
            if parent.game_status is RESULTS.WON:
                parent.win += 1
            parent = parent.parent
        return parent

    def run(self, root):
        root = self.select(root)
        root = self.expand(root)
        root = self.simulate(root)
        root =self.backpropagation(root)

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