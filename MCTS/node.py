#!/usr/bin/env python3

import numpy as np
from abc import ABC, abstractmethod


class Node(ABC):

    def __init__(self, gym_env, parent=None):
        """
        Parameters
        ----------
        env : gym environment
        parent : Node
        """
        self.env = gym_env
        self.parent = parent
        self.children = []

    @property
    @abstractmethod
    def untried_actions(self):
        """

        Returns
        -------
        list of mctspy.games.common.AbstractGameAction

        """
        pass

    @property
    @abstractmethod
    def q(self):
        pass

    @property
    @abstractmethod
    def n(self):
        pass

    @abstractmethod
    def expand(self):
        pass

    @abstractmethod
    def is_terminal_node(self):
        pass

    @abstractmethod
    def rollout(self):
        pass

    @abstractmethod
    def backpropagate(self, reward):
        pass

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)], np.argmax(choices_weights)

    def rollout_policy(self, possible_moves):     
        return possible_moves[np.random.randint(len(possible_moves))]
