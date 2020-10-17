import re
from gym_connect.envs.enums.results_enum import RESULTS
import numpy as np
import gym
from copy import deepcopy

class Node:
    #TODO: make game_status enum
    def __init__(self, gym_env, game_status, move, parent):
        self.parent = parent
        self.move = move
        self.win = 0
        self.games = 0
        self.children = None
        self.env = gym_env
        self.game_status = game_status

    def set_children(self, children):
        self.children = children

    def get_uct(self):
        if self.games == 0:
            return None
        return (self.win/self.games) + np.sqrt(2*np.log(self.parent.games)/self.games)
    
    def copy(self):
        """Return deep copied env"""
        return deepcopy(self)
