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
        self.state = self.env.state
        self.game_status = game_status

    def set_children(self, children):
        self.children = children

    def get_uct(self):
        if self.games == 0:
            return None
        return (self.win/self.games) + np.sqrt(2*np.log(self.parent.games)/self.games)


    def select_move(self):
        """
        Select best move and advance
        :return:
        """
        if self.children is None:
            return None, None

        winners = [child for child in self.children if child.game_status is RESULTS.WON]
        if len(winners) > 0:
            return winners[0], winners[0].move

        games = [child.win/child.games if child.games > 0 else 0 for child in self.children]
        
        # for child in self.children:
        #     print(child.games, child.win)

        for child in self.children:
            ratio = -1 if child.games == 0 else child.win/child.games
            msg = "#WIN : {0} -- #GAMES : {1} -- WIN_RATIO : {2:.3f} -- MOVE : {3}".format(
                child.win,
                child.games,
                ratio,
                child.move
            )
            print(msg)

        best_child = self.children[np.argmax(games)]
        return best_child, best_child.move


    def get_children_with_move(self, move):
        if self.children is None:
            return None
        for child in self.children:
            if child.move == move:
                return child

        raise Exception('Not existing child')
    
    def copy(self):
        """Return deep copied env"""
        return deepcopy(self)