import numpy as np
import random
from gym_connect.envs.enums.results_enum import RESULTS
from gym_connect.envs.enums.player import PLAYER
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

    def select_move(self):
        """
        Select best move and advance
        :return:
        """
        if self.children is None:
            return None

        winners = [child for child in self.children if child.game_status is RESULTS.WON]
        if len(winners) > 0:
            print("Winner node is found directly and returning : ",winners[0].move)
            return winners[0].move

        else:
            result, move = self.random_play_improved(self)
            print("Winner node is found via random_play_improved and returning : ",move, result)
            return move
            # if result is RESULTS.WON:
            #     print("Winner node is found via random_play_improved and returning : ",move)
            #     return move
            # games = [child.win/child.games if child.games > 0 else 0 for child in self.children]
            # best_child = self.children[np.argmax(games)]
            # print("Winner node is found via win/games ratio and returning : ",best_child.move)
            # return best_child.move

    def random_play_improved(self, node_env):
        node_env_origin = node_env.copy()
        # If can win, win
        while True:
            moves = node_env_origin.env.get_valid_actions()

            # if no moves left to play, game is finished
            if len(moves) == 0:
                return True

            winning_moves = self.get_winning_moves(node_env_origin, PLAYER.FIRST)
            loosing_moves = self.get_winning_moves(node_env_origin, PLAYER.SECOND)

            selected_move = None
            if len(winning_moves) > 0:
                selected_move = winning_moves[0]
            elif len(loosing_moves) == 1:
                selected_move = loosing_moves[0]
            else:
                selected_move = random.choice(moves)

            _, _, is_done = node_env_origin.env.step(selected_move)
            if is_done:
                if node_env_origin.env.PLAYER is PLAYER.FIRST:
                    return RESULTS.WON, selected_move
                return RESULTS.LOST, selected_move
    
    def get_winning_moves(self, node_env, player):
        winning_moves_list = []
        moves = node_env.env.get_valid_actions()
        for move in moves:
            env_copied = node_env.env.copy()
            _, _, is_done = env_copied.step(move)
            if is_done:
                if env_copied.PLAYER is player:
                    winning_moves_list.append(move)
        return winning_moves_list

    def get_children_with_move(self, move):
        if self.children is None:
            new_node = self.copy()
            _, _, is_done = new_node.env.step(move)
            game_status = RESULTS.NOT_FINISHED
            if is_done:
                if new_node.env.PLAYER is PLAYER.SECOND:
                    game_status = RESULTS.LOST
                else:
                    game_status = RESULTS.WON
            new_node = Node(
                    gym_env=new_node.env.copy(),
                    game_status=game_status,
                    move=move,
                    parent=self.copy())
            return new_node

        for child in self.children:
            if child.move == move:
                return child

        raise Exception('Not existing child')

    def copy(self):
        """Return deep copied env"""
        return deepcopy(self)

    def print_children(self):
        for child in self.children:
            ratio = -1 if child.games == 0 else child.win/child.games
            msg = "#WIN : {0} -- #GAMES : {1} -- WIN_RATIO : {2:.3f} -- MOVE : {3}".format(
                child.win,
                child.games,
                ratio,
                child.move
            )
            print(msg)
