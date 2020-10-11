from gym_connect.envs.enums.player import PLAYER
import numpy as np
import random

import gym
import gym_connect
from gym_connect.envs.enums.results_enum import RESULTS


def get_winning_moves(node_env, player):
    winning_moves_list = []
    moves = node_env.env.get_valid_actions()
    for move in moves:
        env_copied = node_env.env.copy()
        _, _, is_done = env_copied.step(move)
        if is_done:
            if env_copied.PLAYER is player:
                winning_moves_list.append(move)
    return winning_moves_list

def random_play_improved(node_env):
    node_env_origin = node_env.copy()
    # If can win, win
    while True:
        moves = node_env_origin.env.get_valid_actions()

        # if no moves left to play, game is finished
        if len(moves) == 0:
            return True

        winning_moves = get_winning_moves(node_env_origin, PLAYER.FIRST)
        loosing_moves = get_winning_moves(node_env_origin, PLAYER.SECOND)
        
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
                return RESULTS.WON
            return RESULTS.LOST