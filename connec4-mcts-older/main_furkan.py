#!/usr/bin/env python3
from connect4_furkan.mcts import MonteCarloTreeSearch
from connect4_furkan.connect4 import create_grid
from connect4_furkan.node import Node


root = Node(create_grid(), 0, None,  None)
mcts = MonteCarloTreeSearch(root)
mcts.train(training_step=100)


def utils_print(grid):
    print_grid = grid.astype(str)
    print_grid[print_grid == '-1'] = 'X'
    print_grid[print_grid == '1'] = 'O'
    print_grid[print_grid == '0'] = ' '
    res = str(print_grid).replace("'", "")
    res = res.replace('[[', '[')
    res = res.replace(']]', ']')
    print(' ' + res)
    print('  ' + ' '.join('0123456'))

import numpy as np
from connect4.connect4 import play

if __name__ == "__main__":
    print('training finished')
    # test AI with real play
    grid = create_grid()
    round = 0
    training_time = 900
    node = mcts.root
    utils_print(grid)
    while True:
        if (round % 2) == 0:
            move = int(input())
            new_node = node.get_children_with_move(move)
            node = mcts.train_mcts_online(node, training_time).get_children_with_move(move)
        else:
            node_new, move = node.select_move()
            node = mcts.train_mcts_online(node, training_time)
            node, move = node.select_move()
            # node, move = mcts.get_best_move(node)

        grid, winner = play(grid, move)
        
        utils_print(grid)


        assert np.sum(node.state - grid) == 0, node.state
        if winner != 0:
            print('Winner : ', 'X' if winner == -1 else 'O')
            break
        round += 1



