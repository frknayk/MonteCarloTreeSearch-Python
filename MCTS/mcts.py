#!/usr/bin/env python3

class MonteCarloTreeSearch(object):

    def __init__(self, node):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : MCTS.Node
        """
        self.root = node

    def best_action(self, simulations_number):
        """

        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action

        Returns
        -------

        """
        for _ in range(0, simulations_number):
            # Select node  
            v = self._tree_policy()
            # Run simulation from selected node
            reward = v.rollout()
            # Backpropagation
            v.backpropagate(reward)
        # to select best child go for exploitation only
        return self.root.best_child(c_param=0.)

    def _tree_policy(self):
        """
        selects node to run rollout/playout for

        Returns
        -------

        """
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
