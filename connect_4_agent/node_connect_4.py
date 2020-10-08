from MCTS.node import Node
from collections import defaultdict

class NodeConnect4(Node):

    def __init__(self, gym_env, parent=None):
        super().__init__(gym_env, parent)
        self._number_of_visits = 0.
        self._results = defaultdict(int)
        self._untried_actions = None

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.env.get_valid_actions()
        return self._untried_actions

    @property
    def q(self):
        #TODO: Make this parametric
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self.untried_actions.pop()
        state, reward, is_done = self.env.step(action)
        # deepcopy env
        child_env = self.env.copy()
        child_node = NodeConnect4(
            child_env, parent=self
        )
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        is_done = self.env.is_game_over()
        return is_done

    def rollout(self):
        current_rollout_env = self.env.copy()

        second_player = current_rollout_env.PLAYER

        is_done = current_rollout_env.is_game_over()

        while not is_done:
            possible_moves = current_rollout_env.get_valid_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_env.step(action)
            is_done = current_rollout_env.is_game_over()
            
            if current_rollout_env.get_valid_actions().__len__() == 0:
                if is_done is False:
                    debug = True
                    current_rollout_env.is_game_over()

        # Check if the move is the winning move
        game_result, winning_player = current_rollout_env.check_win()

        # In my MCTS implementation
        # Always PLAYER.first starts to play
        if winning_player is second_player:
            return game_result.value * -1

        return game_result.value

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)