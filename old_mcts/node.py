from typing import List, Dict
from copy import copy, deepcopy
import random

# from mcts_env import State, get_possible_actions
from mcts.state import State
from game_mechanics import Orientation, transition_function, ARENA_HEIGHT, ARENA_WIDTH


class Node:
    def __init__(self, state: State, is_terminal: bool):
        self.state = state
        self.is_terminal = is_terminal
        # No guarantee that these NODES exist in the MCTS TREE!
        self.parent_states = self._get_possible_parent_states()
        # No guarantee that these NODES exist in the MCTS TREE!
        self.child_states = self._get_possible_children()
        # self.key = self.state.key
        self.key = self.state.state_id

    def _get_possible_parent_states(self) -> List[State]:
        # Create a new state
        curr_state = self.state
        # parents = []

        prev_state = copy(curr_state)
        # bike_move = copy(bike_move)
        prev_state.player = copy(curr_state.player)
        prev_state.opponents = [copy(bike) for bike in curr_state.opponents]

        # change: positions, direction (Orientation.EAST), alive must be True

        all_players = [prev_state.player] + prev_state.opponents

        for player in all_players:

            me = prev_state.player
            is_me = player == me

            curr_position = player.positions.pop(0)
            was_alive = True if is_me else len(player.positions) >= len(me.positions)
            player.alive = was_alive

            if not was_alive:
                continue

            # Still need to set prev direction below

            # South is 0 (y -= 1), North is 2 (y += 1)
            # East is 1 (x += 1), West is 3 (x -= 1)

            prev_position = player.positions[0]

            if len(player.positions) == 1:
                x1, y1 = curr_position
                x2, y2 = prev_position
                x_move = x1 - x2
                y_move = y1 - y2
                if x_move == 1:
                    # possible_prev_directions = [ Orientation.SOUTH, Orientation.EAST, Orientation.NORTH ]
                    prev_direction = Orientation.SOUTH
                elif x_move == -1:
                    # possible_prev_directions = [ Orientation.SOUTH, Orientation.NORTH, Orientation.WEST ]
                    prev_direction = Orientation.SOUTH
                elif y_move == 1:
                    # possible_prev_directions = [ Orientation.SOUTH, Orientation.EAST, Orientation.WEST ]
                    prev_direction = Orientation.SOUTH
                elif y_move == -1:
                    # possible_prev_directions = [ Orientation.EAST, Orientation.NORTH, Orientation.WEST ]
                    prev_direction = Orientation.EAST
            
            else:
                two_prev_position = player.positions[1]
                x1, y1 = prev_position
                x2, y2 = two_prev_position
                x_move = x1 - x2
                y_move = y1 - y2
                if x_move == 1:
                    prev_direction = Orientation.EAST
                elif x_move == -1:
                    prev_direction = Orientation.WEST
                elif y_move == 1:
                    prev_direction = Orientation.SOUTH
                elif y_move == -1:
                    prev_direction = Orientation.NORTH
                # possible_prev_directions = [prev_direction]
            
            # Set the prev direction
            player.direction = prev_direction
        

        return [prev_state]


    def _get_possible_children(self) -> Dict[int, State]:
        """Gets the possible children of this node."""
        curr_state = self.state
        all_players = [curr_state.player] + curr_state.opponents

        possible_moves = [1, 2, 3]

        # children = {}
        # for action in possible_moves:
        #     next_state = transition_function(curr_state, action, curr_state.player)
        #     children[action] = next_state
        # return children


        def my_choose_move_square(state, player) -> int:
            orientation = player.direction
            head = player.head

            if orientation == 0 and head[1] <= 3:
                return 3
            if orientation == 3 and head[0] <= 3:
                return 3
            if orientation == 2 and head[1] >= ARENA_HEIGHT - 3:
                return 3
            if orientation == 1 and head[0] >= ARENA_WIDTH - 3:
                return 3
            return 1
        
        children = {}
        for action in possible_moves:
            next_state = transition_function(curr_state, action, curr_state.player)
            for i in range(len(curr_state.opponents)):
                player = next_state.opponents[i]
                if not player.alive:
                    continue
                other_action = my_choose_move_square(next_state, player)
                next_state = transition_function(next_state, other_action, player)
            children[action] = next_state

        # children = {}
        # for player in all_players:
        #     action = my_choose_move_square(curr_state, player)
        #     next_state = transition_function(curr_state, action, player)
        #     children[action] = next_state

        return children

