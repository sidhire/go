from typing import Callable, Dict
import math
import random

from mcts.state import State
from mcts.node import Node
# from mcts_env import (
from game_mechanics import (
    # State,
    # StateID,
    transition_function,
    reward_function,
    is_terminal,
    # validate_mcts,
)


class MCTS:
    def __init__(
        self,
        initial_state: State,
        rollout_policy: Callable[[State], int],
        explore_coeff: float,
        verbose: int = 0,
    ):
        self.root_node = Node(initial_state, False)
        # self.total_return: Dict[StateID:float] = {self.root_node.key: 0.0}
        self.total_return: Dict = {self.root_node.key: 0.0}
        # self.N: Dict[StateID:int] = {self.root_node.key: 0}
        self.N: Dict = {self.root_node.key: 0}
        # self.tree: Dict[StateID:State] = {self.root_node.key: self.root_node}
        self.tree: Dict = {self.root_node.key: self.root_node}

        self.rollout_policy = rollout_policy
        self.explore_coeff = explore_coeff

        self.verbose = verbose
        self.select_count = 0

    def do_rollout(self) -> None:
        if self.verbose:
            print("\nNew rollout started from", self.root_node.state)

        selected_node = self._select()
        simulation_node = self._expand(selected_node)
        total_return = self._simulate(simulation_node)
        self._backup(simulation_node, total_return)

    def _select(self) -> Node:
        """
        Selects a node to simulate from, given the current state
         and tree.

        Write this as exercise 4
        """
        self.select_count += 1
        node = self.root_node

        # In case it's the first rollout - add all children of root node
        self._expand(node) # TODO I don't think this is needed, but the test will fail (see my msg to Henry and change the test)
        # return self.tree[node.child_states[0].key]

        # If not fully expanded children, select this node
        while not node.is_terminal and all(
                state.state_id in self.tree for state in node.child_states.values()
        ):
            child_nodes = {a: self.tree[state.state_id] for a, state in node.child_states.items()}
            # TODO CHANGE IT BACK TO USE UTC
            node = self._uct_select(self.N[node.key], child_nodes)
            # node = random.choice(list(child_nodes.values()))
            if self.verbose:
                print("UCT selected:", node.state)

        return node

    def _expand(self, node: Node) -> Node:
        """
        Unless the selected node is a terminal state, expand the
         selected node by adding its children nodes to the tree.

        Write this as exercise 5
        """
        if node.is_terminal:
            return node

        child_nodes = []
        
        # IMPORTANT: I think thous sohuld say node instead of root_node:
        # for action, state in self.root_node.child_states.items():
        for action, state in node.child_states.items():
            # Just add states as nodes to the tree not already in there
            if state.state_id not in self.tree:
                child_node = Node(state, is_terminal(state))
                self.add_child_node(child_node)
                child_nodes.append(child_node)

        return random.choice(child_nodes) if child_nodes else node

    def _simulate(self, node: Node) -> float:
        """
        Simulates a full episode to completion from `node`,
         outputting the total return from the episode.

        Write this as exercise 2
        """
        prev_state = node.state
        state = None
        if not node.is_terminal:
            action = self.rollout_policy(node.state)
            if state:
                prev_state = state
            # TODO not sure if the last arg for bike_move should be this state's player or the prev state's player
            state = transition_function(node.state, action, node.state.player)

            while not is_terminal(state):
                action = self.rollout_policy(state)
                if self.verbose:
                    print("Simulation take move:", action)
                if state:
                    prev_state = state
                state = transition_function(state, action, state.player)
        else:
            state = node.state

        # TODO not sure if the bike for bike_move should be the player in this state or in the previous state. I think in the previous state. — TODO IMPORTANT: I think it should be the first one but I'm not sure.
        # total_return = reward_function(state, prev_state.player)
        # total_return = reward_function(state, state.player)
        total_return = 0 if state.player.alive else -1
        # if total_return < 0:
        #     print("AAAAAAAAAAAAAAAAAAAAAAAAA")

        if self.verbose:
            print("Simulation return:", total_return, state)

        return total_return

    def _backup(self, node: Node, total_ep_return: float) -> None:
        """
        Update the Monte Carlo action-value estimates of all
         parent nodes in the tree with the return from the
         simulated trajectory.

        Write this as exercise 3
        """
        prev_added_parents = [node]

        while prev_added_parents:
            newly_added_parents = []
            for node in prev_added_parents:
                self.total_return[node.key] += total_ep_return
                self.N[node.key] += 1
                if self.verbose >= 2:
                    print(
                        "Backing up node:", node.state, self.N[node.key], self.total_return[node.key]
                    )
                newly_added_parents += [
                    self.tree[state.state_id] for state in node.parent_states if state.state_id in self.tree
                ]
            prev_added_parents = newly_added_parents

    def choose_action(self) -> int:
        """
        Once we've simulated all the trajectories, we want to
         select the action at the current timestep which
         maximises the action-value estimate.
        """
        if self.verbose:
            print(
                "Q estimates & N:",
                {
                    a: (round(self.Q(state.state_id), 2), self.N[state.state_id])
                    for a, state in self.root_node.child_states.items()
                },
            )
        # if self.N[self.root_node.key] != 0:
        #     print("ROOT N IS NOT ZERO!")
        return max(
            self.root_node.child_states.keys(),
            key=lambda a: self.Q(self.root_node.child_states[a].state_id),
        )

    def Q(self, node_id: str) -> float:
        return self.total_return[node_id] / (self.N[node_id] + 1e-15)

    def _uct_select(self, N: int, children_nodes: Dict[int, Node]) -> Node:
        max_uct_value = -math.inf
        max_uct_nodes = []
        for child_node in children_nodes.values():
            # q = -child_node.state.player_to_move * self.Q(child_node.key)
            q = self.Q(child_node.key)
            uct_value = q + self.explore_coeff * math.sqrt(
                math.log(N + 1) / (self.N[child_node.key] + 1e-15)
            )
            # if self.verbose >= 2:
            #     print(
            #         child_node.state,
            #         "UCT value",
            #         round(uct_value, 2),
            #         "Q",
            #         round(q, 2),
            #         "Sign:",
            #         child_node.state.player_to_move,
            #     )

            if uct_value > max_uct_value:
                max_uct_value = uct_value
                max_uct_nodes = [child_node]
            elif uct_value == max_uct_value:
                max_uct_nodes.append(child_node)

        len_nodes = len(max_uct_nodes)
        chosen_node = max_uct_nodes[int(random.random() * len_nodes)]
        if self.verbose and self.N[chosen_node.key] == 0:
            print("Exploring!")
        return chosen_node

    def add_child_node(self, child_node: Node):
        self.tree[child_node.key] = child_node
        self.total_return[child_node.key] = 0
        self.N[child_node.key] = 0

    def prune_tree(self, action_taken: int, successor_state: State) -> None:
        # If it's the terminal state we don't care about pruning the tree
        if is_terminal(successor_state):
            return

        self.root_node = self.tree.get(
            successor_state.state_id, Node(successor_state, is_terminal(successor_state))
        )
        self.N[self.root_node.key] = 0
        self.total_return[self.root_node.key] = 0

        # Build a new tree dictionary
        new_tree = {self.root_node.key: self.root_node}

        prev_added_nodes = {self.root_node.key: self.root_node}
        while prev_added_nodes:
            newly_added_nodes = {}

            for node in prev_added_nodes.values():
                child_nodes = {
                    state.state_id: self.tree[state.state_id]
                    for state in node.child_states.values()
                    if state.state_id in self.tree
                }
                new_tree.update(child_nodes)
                newly_added_nodes.update(child_nodes)

            prev_added_nodes = newly_added_nodes

        self.tree = new_tree
        self.total_return = {key: self.total_return[key] for key in self.tree}
        self.N = {key: self.N[key] for key in self.tree}


# if __name__ == "__main__":
#     validate_mcts(MCTS)

