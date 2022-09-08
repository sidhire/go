import sys
import os
import torch
import random

go_path = "/Users/siddharth/code/delta-rl/go"
is_sid = any("siddharth" in s for s in sys.path)
if is_sid and go_path not in sys.path:
    sys.path.append(go_path)

from mcts.state import State


def test_state_3():
    assert State

# def test_init():
#     # Enter code here
#     x = MCTS(
#         initial_state=State([0, 0, 0, 0, 0, 0, 0, 0, 0], 1),
#         rollout_policy=lambda x: get_possible_actions(x)[
#             int(random.random() * len(get_possible_actions(x)))
#         ],
#         explore_coeff=0.5,
#         verbose=True,
#     )
#     assert x


# def test_1_simulate_from_terminal_state(self):
#     # Enter code here
#     mcts = MCTS(
#         initial_state=State([0, 0, 0, 0, 0, 0, 0, 0, 0], 1),
#         rollout_policy=lambda x: get_possible_actions(x)[
#             int(random.random() * len(get_possible_actions(x)))
#         ],
#         explore_coeff=0.5,
#         verbose=True,
#     )
#     node = Node(State([1, 1, 1, 0, -1, -1, 0, 0, 0], -1), True)
#     total_return = mcts._simulate(node)
#     assert total_return in {-1, 0, 1}, f"total_return returned from ._simulate() = {total_return}, must be in [-1, 0, 1]"


# def test_1_simulate_from_base(self):
#     # Enter code here
#     mcts = MCTS(
#     initial_state=State([0, 0, 0, 0, 0, 0, 0, 0, 0], 1),
#         rollout_policy=lambda x: get_possible_actions(x)[
#             int(random.random() * len(get_possible_actions(x)))
#         ],
#         explore_coeff=0.5,
#         verbose=True,
#     )
#     node = Node(State([0, 0, 0, 0, 0, 0, 0, 0, 0], 1), False)
#     total_return = mcts._simulate(node)
#     assert total_return in {-1, 0, 1}, f"total_return returned from ._simulate() = {total_return}, must be in [-1, 0, 1]"


# def test_2_backup_win_base(self):
#     # Enter code here
#     mcts = MCTS(
#         initial_state=State([0, 0, 0, 0, 0, 0, 0, 0, 0], 1),
#         rollout_policy=lambda x: get_possible_actions(x)[
#             int(random.random() * len(get_possible_actions(x)))
#         ],
#         explore_coeff=0.5,
#         verbose=True,
#     )
#     state = State([0, 0, 0, 0, 0, 0, 0, 0, 0], 1)
#     node = Node(state, False)
#     total_return = 1
#     mcts.tree[state.key] = node
#     mcts.N[state.key] = 3
#     mcts.total_return[state.key] = 0

#     mcts._backup(node, total_return)

#     assert mcts.total_return[
#                state.key] == 1, f"total_return dictionary not updated correctly! total_return[state.id] = {mcts.total_return[state.key]}, when it should be 1!"
#     assert mcts.N[
#                state.key] == 4, f"N dictionary not updated correctly! N[state.id] = {mcts.N[state.key]}, when it should be 4!"


# def test_2_backup_lose_state_and_parent(self):
#     # Enter code here
#     mcts = MCTS(
#         initial_state=State([0, 0, 0, 0, 0, 0, 0, 0, 0], 1),
#         rollout_policy=lambda x: get_possible_actions(x)[
#             int(random.random() * len(get_possible_actions(x)))
#         ],
#         explore_coeff=0.5,
#         verbose=True,
#     )
#     state = State([-1, 1, 0, 0, 0, 0, 0, 0, 0], 1)
#     parent_state_1 = State([0, 1, 0, 0, 0, 0, 0, 0, 0], -1)
#     parent_state_2 = State([0, 0, 0, 0, 0, 0, 0, 0, 0], 1)

#     node = Node(state, False)
#     parent_node_1 = Node(parent_state_1, False)
#     parent_node_2 = Node(parent_state_2, False)

#     mcts.tree[state.key] = node
#     mcts.tree[parent_state_1.key] = parent_node_1
#     mcts.tree[parent_state_2.key] = parent_node_2

#     mcts.N[state.key] = 3
#     mcts.N[parent_state_1.key] = 4
#     mcts.N[parent_state_2.key] = 5

#     mcts.total_return[state.key] = 0
#     mcts.total_return[parent_state_1.key] = 1
#     mcts.total_return[parent_state_2.key] = 1

#     total_return = 1

#     mcts._backup(node, total_return)

#     assert mcts.total_return[
#                state.key] == 1, f"total_return dictionary not updated correctly! total_return[state.key] = {mcts.total_return[state.key]}, when it should be 1!"
#     assert mcts.total_return[
#                parent_state_1.key] == 2, f"total_return dictionary not updated correctly for parent nodes! total_return[state.key] = {mcts.total_return[parent_state_1.key]}, when it should be 1!"
#     assert mcts.total_return[
#                parent_state_2.key] == 2, f"total_return dictionary not updated correctly for parents! total_return[state.key] = {mcts.total_return[parent_state_2.key]}, when it should be 1!"

#     assert mcts.N[
#                state.key] == 4, f"N dictionary not updated correctly! N[state.key] = {mcts.N[state.key]}, when it should be 4!"
#     assert mcts.N[
#                parent_state_1.key] == 5, f"N dictionary not updated correctly for parents! N[state.key] = {mcts.N[parent_state_1.key]}, when it should be 4!"
#     assert mcts.N[
#                parent_state_2.key] == 6, f"N dictionary not updated correctly for parents! N[state.key] = {mcts.N[parent_state_2.key]}, when it should be 4!"


# def test_3_select_empty(self):
#     # Enter code here
#     root = State([0, 0, 0, 0, 0, 0, 0, 0, 0], 1)
#     mcts = MCTS(
#         initial_state=root,
#         rollout_policy=lambda x: get_possible_actions(x)[
#             int(random.random() * len(get_possible_actions(x)))
#         ],
#         explore_coeff=0.5,
#         verbose=True,
#     )

# #   node = mcts._select()
# #   assert isinstance(node, Node), f"Node returned from ._select() is not of type Node, instead it's of type: {type(node)}"
# #   assert not node.is_terminal, "Your ._select() function selected a terminal node from the root. It should select a node where 1 move has been played."
# #   assert len(node.parent_states) == 0, f"Your ._select() function returns a node with too many possible parent states: {node.parent_states}. It should have 0 parents. Node returned has state: {node.state}"
# #   assert node.state == root, f"Your ._select() function returns a node which isn't the root node given at initialization, with state: {root}. Instead, it returned the node with this state: {node.state}"


# def test_3_select_exploit(self):
#     # Enter code here
#     root = State([0, 0, 0, 0, 0, 0, 0, 0, 0], 1)
#     mcts = MCTS(
#         initial_state=root,
#         rollout_policy=lambda x: get_possible_actions(x)[
#             int(random.random() * len(get_possible_actions(x)))
#         ],
#         explore_coeff=0,
#         verbose=True,
#     )
#     big_boy = 8
#     for i in range(9):
#         board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
#         board[i] = 1
#         state = State(board, -1)
#         mcts.tree[state.key] = Node(state, False)
#         mcts.N[state.key] = 1 if i != big_boy else 10
#         mcts.total_return[state.key] = -2 if i != big_boy else 10

#     node = mcts._select()

#     assert isinstance(node,
#                       Node), f"Node returned from ._select() is not of type Node, instead it's of type: {type(node)}"
#     assert not node.is_terminal, f"Your ._select() function selected a terminal node with state: {node.state} from the root with no pieces played. It should select a node where 1 move has been played."
#     assert node.state == state, f"Your ._select() function selected a terminal node with state: {node.state}, when it should have selected state: {state}"
#     assert len(
#         node.parent_states) == 1, f"Your ._select() function returns a node with an incorrect number of possible parent states: {node.parent_states}. It should have 1 parent - an empty board with 1 going first. Node returned has state: {node.state}"
#     assert node.parent_states[
#                0] == root, f"Your ._select() function returns a node whose parent isn't the root node given at initialization, with state: {root}. Instead, it returned a node with parent: {node.parent_states[0]}"


# def test_4_expand_root(self):
#     # Enter code here
#     root = State([0, 0, 0, 0, 0, 0, 0, 0, 0], 1)
#     mcts = MCTS(
#         initial_state=root,
#         rollout_policy=lambda x: get_possible_actions(x)[
#             int(random.random() * len(get_possible_actions(x)))
#         ],
#         explore_coeff=0,
#         verbose=True,
#     )

#     node = mcts._expand(mcts.tree[mcts.root_node.key])

#     assert isinstance(node,
#                       Node), f"Node returned from ._expand() is not of type Node, instead it's of type: {type(node)}"
#     assert not node.is_terminal, f"Your ._expand() function selected a terminal node with state: {node.state} from the root with no pieces played. It should select a node where 1 move has been played."
#     assert len(
#         node.parent_states) == 1, f"Your ._expand() function returns a node with an incorrect number of possible parent states: {node.parent_states}. It should have 1 parent - an empty board with 1 going first. Node returned has state: {node.state}"
#     assert node.parent_states[
#                0] == root, f"Your ._expand() function returns a node whose parent isn't the root node given at initialization, with state: {root}. Instead, it returned a node with parent: {node.parent_states[0]}"


# def test_4_expand_terminal(self):
#     # Enter code here
#     root = State([0, 0, 0, 0, 0, 0, 0, 0, 0], 1)
#     mcts = MCTS(
#         initial_state=root,
#         rollout_policy=lambda x: get_possible_actions(x)[
#             int(random.random() * len(get_possible_actions(x)))
#         ],
#         explore_coeff=0,
#         verbose=True,
#     )
#     terminal_state = State([1, 1, 1, -1, -1, 0, 0, 0, 0], -1)
#     terminal_node = Node(terminal_state, True)

#     node = mcts._expand(terminal_node)

#     assert isinstance(node,
#                       Node), f"Node returned from ._expand() is not of type Node, instead it's of type: {type(node)}"
#     assert node == terminal_node, f"Node returned from ._expand() should be the terminal node, with state: {terminal_node.state}. Instead returned node with state: {node.state}"
#     assert node.is_terminal, f"Your ._expand() function selected a terminal node with state: {node.state} from the root with no pieces played. It should select a node where 1 move has been played."


# def test_rollout(self):
#   # Enter code here
#   root = State([0, 0, 0, 0, 0, 0, 0, 0, 0], 1)
#   mcts = MCTS(
#       initial_state=root,
#       rollout_policy=lambda x: get_possible_actions(x)[
#           int(random.random() * len(get_possible_actions(x)))
#       ],
#       explore_coeff=0,
#       verbose=True,
#   )
#   mcts.do_rollout()
