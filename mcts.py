from audioop import add
from typing import Any, Dict

import numpy as np

from tqdm import tqdm
import time
import random
from copy import copy, deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary

from game_mechanics import GoEnv, choose_move_randomly, play_go, transition_function, DeltaEnv
from net import softmax_with_legal_move_masking
from config import device


def hash_board(board: np.ndarray, is_my_move: bool):
    board_hash = board.tobytes()
    turn_hash = np.array(is_my_move).tobytes()
    return board_hash + turn_hash


def hash_parent_plus_move(parent_hash: bytes, move_taken_from_parent: int):
    return parent_hash + np.array(move_taken_from_parent).tobytes()

def add_to_tree(tree: Dict, parent_plus_move_to_node_tree: Dict, board: np.ndarray, legal_moves:np.ndarray, parent_hash: bytes, move_that_got_you_here: int, node_env: DeltaEnv):
    has_parent = bool(parent_hash)
    parent = tree[parent_hash] if has_parent else None

    is_my_move = not parent["is_my_move"] if has_parent else True
    board_hash = hash_board(board, is_my_move)

    # TODO This assert shouldn't throw...but it does for some reason. My guess is that it's because we're repeating states. So let's just ignore it for now and see what happens.
    # assert board_hash not in tree
    if board_hash in tree:
        return board_hash

    new_node = {
        "hash": board_hash,
        "board": board,
        "legal_moves": legal_moves,
        "total_value": parent["total_value"] / parent["updated_count"] if has_parent else 0,
        "selected_count": 0,
        "updated_count": 1, # I'm making this one to avoid divide by zero?
        # "n": 0, # TODO Should this be 1?
        "parent_hash": parent_hash,
        # "children_hashes": [],
        "is_terminal": node_env.done, # Think we don't need to pass this in
        "is_my_move": is_my_move,
        "move_that_got_you_here": move_that_got_you_here,
        "node_env": node_env,
    }
    tree[board_hash] = new_node
    if has_parent:
        assert move_that_got_you_here != None
        parent_plus_move_hash = hash_parent_plus_move(parent_hash, move_that_got_you_here)
        parent_plus_move_to_node_tree[parent_plus_move_hash] = new_node
    return board_hash


def get_children_hashes(tree: Dict, parent_plus_move_to_node_tree: Dict, node_hash: bytes):
    node = tree[node_hash]
    children_hashes_in_parent_plus_move_to_node_tree = [hash_parent_plus_move(node_hash, move) for move in node["legal_moves"]]

    children = []
    for hash in children_hashes_in_parent_plus_move_to_node_tree:
        if hash in parent_plus_move_to_node_tree:
            children.append(parent_plus_move_to_node_tree[hash])
        
    children_hashes = [child["hash"] for child in children]
    return children_hashes


def select_node_randomly(tree: Dict, parent_plus_move_to_node_tree: Dict, root_hash: bytes):
    PROBABILITY_OF_PICKING_SELF = 0.5
    node_hash = root_hash
    while True:
        node = tree[node_hash]

        is_terminal = node["is_terminal"]
        if is_terminal:
            # return None # TODO This seems bad.
            break # TODO This also seems bad...

        children_hashes = get_children_hashes(tree, parent_plus_move_to_node_tree, node_hash)

        is_leaf = not children_hashes
        # is_leaf = not node["children_hashes"]
        pick_self = random.random() < PROBABILITY_OF_PICKING_SELF
        if is_leaf or pick_self:
            break

        # If the selected node is fully expanded, we don't want to expand it. TODO Do we have to do anything about this?

        # pick a child randomly (since it does have children)
        node_hash = random.choice(children_hashes)

    node["selected_count"] += 1
    return node_hash


def mcts(n: int, observation: np.ndarray, legal_moves: np.ndarray, env: DeltaEnv, network: nn.Module):

    tree = {}
    # Whereas the normal tree maps a hash(board + my turn) to a node, this tree maps hash(a parent state + the move taken) to the node. This is useful when you're trying to figure out whether all of a node's children are already in the tree.
    parent_plus_move_to_node_tree = {}
    
    # Add the root to the tree
    root_hash = add_to_tree(
        tree=tree,
        parent_plus_move_to_node_tree=parent_plus_move_to_node_tree,
        board=observation,
        legal_moves=legal_moves,
        parent_hash=None,
        move_that_got_you_here=None,
        node_env=env,
    )
    
    # print()
    # print(f"RUNNING {n} MCTS:")
    # for i in tqdm(range(n)):
    for i in range(n):

        # SELECT
        # TODO Use actual PUCT instead of random selection.
        node_hash = select_node_randomly(tree, parent_plus_move_to_node_tree, root_hash)
        if not node_hash:
            print("Did not pick a node for mcts")
            continue
        node = tree[node_hash]

        # EXPAND
        # If the selected node is fully expanded, don't expand it

        # TODO This line may error if terminal nodes don't have a legal_moves array...idk if they do. If it does error, just check for it being a terminal node ahead of this.
        moves_not_in_tree = [move for move in node["legal_moves"] if hash_parent_plus_move(node_hash, move) not in parent_plus_move_to_node_tree]

        is_fully_expanded = not moves_not_in_tree
        is_terminal = node["is_terminal"]
        if is_fully_expanded or is_terminal:
            node_to_evaluate_hash = node_hash
        else:
            # Add new node to the tree. It's the one you want to evaluate.
            move = random.choice(moves_not_in_tree)
            new_env = transition_function(node["node_env"], move)
            node_to_evaluate_hash = add_to_tree(
                tree=tree,
                parent_plus_move_to_node_tree=parent_plus_move_to_node_tree,
                board=new_env.observation,
                legal_moves=new_env.legal_moves,
                parent_hash=node_hash,
                move_that_got_you_here=move,
                node_env=new_env,
            )

        assert node_to_evaluate_hash
        node_to_evaluate = tree[node_to_evaluate_hash]

        # TODO Not sure if we should break or something if it is a terminal node.

        # EVALUATE
        multiplier = 1 if node_to_evaluate["is_my_move"] else -1
        # NOTE I think you want no grad here but I'm not sure.
        with torch.no_grad():
            p, v = network.forward_with_cache(node_to_evaluate["board"], node_to_evaluate["legal_moves"])
            v = multiplier * v

        # BACKUP
        node_to_update_hash = node_to_evaluate_hash
        while True:
            node_to_update = tree[node_to_update_hash]
            node_to_update["total_value"] += v
            node_to_update["updated_count"] += 1
            if not node_to_update["parent_hash"]:
                break
            node_to_update_hash = node_to_update["parent_hash"]

    # TODO Here is a good time to inspect the tree to see if it's doing anything.

    # Get a length-82 array of action values and softmax them to get a distribution
    board_size = observation.shape[0] # TODO Change if making 3D
    pi = torch.zeros(board_size * board_size + 1, device=device)
    root_node = tree[root_hash]
    for move in root_node["legal_moves"]: # TODO Maybe guard against empty
        # children_hashes_in_parent_plus_move_to_node_tree 
        child_hash_in_parent_plus_move_to_node_tree = hash_parent_plus_move(root_hash, move)
        child_was_explored_during_mcts = child_hash_in_parent_plus_move_to_node_tree in parent_plus_move_to_node_tree
        if child_was_explored_during_mcts:
            child = parent_plus_move_to_node_tree[child_hash_in_parent_plus_move_to_node_tree]
            move_value = child["total_value"] / child["updated_count"]
            pi[move] = move_value
    
    # TODO This mask shouldn't be necessary...there's probably a bug in my code. The MCTS should take care of not proposing any illegal moves......
    pi = softmax_with_legal_move_masking(pi, legal_moves, board_size)

    # Return the distribution based on the tree (so I think get the child nodes of the current node based on their value... softmax maybe? Also need to do legal action masking? Or can that be taken care of upstream? I think upstream is better.)
    return pi