from typing import Any, Dict

import numpy as np

from tqdm import tqdm
import time
import random
from copy import copy, deepcopy

# from check_submission import check_submission
from game_mechanics import GoEnv, choose_move_randomly, load_pkl, play_go, transition_function, BOARD_SIZE, DeltaEnv

import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary

from statistics import mean

TEAM_NAME = "OPEC"

# PARAMETERS
NUM_MCTS_FOR_FINAL_TESTING = 50
NUM_GAMES_FOR_FINAL_TESTING = 100

# Actually 5 but we shave off 0.5
SECONDS_PER_MOVE_IN_COMPETITION = 4.5
# TODO Change this
# SECONDS_PER_MOVE_IN_COMPETITION = 0.5

device = "cpu"

###############################
######## START net.py #########
###############################

def convert_legal_moves_to_mask(legal_moves, board_size):
    mask = torch.zeros(board_size * board_size + 1, dtype=torch.int, device=device)
    mask.index_fill_(0, torch.tensor(legal_moves, device=device), 1)
    return mask

# As described here: https://calm-silver-e6f.notion.site/6-Proximal-Policy-Optimization-PPO-3b5c45aa6ff34523a31ba08f3b324b23#4ccd589883eb4e05828b39dbc9fef135
def softmax_with_legal_move_masking(move_distribution, legal_moves, board_size):
    mask = convert_legal_moves_to_mask(legal_moves, board_size)
    move_distribution = F.softmax(move_distribution * mask, dim=-1)
    move_distribution = move_distribution * mask
    move_distribution = move_distribution / (move_distribution.sum() + 1e-13)
    return move_distribution

#############################
######## END net.py #########
#############################


################################
######## START mcts.py #########
################################

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


def mcts(n_or_seconds: str, observation: np.ndarray, legal_moves: np.ndarray, env: DeltaEnv, network: nn.Module, n: int = None, seconds: float = None):
    assert n_or_seconds in ["n", "seconds"]
    assert n or seconds
    assert not (n and seconds)

    if seconds:
        n = 1_000_000_000

    start = time.time()
    mcts_runs_count = 0

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
    for _ in range(n):

        if seconds:
            now = time.time()
            elapsed_time = now - start
            if elapsed_time > seconds:
                break
        
        mcts_runs_count += 1

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
    moves_for_pi = [] # { move, rating, votes }
    for move in root_node["legal_moves"]: # TODO Maybe guard against empty
        # children_hashes_in_parent_plus_move_to_node_tree 
        child_hash_in_parent_plus_move_to_node_tree = hash_parent_plus_move(root_hash, move)
        child_was_explored_during_mcts = child_hash_in_parent_plus_move_to_node_tree in parent_plus_move_to_node_tree
        if child_was_explored_during_mcts:
            child = parent_plus_move_to_node_tree[child_hash_in_parent_plus_move_to_node_tree]

            # This way of getting the move's value is shitty because a node that was picked once and randomly estimated highly will show up at the top.
            # move_value = child["total_value"] / child["updated_count"]
            # Instead do it this way (https://stackoverflow.com/questions/1411199/what-is-a-better-way-to-sort-by-a-5-star-rating):

            moves_for_pi.append({
                "move": move,
                "rating": (child["total_value"] / child["updated_count"]).item(),
                "votes": child["updated_count"]
            })
            # rating = child["total_value"] / child["updated_count"] # average for the movie (mean)
            # votes = child["updated_count"] # number of votes for the movie
            # min_votes_needed = 1 # minimum votes required to be listed in the Top 250 (currently 25000)
            # average_rating = None # the mean vote across the whole report (currently 7.0)
            # move_value = (rating * votes + average_rating * min_votes_needed) / (votes + min_votes_needed)

            # pi[move] = move_value
    
    average_rating = mean(move_for_pi_dict["rating"] * move_for_pi_dict["votes"] for move_for_pi_dict in moves_for_pi)
    MIN_VOTES_NEEDED = 3 # minimum votes required to be listed in the Top 250 (currently 25000)
    for move_for_pi_dict in moves_for_pi:
        move = move_for_pi_dict["move"]
        rating = move_for_pi_dict["rating"]
        votes = move_for_pi_dict["votes"]
        move_value = (rating * votes + average_rating * MIN_VOTES_NEEDED) / (votes + MIN_VOTES_NEEDED)
        pi[move] = move_value
    
    # TODO This mask shouldn't be necessary...there's probably a bug in my code. The MCTS should take care of not proposing any illegal moves......
    pi = softmax_with_legal_move_masking(pi, legal_moves, board_size)

    # print("mcts_runs_count", mcts_runs_count)

    # Return the distribution based on the tree (so I think get the child nodes of the current node based on their value... softmax maybe? Also need to do legal action masking? Or can that be taken care of upstream? I think upstream is better.)
    return pi

################################
######## END mcts.py #########
################################


################################
######## START main.py #########
################################

# Version to be used in competition, which does a lot of mcts per move.
def choose_move(observation: np.ndarray, legal_moves: np.ndarray, env, neural_network: nn.Module) -> int:
    pi = mcts(
        n_or_seconds="seconds",
        observation=observation,
        legal_moves=legal_moves,
        env=env,
        network=neural_network,
        seconds=SECONDS_PER_MOVE_IN_COMPETITION,
    )
    return pi.argmax().item()
    # return choose_move_randomly(observation, legal_moves, env)


def choose_move_for_testing(observation: np.ndarray, legal_moves: np.ndarray, env, neural_network: nn.Module) -> int:
    pi = mcts(
        n_or_seconds="n",
        observation=observation,
        legal_moves=legal_moves,
        env=env,
        network=neural_network,
        n=NUM_MCTS_FOR_FINAL_TESTING
    )
    return pi.argmax().item()


def convert_to_no_network(choose_move_fn, network):
    if choose_move_fn == choose_move_randomly:
        return choose_move_randomly
    return lambda observation, legal_moves, env: choose_move_fn(observation, legal_moves, env, network)


def play_n_games(n, your_choose_move, your_network, opponent_choose_move, opponent_network, verbose=False):
    wins = 0
    iter = tqdm(range(n)) if verbose else range(n)
    if verbose:
        print()
        print(f"PLAYING {n} GAMES FOR TESTING. Opponent is {'RANDOM' if opponent_choose_move == choose_move_randomly else 'A NETWORK'}")
    for i in iter:
        win = 1 if play_go(
            your_choose_move=convert_to_no_network(your_choose_move, your_network),
            opponent_choose_move=convert_to_no_network(opponent_choose_move, opponent_network),
            game_speed_multiplier=1,
            render=False,
            verbose=False,
        ) == 1 else 0
        # print(win)
        wins += win
        win_percentage = wins / (i + 1)
        if verbose:
            print(f"So far winning {win_percentage:.2f} of games")
    if verbose:
        print(f"Won {win_percentage:.2f} of {n} games")
    return win_percentage


# if __name__ == "__main__":
def main():

    my_network = load_pkl(TEAM_NAME)
    my_network.eval()

    def choose_move_no_network(observation: np.ndarray, legal_moves: np.ndarray, env) -> int:
        return choose_move(observation, legal_moves, env, my_network)

    # Test against a random opponent with limited MCTS
    play_n_games(
        n=NUM_GAMES_FOR_FINAL_TESTING,
        your_choose_move=choose_move_for_testing,
        your_network=my_network,
        opponent_choose_move=choose_move_randomly,
        opponent_network=None,
        verbose=True,
    )

    # Testing with competition time for MCTS
    # play_n_games(
    #     n=10,
    #     your_choose_move=choose_move,
    #     your_network=my_network,
    #     opponent_choose_move=choose_move_randomly,
    #     opponent_network=None,
    #     verbose=True,
    # )

    # TypeError: render() got an unexpected keyword argument 'screen_override'
    # play_go(
    #     your_choose_move=choose_move_no_network,
    #     opponent_choose_move=choose_move_no_network,
    #     game_speed_multiplier=10,
    #     render=True,
    #     verbose=False,
    # )

main()