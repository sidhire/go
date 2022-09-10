from lib2to3.pytree import convert
from typing import Any, Dict

import numpy as np

from tqdm import tqdm
import time
import random

# from check_submission import check_submission
from game_mechanics import GoEnv, choose_move_randomly, load_pkl, play_go, save_pkl, BOARD_SIZE

import torch
from torch import nn
from torchinfo import summary

# from mcts.state import State
# from mcts.mcts import MCTS
from net import Net

# PARAMETERS
NUM_MOVES = BOARD_SIZE * BOARD_SIZE + 1

# NETWORK HYPERPARAMETERS


TEAM_NAME = "OPEC"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


def train():
    return nn.Sequential(
        nn.Linear(1, 1)
    )
    # raise NotImplementedError()



network = Net(board_size=BOARD_SIZE)

# Initialize lazy layers of the network by passing a dummy tensor of the correct shape through it.
dummy_observation = torch.rand(1, 1, BOARD_SIZE, BOARD_SIZE)
dummy_legal_moves = np.arange(NUM_MOVES)
_ = network(dummy_observation, dummy_legal_moves)
# summary(network, input_size=(1, 1, BOARD_SIZE, BOARD_SIZE))

def choose_move(observation: np.ndarray, legal_moves: np.ndarray, env, neural_network: nn.Module) -> int:

    board_width = observation.shape[0]
    observation = torch.tensor(observation, dtype=torch.float)
    observation = observation.reshape(1, 1, board_width, board_width)

    # TODO TODO TODO I left off at passing the correct shit to my network
    # Will need to do legal move masking as well.

    p, v = network(observation, legal_moves)
    print(p.shape, v.shape)

    # print("HI")
    return choose_move_randomly(observation, legal_moves, env)


    # CODE TO DO THIS WITH MCTS:
    # mcts = MCTS(
    #     initial_state=State(observation, legal_moves),
    #     rollout_policy=lambda x: random.choice(legal_moves),
    #     explore_coeff=0.5,
    #     verbose=False,
    # )

    # # Run episode loop
    # # for _ in range(100): # TODO do the timing
    # #     mcts.do_rollout()

    # time_budget = 0.4 # seconds
    # start = time.time()
    # elapsed_time = 0
    # rollout_count = 0
    # while elapsed_time < time_budget:
    #     mcts.do_rollout()
    #     now = time.time()
    #     elapsed_time = now - start
    #     rollout_count += 1
    # # print("rollout_count =", rollout_count)


    # action = mcts.choose_action()
    # return action


def play_n_games(n, your_choose_move=choose_move, opponent_choose_move=choose_move):
    network = None # TODO
    def convert_to_no_network(choose_move_fn):
        if choose_move_fn == choose_move_randomly:
            return choose_move_randomly
        return lambda o, l, env: choose_move_fn(o, l, env, network)
    wins = 0
    print(f"Playing {n} games:")
    for i in tqdm(range(n)):
        win = 1 if play_go(
            your_choose_move=convert_to_no_network(your_choose_move),
            opponent_choose_move=convert_to_no_network(opponent_choose_move),
            game_speed_multiplier=1,
            render=False,
            verbose=False,
        ) == 1 else 0
        # print(win)
        wins += win
    win_percentage = wins / n
    print(f"won {win_percentage:.2f} of {n} games")
    return win_percentage


if __name__ == "__main__":

    ## Example workflow, feel free to edit this! ###
    file = train()
    save_pkl(file, TEAM_NAME)

    # my_network = load_pkl(TEAM_NAME)

    # Code below plays a single game against a random
    #  opponent, think about how you might want to adapt this to
    #  test the performance of your algorithm.
    def choose_move_no_network(observation: np.ndarray, legal_moves: np.ndarray, env) -> int:
        """The arguments in play_game() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(observation, legal_moves, env, None)
        # return len(observation) ** 2

    # check_submission(
    #     TEAM_NAME, choose_move_no_network
    # )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    play_n_games(
        n=10,
        # your_choose_move=choose_move_randomly,
        # opponent_choose_move=choose_move_randomly
        your_choose_move=choose_move,
        opponent_choose_move=choose_move_randomly,
    )

    # play_go(
    #     your_choose_move=choose_move_no_network,
    #     opponent_choose_move=choose_move_no_network,
    #     game_speed_multiplier=10,
    #     render=False,
    #     verbose=True,
    # )

