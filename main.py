from lib2to3.pytree import convert
from typing import Any, Dict

import numpy as np

from tqdm import tqdm
import time
import random
from copy import copy, deepcopy

# from check_submission import check_submission
from game_mechanics import GoEnv, choose_move_randomly, load_pkl, play_go, save_pkl, BOARD_SIZE

import torch
from torch import nn
from torchinfo import summary

# from mcts.state import State
# from mcts.mcts import MCTS
from net import Net
from mcts import mcts

from config import device

# PARAMETERS
NUM_MOVES = BOARD_SIZE * BOARD_SIZE + 1
NUM_MCTS_FOR_TRAINING = 50 # TODO Change to 1000

NUM_MCTS_FOR_FINAL_TESTING = 50
NUM_GAMES_FOR_FINAL_TESTING = 100

# Training params
TRAINING_GAMES_TO_PLAY = 500 # 500 took like 2 hours, gonna leave overnight
CHECK_WHETHER_TO_REPLACE_OPPONENT_EVERY_N_GAMES = 50
LR = 1e-3
GAMES_TO_PLAY_WHEN_SEEING_IF_YOURE_BETTER_THAN_OPPONENT = 100
WIN_PERCENTAGE_TO_REPLACE_OPPONENT = 0.6
SAVE_CHECKPOINT_EVERY_N_GAMES = 100
REGULARIZATION_PARAMETER = 1e-4

# NETWORK HYPERPARAMETERS
# Currently in the net.py file.

TEAM_NAME = "OPEC"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"

def initialize_network():
    network = Net(board_size=BOARD_SIZE)
    network.to(device)

    # Initialize lazy layers of the network by passing a dummy tensor of the correct shape through it.
    dummy_observation = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    dummy_legal_moves = np.arange(NUM_MOVES, dtype=int)
    _ = network(dummy_observation, dummy_legal_moves)
    # summary(network, input_size=(1, 1, BOARD_SIZE, BOARD_SIZE))

    return network


def reestablish_env_in_training(opponent_choose_move, opponent_network):
    # if opponent_choose_move != choose_move_randomly:
    #     print("NO FUNCTION PROBLEM! OCM IS SOMETHING OTHER THAN RANDOM. THIS IS GOOD!")
    return GoEnv(
        convert_to_no_network(opponent_choose_move, opponent_network),
        verbose=False,
        # TODO Pass stuff through for this
        # verbose=long_games_count > 2,
        render=False,
        game_speed_multiplier=1,
    )


def copy_network_for_opponent(network):
    # This didn't work for me. I got this error in some cases:
    # RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment
    # return deepcopy(network)
    filepath = "temp_for_copying.pt"
    torch.save(network.state_dict(), filepath)
    new_network = initialize_network()
    new_network.load_state_dict(torch.load(filepath))
    # I don't think this is needed but whatever.
    new_network.clear_cache()
    # Opponent's network is not going to be trained. This probably doesn't matter though.
    new_network.eval()
    return new_network


# Replace the opponent if you are better than them.
def maybe_get_new_opponent(network, opponent_choose_move, opponent_network, env):
    print("--CHECKING IF BETTER THAN OPPONENT--")

    # If you are better than the opponent, then replace their choose_move function with one that uses a clone of your network.
    win_percentage = play_n_games(
        n=GAMES_TO_PLAY_WHEN_SEEING_IF_YOURE_BETTER_THAN_OPPONENT,
        your_choose_move=choose_move_without_mcts,
        your_network=network,
        opponent_choose_move=opponent_choose_move,
        opponent_network=opponent_network,
    )
    print(f"Won {win_percentage} games")

    better_than_opponent = win_percentage > WIN_PERCENTAGE_TO_REPLACE_OPPONENT
    if not better_than_opponent:
        return opponent_choose_move, opponent_network, env

    if opponent_choose_move == choose_move_randomly:
        print("****************REPLACING OPPONENT FOR THE FIRST TIME****************")
    else:
        print("****************REPLACING OPPONENT****************")

    # This is a good network. Let's save it.
    save_checkpoint_during_training(network)

    new_opponent_network = copy_network_for_opponent(network)
    new_opponent_network.to(device)
    new_opponent_choose_move = opponent_choose_move_for_training

    # Need to re-establish the env because it has the opponent baked in.
    new_env = reestablish_env_in_training(opponent_choose_move=new_opponent_choose_move, opponent_network=new_opponent_network)

    return new_opponent_choose_move, new_opponent_network, new_env


def train(existing_network=None):

    network = existing_network if existing_network else initialize_network()
    # opponent_network = None

    p_loss_fn = nn.CrossEntropyLoss()
    v_loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(network.parameters(), lr=LR)

    long_games_count = 0

    # Opponent starts as random, and we only replace it when we are able to beat it.
    opponent_choose_move = choose_move_randomly
    opponent_network = None
    env = reestablish_env_in_training(opponent_choose_move=opponent_choose_move, opponent_network=opponent_network)

    print()
    print(f"PLAYING {TRAINING_GAMES_TO_PLAY} GAMES FOR TRAINING:")
    for game_idx in tqdm(range(TRAINING_GAMES_TO_PLAY)):
        
        p_history, pi_history, v_history, z_history = [], [], [], []
        
        observation, reward, done, info = env.reset()
        legal_moves = info["legal_moves"]
        done = False

        moves_in_game = 0 # Purely for counting purposes

        while not done:
            moves_in_game += 1

            p, v = network.forward_with_cache(observation, legal_moves)
            # TODO Actually write the mcts
            pi = mcts(n=NUM_MCTS_FOR_TRAINING, observation=observation, legal_moves=legal_moves, env=env, network=network)
            # pi = p # Noop to test training code

            p_history.append(p)
            pi_history.append(pi)
            v_history.append(v)

            # TODO Need to do this randomly with temperature — or else it might not learn?
            # action = torch.argmax(pi).item()
            action = torch.distributions.Categorical(probs=pi).sample().item()

            observation, reward, done, info = env.step(action)
            legal_moves = info["legal_moves"]

            if moves_in_game % 100 == 0:
                print(f"MOVES IN MIDDLE OF GAME: {moves_in_game}")


        # print(f"MOVES AT END OF GAME: {moves_in_game} (reward {reward})")
        if moves_in_game > 1000:
            long_games_count += 1
        # TODO Consider breaking if MOVES IN GAME exceeds like 200 or 500
        
        # TODO Could optimize this.
        for _ in range(len(p_history)):
            z_history.append(torch.tensor([reward], dtype=torch.float, device=device))

        # TODO Need to stack the lists into tensors
        p_loss = p_loss_fn(torch.stack(p_history), torch.stack(pi_history))
        v_loss = v_loss_fn(torch.stack(v_history), torch.stack(z_history))
        # Assert it is nonnegative (so we know whether to add a minus sign)
        assert p_loss >= 0
        assert v_loss >= 0
        regularization_term = REGULARIZATION_PARAMETER * nn.utils.parameters_to_vector(network.parameters()).square().sum()
        loss = p_loss + v_loss + regularization_term

        opt.zero_grad()
        loss.backward()
        opt.step()
        # Need to clear the network cache after you train...would be nice to just hook into that but need to research that.
        network.clear_cache()

        should_check_if_better_than_opponent = game_idx % CHECK_WHETHER_TO_REPLACE_OPPONENT_EVERY_N_GAMES == 0
        if should_check_if_better_than_opponent:
            # We are reassigning these variables which are defined above.
            opponent_choose_move, opponent_network, env = maybe_get_new_opponent(network, opponent_choose_move=opponent_choose_move, opponent_network=opponent_network, env=env)
        
        if game_idx % SAVE_CHECKPOINT_EVERY_N_GAMES == 0:
            save_checkpoint_during_training(network)
    
    return network


def save_checkpoint_during_training(network):
    filename = f"{TEAM_NAME}_training_checkpoint"
    # TODO May need something like this if using GPU, but it would impede training.
    # file.to("cpu")
    save_pkl(network, filename)


def opponent_choose_move_for_training(observation: np.ndarray, legal_moves: np.ndarray, env, neural_network: nn.Module) -> int:
    p, v = neural_network.forward_with_cache(observation, legal_moves)
    # Do not have the opponent use MCTS. TODO on whether they actually should...but it slows things down.
    action = torch.distributions.Categorical(probs=p).sample().item()
    return action


def choose_move_without_mcts(observation: np.ndarray, legal_moves: np.ndarray, env, neural_network: nn.Module) -> int:
    p, v = neural_network.forward_with_cache(observation, legal_moves)
    return p.argmax().item()


def choose_move(observation: np.ndarray, legal_moves: np.ndarray, env, neural_network: nn.Module) -> int:
    # return choose_move_randomly(observation, legal_moves, env)
    pi = mcts(n=NUM_MCTS_FOR_FINAL_TESTING, observation=observation, legal_moves=legal_moves, env=env, network=neural_network)
    return pi.argmax().item()

    # TODO Use MCTS to improve this move before returning a move.
    # p, v = neural_network.forward_with_cache(observation, legal_moves)
    # return p.argmax().item()

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


if __name__ == "__main__":

    # TODO Be sure you want this
    TRAIN_FROM_SCRATCH = True
    existing_network = None if TRAIN_FROM_SCRATCH else load_pkl(TEAM_NAME)

    ## Example workflow, feel free to edit this! ###
    file = train(existing_network)
    # file.to("cpu") # TODO May have to change this
    save_pkl(file, TEAM_NAME)

    # my_network = load_pkl(TEAM_NAME + "_training_checkpoint")
    my_network = load_pkl(TEAM_NAME)
    my_network.to(device) # TODO May have to change this for submission

    # Code below plays a single game against a random
    #  opponent, think about how you might want to adapt this to
    #  test the performance of your algorithm.
    def choose_move_no_network(observation: np.ndarray, legal_moves: np.ndarray, env) -> int:
        """The arguments in play_game() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(observation, legal_moves, env, my_network)
        # return len(observation) ** 2

    # check_submission(
    #     TEAM_NAME, choose_move_no_network
    # )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    play_n_games(
        n=NUM_GAMES_FOR_FINAL_TESTING,
        # your_choose_move=choose_move_randomly,
        # opponent_choose_move=choose_move_randomly
        your_choose_move=choose_move,
        your_network=my_network,
        opponent_choose_move=choose_move_randomly,
        opponent_network=None,
        verbose=True,
    )

    # play_go(
    #     your_choose_move=choose_move_no_network,
    #     opponent_choose_move=choose_move_no_network,
    #     game_speed_multiplier=10,
    #     render=False,
    #     verbose=True,
    # )

