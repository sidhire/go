from lib2to3.pytree import convert
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

import numpy as np

from config import device

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


class Net(nn.Module):

    def __init__(self, board_size):
        super(Net, self).__init__()

        # HYPERPARAMS
        # TODO pass all of these in as params
        num_shared_conv_layers = 2 # 2
        conv_kernel_width = 3
        num_conv_kernels = 20 # 100
        pool_kernel_width = 3
        num_shared_linear_layers = 2 # 2
        num_hidden_linear_layer_neurons = 10 # 100
        num_non_shared_linear_layers = 2 # 3
        num_policy_output_dim = board_size * board_size + 1
        num_value_output_dim = 1

        self.shared_layers = nn.ModuleList()
        self.policy_layers = nn.ModuleList()
        self.value_layers = nn.ModuleList()

        self.cache = {}

        # ESTABLISH THE SHARED LAYERS

        for i in range(num_shared_conv_layers):
            self.shared_layers.append(
                # For the first one, in_channels=1, then for the subsequent ones in_channels=num_conv_kernels
                nn.LazyConv2d(out_channels=num_conv_kernels, kernel_size=conv_kernel_width, padding=1)
            )

            # NOTE BatchNorm requires there to be a batch dimension (I think), so you have to reshape your board from the 1x9x9 required for conv layers to instead a 1x1x9x9.
            # TODO BatchNorm is slow per my profiling. Re-enable it if I'm running on a GPU.
            # self.shared_layers.append(
            #     nn.LazyBatchNorm2d() # nn.BatchNorm2d(num_features=num_conv_kernels)
            # )

            self.shared_layers.append(
                nn.ReLU()
            )

            # TODO AlphaGo did not use pooling afaik. But if I don't pool then the first linear layer needs a shit ton of neurons after the flatten. I think the pooling is pretty stupid though and should be removed.
            # is_last = i == num_shared_conv_layers - 1
            # if not is_last:
            #     self.shared_layers.append(
            #         nn.MaxPool2d(kernel_size=pool_kernel_width)
            #     )

        self.shared_layers.append(
            nn.Flatten(start_dim=0)
        )

        for i in range(num_shared_linear_layers):
            self.shared_layers.append(
                nn.LazyLinear(out_features=num_hidden_linear_layer_neurons)
            )
            self.shared_layers.append(
                nn.ReLU()
            )
    
        # ESTABLISH THE POLICY LAYERS
        for i in range(num_non_shared_linear_layers - 1):
            self.policy_layers.append(
                nn.LazyLinear(out_features=num_hidden_linear_layer_neurons)
            )
            self.policy_layers.append(
                nn.ReLU()
            )
        self.policy_layers.append(
            nn.LazyLinear(out_features=num_policy_output_dim)
        )
        # NOTE Instead of adding the softmax here, we add it in the forward function so that we can do legal action masking.
        # self.policy_layers.append(
        #     nn.Softmax(dim=-1)
        # )

        # ESTABLISH THE VALUE LAYERS
        for i in range(num_non_shared_linear_layers - 1):
            self.value_layers.append(
                nn.LazyLinear(out_features=num_hidden_linear_layer_neurons)
            )
            self.value_layers.append(
                nn.ReLU()
            )
        self.value_layers.append(
            nn.LazyLinear(out_features=num_value_output_dim)
        )

    # I think both observation and legal_moves can be either np arrays or tensors.
    def forward(self, observation, legal_moves):

        # TODO This feels like a bad abstraction...should probably be doing this transformation outside the forward function.
        board_size = observation.shape[0]
        x = torch.tensor(observation, dtype=torch.float, device=device)
        x = x.reshape(1, 1, board_size, board_size)

        for layer in self.shared_layers:
            x = layer(x)

        p = x
        for layer in self.policy_layers:
            p = layer(p)
        # We do the softmax here instead of adding it as a layer so that we can do legal action masking.
        p = softmax_with_legal_move_masking(p, legal_moves, board_size)
        
        v = x
        for layer in self.value_layers:
            v = layer(v)

        return p, v
    
    def forward_with_cache(self, observation, legal_moves):
        # If these assertions are false, need to rethink the hashing since .tobytes() works on np arrays but not on torch tensors.
        assert type(observation) == np.ndarray
        assert type(legal_moves) == np.ndarray

        # Probably don't need to hash legal_moves since it is implicit in the observation.
        # hash = observation.tobytes() + legal_moves.tobytes()
        hash = observation.tobytes()

        if hash in self.cache:
            # print("Cache HITTTTTTTTTT!")
            return self.cache[hash]
        # else:
        #     print("Cache miss")

        result = self.forward(observation, legal_moves)
        self.cache[hash] = result
        return result
    
    def clear_cache(self):
        self.cache = {}
