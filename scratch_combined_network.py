import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# This is a study to determine the correct shape of the network. It assumes the board is 2d like in Delta's code rather than 3d like in AGZ.

class Net(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Net, self).__init__()

        # HYPERPARAMS
        # TODO pass all of these in as params
        num_shared_conv_layers = 2
        conv_kernel_width = 3
        num_conv_kernels = 100
        pool_kernel_width = 3
        num_shared_linear_layers = 2
        num_hidden_linear_layer_neurons = 100
        num_non_shared_linear_layers = 3
        num_policy_output_dim = 9 * 9 + 1
        num_value_output_dim = 1

        self.shared_layers = nn.ModuleList()
        self.policy_layers = nn.ModuleList()
        self.value_layers = nn.ModuleList()

        # ESTABLISH THE SHARED LAYERS

        for i in range(num_shared_conv_layers):
            self.shared_layers.append(
                # For the first one, in_channels=1, then for the subsequent ones in_channels=num_conv_kernels
                nn.LazyConv2d(out_channels=num_conv_kernels, kernel_size=conv_kernel_width, padding=1)
            )

            # NOTE BatchNorm requires there to be a batch dimension (I think), so you have to reshape your board from the 1x9x9 required for conv layers to instead a 1x1x9x9.
            self.shared_layers.append(
                nn.LazyBatchNorm2d()
                # nn.BatchNorm2d(num_features=num_conv_kernels)
            )

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
        self.policy_layers.append(
            nn.Softmax(dim=-1)
        )

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

    def forward(self, x):
        for layer in self.shared_layers:
            x = layer(x)

        p = x
        for layer in self.policy_layers:
            p = layer(p)
        
        v = x
        for layer in self.value_layers:
            v = layer(v)

        return p, v

network = Net(None, None, None)

observation = torch.rand(9, 9, dtype=torch.float)

# NOTE Need to add a batch dimension in order for BatchNorm to work. So that means the shape has to be (1, 1, 9, 9) instead of (1, 9, 9).
t = observation.reshape(1, 1, 9, 9)
print("shape of t:", t.shape)

result = network(t)
print("result is", result)

# summary(network, input_size=(1, 1, 28, 28))
summary(network, input_size=(1, 1, 9, 9))

