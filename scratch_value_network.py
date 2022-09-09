import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# This is a study to determine the correct shape of the network.

# NOTE ValueNet is for initial testing. It should ultimately be multiheaded and called Net.
class ValueNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(ValueNet, self).__init__()

        # HYPERPARAMS
        num_conv_layers = 2
        conv_kernel_width = 3
        num_conv_kernels = 100
        pool_kernel_width = 3
        num_linear_layers = 2
        num_hidden_linear_layer_neurons = 100
        num_output_dim = 1 # this is the number of output neurons, which for a value fn should be 1

        self.layers = nn.ModuleList()

        for i in range(num_conv_layers):
            self.layers.append(
                # For the first one, in_channels=17, then for the subsequent ones in_channels=num_conv_kernels
                nn.LazyConv2d(out_channels=num_conv_kernels, kernel_size=conv_kernel_width, padding=1)
            )

            # TODO Idk why this doesn't work... error is:
            # ValueError: expected 4D input (got 3D input)
            # self.layers.append(
            #     nn.BatchNorm2d(num_features=num_conv_kernels)
            #     # nn.LazyBatchNorm2d()
            # )

            self.layers.append(
                nn.ReLU()
            )

            # TODO AlphaGo did not use pooling afaik. But if I don't pool then the first linear layer needs a shit ton of neurons after the flatten. I think the pooling is pretty stupid though and should be removed.
            # is_last = i == num_conv_layers - 1
            # if not is_last:
            #     self.layers.append(
            #         nn.MaxPool2d(kernel_size=pool_kernel_width)
            #     )

        self.layers.append(
            nn.Flatten(start_dim=0)
        )

        for i in range(num_linear_layers):
            is_last = i == num_linear_layers - 1
            if not is_last:
                self.layers.append(
                    nn.LazyLinear(out_features=num_hidden_linear_layer_neurons)
                )
                self.layers.append(
                    nn.ReLU()
                )
            else:
                self.layers.append(
                    nn.LazyLinear(out_features=num_output_dim)
                )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

network = ValueNet(None, None, None)

# This works
# network = nn.Sequential(
#     # in_channels = 17 here but it's implicit
#     nn.LazyConv2d(out_channels=20, kernel_size=3)
# )

observation = torch.rand(9, 9, 17, dtype=torch.float)

# UserWarning: The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor.
# t = observation.T
t = observation.permute(*torch.arange(observation.ndim - 1, -1, -1))
print("shape of t:", t.shape)

result = network(t)
print("result is", result)

# summary(network, input_size=(1, 1, 28, 28))
summary(network, input_size=(17, 9, 9))
