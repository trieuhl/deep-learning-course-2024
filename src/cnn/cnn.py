import torch


class CNN(torch.nn.Module):
    def __init__(self, in_chanels, out_chanels_1, out_chanels_2, kernel_size, pooling_kernel_size, hidden_size_1,
                 hidden_size_2, n_classes, activation="None"):
        super(CNN, self).__init__()

        # activation
        if activation == "sigmoid":
            self.activation_function = torch.nn.Sigmoid()
        elif activation == "relu":
            self.activation_function = torch.nn.ReLU()

        # layers
        self.conv1_layer = torch.nn.Conv2d(
            in_channels=in_chanels,
            out_channels=out_chanels_1,
            kernel_size=kernel_size
        )
        self.pooling_layer = torch.nn.MaxPool2d(
            kernel_size=pooling_kernel_size
        )
        self.conv2_layer = torch.nn.Conv2d(
            in_channels=out_chanels_1,
            out_channels=out_chanels_2,
            kernel_size=kernel_size
        )

        self.fc1_layer = torch.nn.Linear(
            in_features=out_chanels_2 * kernel_size * kernel_size,
            out_features=hidden_size_1
        )
        self.fc2_layer = torch.nn.Linear(
            in_features=hidden_size_1,
            out_features=hidden_size_2
        )
        self.out_layer = torch.nn.Linear(
            in_features=hidden_size_2,
            out_features=n_classes
        )

    def forward(self, x):
        # conv1 -> relu -> pooling
        conv1 = self.conv1_layer(x)
        relu1 = self.activation_function(conv1)
        pool1 = self.pooling_layer(relu1)

        # conv2 -> relu -> pooling
        conv2 = self.conv2_layer(pool1)
        relu2 = self.activation_function(conv2)
        pool2 = self.pooling_layer(relu2)

        # flatten all dimensions except batch
        flatten = torch.flatten(pool2, 1)

        # fully-connected
        fc1 = self.activation_function(self.fc1_layer(flatten))
        fc2 = self.activation_function(self.fc2_layer(fc1))

        # out
        out = self.out_layer(fc2)

        return out
