import torch


class MLP(torch.nn.Module):
    def __init__(self, n_hidden_nodes, image_width, image_height, color_channels, n_classes, activation="None"):
        super(MLP, self).__init__()

        self.n_hidden_nodes = n_hidden_nodes
        self.n_classes = n_classes

        if activation == "sigmoid":
            self.activation_function = torch.nn.Sigmoid()
        elif activation == "relu":
            self.activation_function = torch.nn.ReLU()

        self.image_width = image_width
        self.image_height = image_height
        self.color_channels = color_channels
        input_size = image_width * image_height * color_channels

        # hidden layer
        self.hidden_layer = torch.nn.Linear(input_size, self.n_hidden_nodes)

        # output layer
        self.output_layer = torch.nn.Linear(self.n_hidden_nodes, self.n_classes)

    def forward(self, x):
        # transform
        x = x.view(-1, self.image_width * self.image_height * self.color_channels)

        # z1 = Wx + b
        z = self.hidden_layer(x)

        # activation: a1
        a = self.activation_function(z)

        # output
        y = self.output_layer(a)

        # softmax
        predicted_class = torch.nn.functional.log_softmax(y)

        return predicted_class
