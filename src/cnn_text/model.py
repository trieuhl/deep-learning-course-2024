import torch.nn
from torch import nn


class SentimentCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_sizes, output_size, drop_prob):
        super(SentimentCNN, self).__init__()
        # layers
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)

        # cnn
        self.convs_1d_layers = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim), padding=(k - 2, 0))
            for k in kernel_sizes])

        # fc
        self.fc_layer = nn.Linear(len(kernel_sizes) * num_filters, output_size)

        # 4. dropout and sigmoid layers
        self.dropout = nn.Dropout(drop_prob)
        self.activation = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        """
        Convolutional + max pooling layer
        """
        # squeeze last dim to get size: (batch_size, num_filters, conv_seq_length)
        # conv_seq_length will be ~ 200
        x = torch.nn.ReLU()(conv(x)).squeeze(3)

        # 1D pool over conv_seq_length
        # squeeze to get size: (batch_size, num_filters)
        x_max = torch.nn.MaxPool1d(x, x.size(2)).squeeze(2)
        return x_max

    def forward(self, x):
        embeds = self.embedding_layer(x)  # (batch_size, seq_length, embedding_dim)
        embeds = embeds.unsqueeze(1)
        conv_results = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d_layers]
        x = torch.cat(conv_results, 1)
        x = self.dropout(x)
        logit = self.fc_layer(x)
        out = self.activation(logit)
        return out
