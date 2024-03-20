import torch
from src.cnn_text.w2v_loader import W2VLoader
from src.cnn_text.dataloader import read_data
from src.cnn_text.dataloader import Loader
from src.cnn_text.model import SentimentCNN
from src.cnn_text.trainer import Trainer


def main():
    # data
    reviews, labels = read_data(samples_num=1000)

    # load word2vec
    w2v = W2VLoader(
        w2v_path='data/word2vec_model/GoogleNews-vectors-negative300.bin'
    )

    # load data
    data_loader = Loader(
        embed_lookup=w2v.embed_lookup,
        batch_size=200,
        seq_length=10,
        split_frac=0.8
    )

    train_loader, valid_loader, test_loader = data_loader.load(reviews, labels)

    # model
    num_filters = 100
    kernel_sizes = [3, 4, 5]
    vocab_size = w2v.vocab_size
    embedding_dim = w2v.embedding_dim

    model = SentimentCNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_filters=num_filters,
        kernel_sizes=kernel_sizes,
        output_size=1,
        drop_prob=0.5
    )

    learning_rate = 0.001
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader
    )
    n_epochs = 10

    for epoch in range(1, n_epochs + 1):
        trainer.train(
            epoch=epoch
        )
        trainer.validate(
            validation_loader=valid_loader
        )

    return


if __name__ == '__main__':
    main()
