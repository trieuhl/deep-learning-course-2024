import torch
from src.mlp.dataloader import DataLoader
from src.mlp.mlp import MLP
from src.mlp.trainer import Trainer


def main():
    # load data
    dataloader = DataLoader(batch_size=4)
    trainloader, testloader = dataloader.load(path='data/cifar10', download=False)
    n_classes = len(dataloader.classes)

    # model
    n_hidden_nodes = 100
    model = MLP(
        n_hidden_nodes=n_hidden_nodes,
        n_classes=n_classes,
        activation="relu",
        image_height=32,
        image_width=32,
        color_channels=3
    )

    # optimizer
    learning_rate = 0.005
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # train
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=trainloader
    )
    n_epochs = 10

    for epoch in range(1, n_epochs + 1):
        trainer.train(
            epoch=epoch
        )
        trainer.validate(
            validation_loader=testloader
        )


if __name__ == '__main__':
    main()
