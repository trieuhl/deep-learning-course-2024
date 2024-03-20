import torch
from src.mlp.dataloader import DataLoader
from src.cnn.cnn import CNN
from src.cnn.trainer import Trainer


def main():
    # load data
    dataloader = DataLoader(batch_size=4)
    trainloader, testloader = dataloader.load(path='data/cifar10', download=False)
    n_classes = len(dataloader.classes)

    # model
    n_hidden_nodes = 100
    model = CNN(
        activation="relu",
        in_chanels=3,
        out_chanels_1=6,
        out_chanels_2=16,
        kernel_size=5,
        pooling_kernel_size=2,
        hidden_size_1=120,
        hidden_size_2=84,
        n_classes=n_classes
    )

    # optimizer
    learning_rate = 0.005
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

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
