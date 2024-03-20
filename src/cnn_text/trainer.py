import torch
import numpy as np


class Trainer():
    def __init__(self, model, optimizer, criterion, train_loader):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader

    def train(self, epoch, log_interval=100):
        # self.model.train()

        # loop through mini-batches
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # reset
            self.optimizer.zero_grad()

            # forward
            pred = self.model(data)

            # loss
            loss = self.criterion(pred.squeeze(), target.float())

            # backpropagation
            loss.backward()
            self.optimizer.step()

            # log
            if batch_idx % log_interval == 0:
                print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.data.item()))

    def validate(self, validation_loader):
        self.model.eval()

        total_loss = 0
        total_acc = 0

        # loop through mini-batches
        for data, target in validation_loader:
            # forward
            pred = self.model(data)
            pred_class = torch.round(pred.squeeze())  # rounds to the nearest integer

            # loss
            loss = self.criterion(pred.squeeze(), target.float())
            total_loss += loss.item()

            # accuracy
            acc_tensor = pred_class.eq(target.float().view_as(pred_class))
            acc = np.squeeze(acc_tensor.numpy())
            total_acc += np.sum(acc)

        avg_loss = total_loss / len(validation_loader)
        avg_acc = total_acc / len(validation_loader.dataset)
        accuracy = 100. * avg_acc

        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avg_loss, avg_acc, len(validation_loader.dataset), accuracy))
