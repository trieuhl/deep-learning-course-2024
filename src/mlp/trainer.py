import torch


class Trainer():
    def __init__(self, model, optimizer, train_loader):
        self.model = model
        self.optimizer = optimizer
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
            loss = torch.nn.functional.nll_loss(pred, target)

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
            # get the index of the max log-probability
            pred_class = pred.data.max(1)[1]

            # loss
            loss = torch.nn.functional.nll_loss(pred, target)
            total_loss += loss.item()

            # accuracy
            acc = pred_class.eq(target.data).cpu().sum()
            total_acc += acc

        avg_loss = total_loss / len(validation_loader)
        avg_acc = total_acc / len(validation_loader.dataset)
        accuracy = 100. * avg_acc

        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avg_loss, avg_acc, len(validation_loader.dataset), accuracy))
