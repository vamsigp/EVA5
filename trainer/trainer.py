from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class Trainer():

    def __init__(self, model, device, train_loader, test_loader, optimizer, loss_func, lr_scheduler):
        self.is_last_epoch = False
        self.train_losses = []  # detailed training loss
        self.test_losses = []
        self.train_acc = []  # detailed  training accuracy
        self.train_acc_total = []  # per epoch training accuracy
        self.train_loss_total = []  # per epoch train loss
        self.test_acc = []
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.lr_scheduler = lr_scheduler

    def train_model(self, lambda_l1, epochs=5):
        for epoch in range(epochs):
            if self.lr_scheduler is not None:
                print("Current EPOCH:", epoch, "last LR=", self.lr_scheduler.get_last_lr(), "LR = ", self.lr_scheduler.get_lr())
            self.train(epoch, lambda_l1)
            self.is_last_epoch = epoch == epochs
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.test()
        return (self.train_loss_total, self.train_acc_total, self.test_losses, self.test_acc)

    def get_detailed_train_stats(self):
        return (self.train_losses, self.train_acc)

    def train(self, epoch, lambda_l1):
        self.model.train()
        pbar = tqdm(self.train_loader)
        correct = 0
        processed = 0
        loss = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(self.device), target.to(self.device)

            # Init
            self.optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            y_pred = self.model(data)

            # Calculate loss
            loss = self.loss_func(y_pred, target)

            # L2 loss

            # L1 loss
            loss_l1 = 0
            # lambda_l1 = 0.05
            if lambda_l1 > 0:
                for p in self.model.parameters():
                    loss_l1 = loss_l1 + p.abs().sum()
                loss = loss + lambda_l1 * loss_l1

            self.train_losses.append(loss)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(
                desc=f'Train set: Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
            self.train_acc.append(100 * correct / processed)

        training_accuracy_perepoch = 100 * correct / processed
        self.train_acc_total.append(training_accuracy_perepoch)
        self.train_loss_total.append(loss)

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss_func(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                is_correct = pred.eq(target.view_as(pred))
                correct += is_correct.sum().item()

        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

        self.test_acc.append(100. * correct / len(self.test_loader.dataset))

    def getValues(self):
        return (self.train_loss_total, self.train_acc_total, self.test_losses, self.test_acc)

    def get_misclassified(self):
        self.model.eval()
        misclassified_imgs = []
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                is_correct = pred.eq(target.view_as(pred))
                if True:
                    misclassified_inds = (is_correct == 0).nonzero()[:, 0]
                    for mis_ind in misclassified_inds:
                        if len(misclassified_imgs) == 25:
                            break
                        misclassified_imgs.append({
                            "target": target[mis_ind].cpu().numpy(),
                            "pred": pred[mis_ind][0].cpu().numpy(),
                            "img": data[mis_ind].cpu().numpy()
                        })
                correct += is_correct.sum().item()

        return misclassified_imgs

    def classwise_acc(self, classes):
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        # print class-wise test accuracies
        print()
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
        print()
