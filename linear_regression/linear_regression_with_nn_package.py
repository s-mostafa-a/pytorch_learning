import numpy as np
import torch


class RegressionClassifierWithoutNN:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.Xs = torch.from_numpy(x)
        self.Ys = torch.from_numpy(y)
        self.w = torch.randn(self.Ys.size()[1], self.Xs.size()[1], requires_grad=True)
        self.b = torch.randn(self.Ys.size()[1], requires_grad=True)

    def predict(self):
        return self.Xs @ self.w.t() + self.b

    @staticmethod
    def mse(t1, t2):
        diff = (t1 - t2) ** 2
        return torch.sum(diff) / diff.numel()

    @staticmethod
    def absolute(t1, t2):
        diff = torch.abs(t1 - t2)
        return torch.sum(diff) / diff.numel()

    def train(self, epochs=500, step=1e-5, loss_func='mse'):
        for i in range(epochs):
            predictions = self.predict()
            if loss_func == 'mse':
                loss = self.mse(predictions, self.Ys)
            elif loss_func == 'abs':
                loss = self.absolute(predictions, self.Ys)
            else:
                raise Exception('Not implemented such loss function')
            loss.backward()
            with torch.no_grad():
                # gradient descent
                self.w -= self.w.grad * step
                self.b -= self.b.grad * step
                # following lines are necessary because pytorch accumulates grad after each time we call .backward()  # noqa
                self.w.grad.zero_()
                self.b.grad.zero_()


Xs = np.array([[73, 67, 43, 12],
               [91, 88, 64, 23],
               [87, 134, 58, 10],
               [102, 43, 37, 42],
               [69, 96, 70, 91]], dtype='float32')
Ys = np.array([[56, 70, 31],
               [81, 101, 10],
               [119, 133, 211],
               [22, 37, 49],
               [103, 119, 20]], dtype='float32')
classifier = RegressionClassifierWithoutNN(x=Xs, y=Ys)
classifier.train(epochs=10000)
Y_tilde = classifier.predict()
print('final loss:', classifier.mse(Y_tilde, classifier.Ys).item())
print(Y_tilde)
print(classifier.Ys)
