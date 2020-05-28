import numpy as np
import torch


class SimpleRegressionClassifier:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.Xs = torch.from_numpy(x)
        self.Ys = torch.from_numpy(y)
        self.w = torch.randn(2, 3, requires_grad=True)
        self.b = torch.randn(2, requires_grad=True)

    def predict(self):
        return self.Xs @ self.w.t() + self.b

    @staticmethod
    def mse(t1, t2):
        diff = (t1 - t2) ** 2
        return torch.sum(diff) / diff.numel()

    def train(self, epochs=500, step=1e-5):
        for i in range(epochs):
            predictions = self.predict()
            loss = self.mse(predictions, self.Ys)
            loss.backward()
            with torch.no_grad():
                # gradient descent
                self.w -= self.w.grad * step
                self.b -= self.b.grad * step
                # following lines are necessary because pytorch accumulates grad after each time we call .backward()  # noqa
                self.w.grad.zero_()
                self.b.grad.zero_()


Xs = np.array([[73, 67, 43],
               [91, 88, 64],
               [87, 134, 58],
               [102, 43, 37],
               [69, 96, 70]], dtype='float32')
Ys = np.array([[56, 70],
               [81, 101],
               [119, 133],
               [22, 37],
               [103, 119]], dtype='float32')
classifier = SimpleRegressionClassifier(x=Xs, y=Ys)
classifier.train(epochs=10000)
Y_tilde = classifier.predict()
print('final loss:', classifier.mse(Y_tilde, classifier.Ys).item())
print(Y_tilde)
print(classifier.Ys)
