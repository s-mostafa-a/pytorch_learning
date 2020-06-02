import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os


class MNIST_GANS:
    def __init__(self, dataset, image_size, criterion=nn.BCELoss(), batch_size=100,
                 hidden_size=2561, latent_size=64):
        self.data_loader = DataLoader(dataset, batch_size, shuffle=True)
        self.criterion = criterion
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.D = nn.Sequential(
            nn.Linear(image_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())
        self.G = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh())
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002)
        self.sample_dir = '../data/samples'
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        self.sample_vectors = torch.randn(self.batch_size, self.latent_size)

    @staticmethod
    def denormalize(x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def train_discriminator(self, images):
        real_labels = torch.ones(self.batch_size, 1)
        fake_labels = torch.zeros(self.batch_size, 1)

        outputs = self.D(images)
        d_loss_real = self.criterion(outputs, real_labels)
        real_score = outputs

        z = torch.randn(self.batch_size, self.latent_size)
        fake_images = self.G(z)
        outputs = self.D(fake_images)
        d_loss_fake = self.criterion(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss, real_score, fake_score

    def train_generator(self):
        z = torch.randn(self.batch_size, self.latent_size)
        fake_images = self.G(z)
        labels = torch.ones(self.batch_size, 1)
        g_loss = self.criterion(self.D(fake_images), labels)

        self.reset_grad()
        g_loss.backward()
        self.g_optimizer.step()
        return g_loss, fake_images

    def save_fake_images(self, index):
        fake_images = self.G(self.sample_vectors)
        fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
        fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
        print('Saving', fake_fname)
        save_image(self.denormalize(fake_images), os.path.join(self.sample_dir, fake_fname),
                   nrow=10)

    def run(self):
        num_epochs = 5
        total_step = len(self.data_loader)
        d_losses, g_losses, real_scores, fake_scores = [], [], [], []

        for epoch in range(num_epochs):
            for i, (images, _) in enumerate(self.data_loader):
                images = images.reshape(self.batch_size, -1)

                d_loss, real_score, fake_score = self.train_discriminator(images)
                g_loss, fake_images = self.train_generator()

                if (i + 1) % 200 == 0:
                    d_losses.append(d_loss.item())
                    g_losses.append(g_loss.item())
                    real_scores.append(real_score.mean().item())
                    fake_scores.append(fake_score.mean().item())
                    print(f'''Epoch [{epoch}/{num_epochs}], Step [{i + 1}/{
                    total_step}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, D(x): {
                    real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}''')
            self.save_fake_images(epoch + 1)


image_size = 784
mnist = MNIST(root='../data', train=True,
              transform=Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))
gans = MNIST_GANS(dataset=mnist, image_size=image_size)
gans.run()
