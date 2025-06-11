"""Generative Adversarial Network (GAN) implementation"""

import torch 
import torch.nn as nn 

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__() 
        
        def block(in_channels, out_channels, normalize=True):
            layers = [nn.Linear(in_channels, out_channels)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            *block(input_dim, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, output_dim), 
            nn.Tanh()  # Output layer with Tanh activation for normalized output
        )

    def forward(self, z):
        img = self.model(z)
        return img




class Discriminator(nn.Module): 
    def _init__(self, input_dim):
        super(Discriminator, self).__init__() 

        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),  # Output layer for binary classification
            nn.Sigmoid()  # Sigmoid activation for output
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

        


def optimizers_G_D(generator, discriminator, learning_rate=0.0002):
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    return generator_optimizer, discriminator_optimizer

def loss_function():
    return nn.BCELoss()

def train_gan(generator, discriminator, dataloader, num_epochs=50, device='cuda'):
    generator.train()
    discriminator.train()

    criterion = loss_function()
    generator_optimizer, discriminator_optimizer = optimizers_G_D(generator, discriminator)

    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            imgs = imgs.view(batch_size, -1).to(device)  # Flatten images

            # Create labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            discriminator_optimizer.zero_grad()

            outputs = discriminator(imgs)
            d_loss_real = criterion(outputs, real_labels)
            d_loss_real.backward()

            z = torch.randn(batch_size, generator.input_dim).to(device)
            fake_imgs = generator(z)
            outputs = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss_fake.backward()

            d_loss = d_loss_real + d_loss_fake
            discriminator_optimizer.step()

            # Train Generator
            generator_optimizer.zero_grad()

            outputs = discriminator(fake_imgs)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            
            generator_optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')


