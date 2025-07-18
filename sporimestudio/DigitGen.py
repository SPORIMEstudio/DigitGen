import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

# Device checking cou ha ya gpu I know I have not gpu :( 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output folder where images stored after 10 epochs
os.makedirs("dcgan_generated", exist_ok=True)

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

#Desi  Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 7, 1, 0, bias=False),   # Output: 256 x 7 x 7
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),   # Output: 128 x 14 x 14
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),     # Output: 1 x 28 x 28
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, 100, 1, 1)
        return self.model(x)

#Desi Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),              # Output: 64 x 14 x 14
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),            # Output: 128 x 7 x 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

#  Models 
generator = Generator().to(device)
discriminator = Discriminator().to(device)

#  Loss and Optimizers
criterion = nn.BCELoss()
lr = 0.0002

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

#  Learning rate scheduler (optional)
scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.5)
scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.5)

#  Training with 100 epochs increased to 500 for blast 😁
epochs = 100
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)

        # === Train Discriminator ===
        z = torch.randn(batch_size, 100, device=device)
        fake_images = generator(z)

        real_labels = torch.full((batch_size, 1), 0.9, device=device)  # label smoothing
        fake_labels = torch.zeros(batch_size, 1, device=device)

        outputs_real = discriminator(real_images)
        loss_real = criterion(outputs_real, real_labels)

        outputs_fake = discriminator(fake_images.detach())
        loss_fake = criterion(outputs_fake, fake_labels)

        loss_D = loss_real + loss_fake
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # === Train Generator ===
        z = torch.randn(batch_size, 100, device=device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        loss_G = criterion(outputs, real_labels)  # wants output to be 1

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    scheduler_G.step()
    scheduler_D.step()

    #  Save output
    if (epoch + 1) % 10 == 0:
        save_image(fake_images[:25], f"dcgan_generated_by_Sporimestudio/epoch_{epoch+1}.png", nrow=5, normalize=True)

    print(f"Epoch [{epoch+1}/{epochs}] Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")