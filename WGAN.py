import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Hyperparameters

BATCH_SIZE = 100
EPOCHS = 15
CRITIC_ITERATION = 5
Z_DIM = 100
LEARNING_RATE = 1e-4
WEIGHT_CLIP = 1e-2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading

dataset = datasets.MNIST(root='/files/', train=True, download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
]))

train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

fraction = len(dataset) / BATCH_SIZE

# Class defining

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            self._block(1, 32, kernel_size=5, stride=1, padding=0),  # (N, 32, 24, 24)
            self._block(32, 64, kernel_size=3, stride=2, padding=1),  # (N, 64, 12, 12)
            self._block(64, 128, kernel_size=3, stride=2, padding=1),  # (N, 128, 6, 6)
            self._block(128, 256, kernel_size=3, stride=2, padding=1),  # (N, 256, 3, 3)
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=0),
            nn.Flatten(),  # (N, 1)
        )

    def _block(self, in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), # (N, 32, 24, 24)
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, 256, kernel_size=3, stride=2, padding=0, output_padding=0),  # (N, 256, 3, 3)
            self._block(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (N, 128, 6, 6)
            self._block(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (N, 64, 12, 12)
            self._block(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (N, 32, 24, 24)
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=1, padding=0, output_padding=0),  # (N, 1, 28, 28)
            nn.Sigmoid()
        )

    def _block(self, in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0, output_padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.gen(x)

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


critic = Critic().to(DEVICE)
gen = Generator(Z_DIM).to(DEVICE)

initialize_weights(critic)
initialize_weights(gen)

opt_disc = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)
opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)

fixed_noise = torch.randn(64, Z_DIM, 1, 1).to(DEVICE)
writer_real = SummaryWriter(f"runs/real")
writer_fake = SummaryWriter(f"runs/fake")

for epoch in range(EPOCHS):
    lossС_pe = 0
    lossG_pe = 0
    train_loop = tqdm(enumerate(train_loader), total=len(train_loader),
                                  desc=f"Epoch: {epoch+1}")
    for batch_idx, (real, _) in train_loop:
        real = real.to(DEVICE)

        for _ in range(CRITIC_ITERATION):
            # Here we want to train discriminator
            # We want to max E[D(x)] - E[D(G(z))], which is equivalent to min -(E[D(x)] - E[D(G(z))])
            noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(DEVICE)
            fake = gen(noise)

            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)

            lossC = -(torch.mean(critic_real) - torch.mean(critic_fake))

            critic.zero_grad()
            lossC.backward(retain_graph=True)
            opt_disc.step()

            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            with torch.no_grad():
                lossС_pe += lossC.item()

        # Here we want to train generator
        # We want to min -E[D(G(z))]

        output = critic(fake).reshape(-1)

        lossG = -torch.mean(output)

        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        with torch.no_grad():
            lossG_pe += lossG.item()
            train_loop.set_postfix(lossD=lossC.item() * fraction, lossG=lossG.item()*fraction)

    with torch.no_grad():
        fake = gen(fixed_noise)

        img_grid_real = torchvision.utils.make_grid(real[:64], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)

        print(f"Epoch: {epoch+1}, LossD: {lossС_pe}, LossG: {lossG_pe}")

        writer_fake.add_image("Fake", img_grid_fake, global_step=epoch+1)
        writer_real.add_image("Real", img_grid_real, global_step=epoch+1)
        writer_real.add_scalar('LossD', lossС_pe, epoch+1)
        writer_fake.add_scalar('LossG', lossG_pe, epoch+1)

writer_fake.close()
writer_real.close()