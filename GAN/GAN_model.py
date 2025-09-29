import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import time

#parameters
num_lines = 5
bit_length = 32
noise_dim = 100
epochs = 200
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1337
random.seed(seed)
torch.manual_seed(seed)

#dataset loader
class MIPSDataset(Dataset):
    def __init__(self, filepath):
        self.data = []
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if len(line.strip()) == 32]
        for i in range(0, len(lines) - num_lines + 1, num_lines):
            stack = lines[i:i + num_lines]
            tensor_stack = torch.tensor([[int(b) for b in line] for line in stack], dtype=torch.float32)
            self.data.append(tensor_stack)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

#models
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_lines * bit_length),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.model(z)
        return out.view(-1, num_lines, bit_length)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_lines * bit_length, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.model(x)

#training
def train_gan(generator, discriminator, dataloader, save_prefix):
    generator.to(device).train()
    discriminator.to(device).train()

    g_opt = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()

    start_time = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        for real_batch in dataloader:
            real_batch = real_batch.to(device) * 2 - 1
            batch_size = real_batch.size(0)

            z = torch.randn(batch_size, noise_dim, device=device)
            fake_batch = generator(z).detach()

            d_real = discriminator(real_batch)
            d_fake = discriminator(fake_batch)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            d_loss = loss_fn(d_real, real_labels) + loss_fn(d_fake, fake_labels)

            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            d_opt.step()

            z = torch.randn(batch_size, noise_dim, device=device)
            fake_batch = generator(z)
            g_loss = loss_fn(discriminator(fake_batch), real_labels)

            g_opt.zero_grad(set_to_none=True)
            g_loss.backward()
            g_opt.step()

        if (epoch + 1) % 10 == 0:
            print(f"[{save_prefix}] Epoch {epoch+1}/{epochs} | D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

    torch.save(generator.state_dict(), f"{save_prefix}_generator.pth")
    torch.save(discriminator.state_dict(), f"{save_prefix}_discriminator.pth")
    print(f"[{save_prefix}] Models saved.")

    #profiling
    end_time = time.time()
    print(f"[{save_prefix}] Training time: {end_time - start_time:.2f} seconds")

    if device.type == 'cuda':
        current = torch.cuda.memory_allocated(device) / (1024 * 1024)
        peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"[{save_prefix}] Current GPU memory: {current:.2f} MB")
        print(f"[{save_prefix}] Peak GPU memory: {peak:.2f} MB")

if __name__ == "__main__":
    datasets = {
        "malware": "Dataset\malware_train_1_cleaned.txt",
        "benign":  "Dataset\benign_train_1_cleaned.txt"
    }

    for label, filepath in datasets.items():
        print(f"\nTraining {label} GAN")
        dataset = MIPSDataset(filepath)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        G = Generator()
        D = Discriminator()
        train_gan(G, D, dataloader, save_prefix=label)

    print("\nTraining complete. Models saved for both malware and benign.")
