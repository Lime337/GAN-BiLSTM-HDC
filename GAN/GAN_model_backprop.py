import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from GAN_model import Generator, Discriminator, MIPSDataset
import os
import time

#parameters
num_lines = 5
bit_length = 32
noise_dim = 100
epochs = 200
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
critic_path = "Semantic_Critic\semantic_critic_multirule.pt"

#critic model
class TinyBiGRU(nn.Module):
    def __init__(self, emb_dim=32, hidden=128, num_rules: int = 7):
        super().__init__()
        self.emb_op = nn.Embedding(64, emb_dim)
        self.rnn = nn.GRU(input_size=emb_dim + 8, hidden_size=hidden,
                          num_layers=1, batch_first=True, bidirectional=True)
        self.head_rules = nn.Linear(hidden*2, num_rules)
        self.head_valid = nn.Linear(hidden*2, 1)

    def forward(self, X, lengths):
        op = X[:,:,0]
        flags = X[:,:,1:].float()
        e = torch.cat([self.emb_op(op), flags], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head_rules(h_cat), self.head_valid(h_cat)


def load_critic(device=device):
    critic = TinyBiGRU().to(device)
    state = torch.load(critic_path, map_location=device)
    critic.load_state_dict(state)
    critic.eval()
    for p in critic.parameters():
        p.requires_grad = False
    return critic

#reward function
def critic_reward(bits_tensor: torch.Tensor, critic: TinyBiGRU) -> torch.Tensor:
    batch_size = bits_tensor.size(0)
    tokenized = torch.zeros(batch_size, num_lines, 9, dtype=torch.long)
    for i in range(batch_size):
        for j in range(num_lines):
            bits = bits_tensor[i, j].detach().cpu().numpy()
            bstr = ''.join(['1' if v >= 0 else '0' for v in bits])
            w = int(bstr, 2)
            op = (w >> 26) & 0x3F
            rs = (w >> 21) & 0x1F
            rt = (w >> 16) & 0x1F
            rd = (w >> 11) & 0x1F
            sh = (w >>  6) & 0x1F
            fn =  w        & 0x3F
            imm=  w        & 0xFFFF
            def signed16(u): return u if u < 0x8000 else u - 0x10000
            token = [
                op % 64,
                int(op == 0x23),
                int(op == 0x2B),
                int(op in {0x04,0x05,0x06,0x07,0x01}),
                int(rs == 29 or rt == 29),
                int(abs(signed16(imm)) <= 16),
                int(op in {0x02,0x03} or (op == 0 and fn in {0x08, 0x09})),
                int(op == 0x03 or (op == 0 and fn == 0x09)),
                int((op == 0 and fn == 0x08 and rs == 31) or rt == 31 or rd == 31)
            ]
            tokenized[i, j] = torch.tensor(token)
    lengths = torch.full((batch_size,), num_lines, dtype=torch.long)
    _, logit_valid = critic(tokenized.to(device), lengths.to(device))
    return torch.sigmoid(logit_valid).squeeze(1)

#training
def train_gan_with_critic(prefix: str, train_path: str):
    dataset = MIPSDataset(train_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    G = Generator().to(device).train()
    D = Discriminator().to(device).train()
    critic = load_critic()

    g_opt = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()

    start_time = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        for real_batch in dataloader:
            real_batch = real_batch.to(device) * 2 - 1
            batch_size = real_batch.size(0)

            z = torch.randn(batch_size, noise_dim, device=device)
            fake_batch = G(z).detach()
            d_real = D(real_batch)
            d_fake = D(fake_batch)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            d_loss = loss_fn(d_real, real_labels) + loss_fn(d_fake, fake_labels)
            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            d_opt.step()

            z = torch.randn(batch_size, noise_dim, device=device)
            gen_batch = G(z).view(batch_size, num_lines, bit_length)
            p_valid = critic_reward(gen_batch, critic)
            d_out = D(gen_batch).squeeze(1)
            reward_loss = -torch.mean(torch.log(p_valid + 1e-6))
            SEM_WEIGHT = 0.3
            g_loss = loss_fn(d_out, real_labels.squeeze(1)) + SEM_WEIGHT * reward_loss
            g_opt.zero_grad(set_to_none=True)
            g_loss.backward()
            g_opt.step()

        if (epoch+1) % 10 == 0:
            print(f"[{prefix}] Epoch {epoch+1}/{epochs} | D: {d_loss.item():.4f} | G: {g_loss.item():.4f} | avg_p_valid: {p_valid.mean().item():.4f}")

    torch.save(G.state_dict(), f"{prefix}_generator_critic.pth")
    torch.save(D.state_dict(), f"{prefix}_discriminator_critic.pth")
    print(f"[{prefix}] Critic-trained models saved.")

    #result metrics
    end_time = time.time()
    print(f"[{prefix}] Training time: {end_time - start_time:.2f} seconds")
    if device.type == 'cuda':
        current = torch.cuda.memory_allocated(device) / (1024 * 1024)
        peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"[{prefix}] Current GPU memory: {current:.2f} MB")
        print(f"[{prefix}] Peak GPU memory: {peak:.2f} MB")

if __name__ == "__main__":
    train_gan_with_critic("malware", "Dataset\malware_train_1_cleaned.txt")
    train_gan_with_critic("benign",  "Dataset\benign_train_1_cleaned.txt")
