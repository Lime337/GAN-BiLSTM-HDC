import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn

#parameters
output_prefix = {"malware"}
NUM_Generation = 500

num_lines = 5
bit_length = 32
noise_dim = 100
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_variant = "critic"
critic_path = "semantic_critic_multirule.pt"

#helper functions
def load_txt_bits(path):
    bits = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s and set(s) <= {"0", "1"}:
                bits.extend(1 if c == "1" else 0 for c in s)
    return np.array(bits, dtype=np.float32)

def generate_fake_text(generator_model, latent_dim, output_path,
                       num_lines=5, bit_length=32, activation='tanh'):
    generator_model.eval()
    with torch.no_grad():
        z = torch.randn(1, latent_dim, device=next(generator_model.parameters()).device)
        out = generator_model(z).detach().cpu().numpy().flatten()

    if activation == 'sigmoid':
        bits = (out > 0.5).astype(np.uint8)
    elif activation == 'tanh':
        bits = ((out + 1) / 2 > 0.5).astype(np.uint8)
    else:
        raise ValueError("activation must be 'sigmoid' or 'tanh'")

    expected = num_lines * bit_length
    if len(bits) < expected:
        bits = np.pad(bits, (0, expected - len(bits)), constant_values=0)
    else:
        bits = bits[:expected]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for i in range(num_lines):
            line_bits = bits[i*bit_length:(i+1)*bit_length]
            f.write(''.join(map(str, line_bits)) + "\n")

    print(f"[+] Wrote {num_lines}×{bit_length} bits to {output_path}")

#discriminator inference
def check_with_discriminator(gen_flat_bits, discriminator_model, device="cpu"):
    x = torch.tensor(gen_flat_bits, dtype=torch.float32, device=device).view(1, num_lines, bit_length)
    x = x * 2.0 - 1.0
    with torch.no_grad():
        logits = discriminator_model(x)
        prob = torch.sigmoid(logits).item()
    return prob

#token creator
op_max    = 64
sp_reg    = 29
ra_reg    = 31
load_ops  = {0x23}
store_ops = {0x2B}
br_i_ops  = {0x04, 0x05, 0x06, 0x07}
regm      = 0x01
j_op      = {0x02, 0x03}

def decode(bits_str: str):
    w = int(bits_str, 2)
    op      = (w >> 26) & 0x3F
    rs      = (w >> 21) & 0x1F
    rt      = (w >> 16) & 0x1F
    rd      = (w >> 11) & 0x1F
    shamt   = (w >>  6) & 0x1F
    funct   =  w        & 0x3F
    imm16   =  w        & 0xFFFF
    target26=  w        & 0x03FFFFFF
    return op, rs, rt, rd, shamt, funct, imm16, target26

def signed16(u: int): return u if u < 0x8000 else u - 0x10000

def tokenize_for_critic(bits_str: str):
    op, rs, rt, rd, sh, fn, imm, tgt = decode(bits_str)
    return (
        op % op_max,
        int(op in load_ops),
        int(op in store_ops),
        int(op in br_i_ops or op == regm),
        int(rs == sp_reg or rt == sp_reg),
        int(abs(signed16(imm)) <= 16),
        int(op in j_op or (op == 0 and fn in {0x08, 0x09})),
        int(op == 0x03 or (op == 0 and fn == 0x09)),
        int((op == 0 and fn == 0x08 and rs == ra_reg) or rt == ra_reg or rd == ra_reg)
    )

def collate_token_seqs(token_seqs):
    lengths = torch.tensor([len(s) for s in token_seqs], dtype=torch.long)
    T = max(lengths).item()
    padded = []
    for s in token_seqs:
        t = torch.tensor(s, dtype=torch.long)
        if t.shape[0] < T:
            pad = torch.zeros((T - t.shape[0], 9), dtype=torch.long)
            t = torch.cat([t, pad], dim=0)
        padded.append(t)
    X = torch.stack(padded, dim=0)
    return X, lengths

class TinyBiGRU(nn.Module):
    def __init__(self, emb_dim=32, hidden=128, num_rules=7):
        super().__init__()
        self.emb_op = nn.Embedding(op_max, emb_dim)
        self.rnn = nn.GRU(input_size=emb_dim + 8, hidden_size=hidden,
                          num_layers=1, batch_first=True, bidirectional=True)
        self.head_rules = nn.Linear(hidden * 2, num_rules)
        self.head_valid = nn.Linear(hidden * 2, 1)
    def forward(self, X, lengths):
        op = X[:, :, 0]
        flags = X[:, :, 1:].float()
        e = torch.cat([self.emb_op(op), flags], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head_rules(h_cat), self.head_valid(h_cat)

def load_frozen_critic(device=device):
    critic = TinyBiGRU().to(device)
    state = torch.load(critic_path, map_location=device)
    critic.load_state_dict(state)
    critic.eval()
    for p in critic.parameters():
        p.requires_grad = False
    return critic

@torch.no_grad()
def critic_p_valid_for_file(path, critic):
    lines = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if len(s) == 32 and set(s) <= {"0", "1"}:
                lines.append(s)
    if not lines:
        print(f"No valid 32-bit binary lines found in {path}")
        return 0.0
    token_seqs = [[tokenize_for_critic(b) for b in lines]]
    X, lengths = collate_token_seqs(token_seqs)
    X, lengths = X.to(device), lengths.to(device)
    _, logit_valid = critic(X, lengths)
    return torch.sigmoid(logit_valid).item()

def compute_avg_training_p_valid(train_files, critic):
    scores = []
    for path in train_files:
        score = critic_p_valid_for_file(path, critic)
        if score > 0.0:
            scores.append(score)
        else:
            print(f"p_valid=0.0 for {path} — possibly malformed or invalid.")
    if not scores:
        print("No valid semantic critic scores from training set — defaulting thresh=0.5")
        return 0.5
    return np.mean(scores)

#comparison
def compare_samples(train_files, gen_files, discriminator, critic=None, sem_thresh=0.5):
    train_vecs = [load_txt_bits(p) for p in train_files]
    gen_vecs   = [load_txt_bits(p) for p in gen_files]
    results = []
    for gi, g in enumerate(gen_vecs):
        best_dot, best_cos = -1, -1
        best_dot_file, best_cos_file = None, None
        for ti, t in enumerate(train_vecs):
            L = max(len(g), len(t))
            gg = np.pad(g, (0, L - len(g)), constant_values=0)
            tt = np.pad(t, (0, L - len(t)), constant_values=0)
            dot = float(np.dot(gg, tt))
            cos = float(cosine_similarity([gg], [tt])[0][0])
            if dot > best_dot:
                best_dot, best_dot_file = dot, train_files[ti]
            if cos > best_cos:
                best_cos, best_cos_file = cos, train_files[ti]
        p_real = check_with_discriminator(g, discriminator, device=device)
        sem_p = None
        if critic is not None:
            sem_p = critic_p_valid_for_file(gen_files[gi], critic)
        results.append((gen_files[gi], best_dot, best_dot_file, best_cos, best_cos_file, p_real, sem_p))
    return results

#main function
if __name__ == "__main__":
    for prefix in output_prefix:
        training_set = [f"{prefix}_train_1.txt"]
        from GAN.GAN_model import Generator, Discriminator
        G = Generator().to(device).eval()
        D = Discriminator().to(device).eval()
        gen_w = f"{prefix}_generator_critic.pth" if model_variant == "critic" else f"{prefix}_generator.pth"
        dis_w = f"{prefix}_discriminator_critic.pth" if model_variant == "critic" else f"{prefix}_discriminator.pth"
        G.load_state_dict(torch.load(gen_w, map_location=device))
        D.load_state_dict(torch.load(dis_w, map_location=device))
        critic = load_frozen_critic(device) if os.path.exists(critic_path) else None

        thresh = compute_avg_training_p_valid(training_set, critic) if critic else 0.5
        print(f"[Dynamic thresh] Set to average training p_valid = {thresh:.3f}")

        gen_paths = []
        for i in range(NUM_Generation):
            out_path = f"output/{prefix}_{model_variant}_{i+1}.txt"
            generate_fake_text(G, latent_dim=noise_dim, output_path=out_path,
                               num_lines=num_lines, bit_length=bit_length, activation='tanh')
            gen_paths.append(out_path)

        results = compare_samples(training_set, gen_paths, D, critic=critic, sem_thresh=thresh)
        realcount = fakecount = numvalidlines = sem_pass = sem_skipped = 0
        sem_scores = []
        for r in results:
            path, best_dot, best_dot_file, best_cos, best_cos_file, p_real, p_sem = r
            verdict = "REAL" if p_real >= 0.5 else "FAKE"
            if verdict == "REAL": realcount += 1
            else: fakecount += 1
            print(f"Generated: {path}")
            print(f"[Discriminator] p(real)={p_real:.4f} → {verdict}")
            from Assembler.verify_instruction_binary import convert_file_to_mips
            vlines = convert_file_to_mips(path)
            numvalidlines += vlines
            if p_sem is not None and vlines == num_lines:
                sem_scores.append(p_sem)
                if p_sem >= thresh:
                    sem_pass += 1
                print(f"[SemanticCritic] p_valid={p_sem:.4f} (pass≥{thresh:.2f} → {'PASS' if p_sem>=thresh else 'FAIL'})")
            elif p_sem is not None:
                sem_skipped += 1
                print(f"[SemanticCritic] SKIPPED: Only {vlines}/{num_lines} valid MIPS lines.")
            print()

        total = realcount + fakecount if (realcount + fakecount) > 0 else 1
        print(f"{prefix}/{model_variant} discriminator REAL%: {(realcount/total)*100:.2f}")
        print(f"{prefix}/{model_variant} valid MIPS%: {(numvalidlines / max(1, NUM_Generation * num_lines))*100:.2f}%")
        if sem_scores:
            print(f"{prefix}/{model_variant} semantic PASS%: {(sem_pass/len(sem_scores))*100:.2f}% | avg p_valid={np.mean(sem_scores):.3f}")
        print(f"{prefix}/{model_variant} semantic SKIPPED (invalid MIPS): {sem_skipped}")
