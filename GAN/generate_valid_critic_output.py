import os
import torch
import numpy as np
from GAN_model import Generator
from GAN_model_backprop import TinyBiGRU, critic_reward, NUM_LINES, BIT_LENGTH, NOISE_DIM, DEVICE

#parameters
lines_target = 27000
batch_size = 512
critic_model = "Semantic_Critic\semantic_critic_multirule.pt"
out_files = {
    "benign": "generated_benign_semvalid.txt",
    "malware": "generated_malware_semvalid.txt"
}
sem_thresh = {
    "benign": 0.104,
    "malware": 0.380
}

#load models
def load_generator(prefix):
    model = Generator().to(DEVICE)
    model.load_state_dict(torch.load(f"GAN\{prefix}_generator_critic.pth", map_location=DEVICE))
    model.eval()
    return model

def load_critic():
    model = TinyBiGRU().to(DEVICE)
    model.load_state_dict(torch.load(critic_model, map_location=DEVICE))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

#convert to strings
def tensor_to_binary_lines(tensor_batch):
    output_lines = []
    for stack in tensor_batch:
        for line in stack:
            line_bin = ''.join(['1' if b >= 0 else '0' for b in line])
            output_lines.append(line_bin)
    return output_lines

#main
@torch.no_grad()
def generate_semantic_valid_lines(generator, critic, label, sem_thresh):
    collected = []
    attempts = 0

    while len(collected) < lines_target:
        z = torch.randn(batch_size, NOISE_DIM, device=DEVICE)
        fake_batch = generator(z)
        p_valid = critic_reward(fake_batch, critic)
        pass_mask = p_valid >= sem_thresh
        passed_stacks = fake_batch[pass_mask]

        if passed_stacks.size(0) > 0:
            new_lines = tensor_to_binary_lines(passed_stacks)
            collected.extend(new_lines[:lines_target - len(collected)])

        attempts += 1
        if attempts % 10 == 0:
            print(f"[{label}] Attempts: {attempts}, Collected: {len(collected)}/{lines_target}")

    #saving output
    with open(out_files[label], "w") as f:
        for line in collected:
            f.write(line + "\n")
    print(f"[{label}] Saved {lines_target} lines to {out_files[label]}")


if __name__ == "__main__":
    print("Loading Semantic Critic")
    critic = load_critic()

    for label in ["benign", "malware"]:
        print(f"\nGenerating {label.upper()} Code (thresh={sem_thresh[label]:.3f})")
        generator = load_generator(label)
        generate_semantic_valid_lines(generator, critic, label, sem_thresh[label])
