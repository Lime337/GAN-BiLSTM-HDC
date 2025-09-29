import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#parameters
benign_file  = "Dataset\benign_train_1_cleaned.txt"
malware_file = "Dataset\malware_train_1_cleaned.txt"

snip_length      = 5
sliding_stride   = 1
train_frac       = 0.8
neg_pos_weight   = 1.0
batch            = 128
epochs           = 8
lr               = 2e-3
seed             = 1337
device           = "cuda" if torch.cuda.is_available() else "cpu"

#rule toggles
use_rules = {
    "store_load": True,
    "branch_inbounds": True,
    "stack_balance": True,
    "base_def_before_use": True,
    "jump_reg_defined": True,
    "call_return_pair": True,
    "word_alignment": True,
}

random.seed(seed)
torch.manual_seed(seed)

#MIPS commands
op_max   = 64
load_ops = {0x23}
store_ops= {0x2B}
br_i_ops = {0x04, 0x05, 0x06, 0x07}
regm   = 0x01
j_op     = {0x02, 0x03}
sp_reg   = 29
ra_reg   = 31

def is_bin32(s: str) -> bool:
    return len(s) == 32 and set(s) <= {"0","1"}

def decode(bits: str) -> Tuple[int,int,int,int,int,int,int,int]:
    w = int(bits, 2)
    op      = (w >> 26) & 0x3F
    rs      = (w >> 21) & 0x1F
    rt      = (w >> 16) & 0x1F
    rd      = (w >> 11) & 0x1F
    shamt   = (w >>  6) & 0x1F
    funct   =  w        & 0x3F
    imm16   =  w        & 0xFFFF
    target26=  w        & 0x03FFFFFF
    return op, rs, rt, rd, shamt, funct, imm16, target26

def signed16(u: int) -> int:
    return u if u < 0x8000 else u - 0x10000

def load_lines(path: str) -> List[str]:
    xs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            b = line.strip()
            if b and is_bin32(b): xs.append(b)
    return xs

def windows(lines: List[str], k: int, stride: int) -> List[List[str]]:
    out = []
    for i in range(0, len(lines) - k + 1, stride):
        out.append(lines[i:i+k])
    return out

#rule checks
def rule_store_load_violation(snip: List[str]) -> int:
    stores = {}
    for i, b in enumerate(snip):
        op, rs, rt, rd, sh, fn, imm, tgt = decode(b)
        if op in store_ops:
            stores.setdefault((rs, imm), []).append(i)
        if op in load_ops:
            if not any(idx < i for idx in stores.get((rs, imm), [])):
                return 1
    return 0

def rule_branch_inbounds_violation(snip: List[str]) -> int:
    T = len(snip)
    for i, b in enumerate(snip):
        op, rs, rt, rd, sh, fn, imm, tgt = decode(b)
        if op in br_i_ops or op == regm:
            target = i + 1 + signed16(imm)
            if target < 0 or target >= T:
                return 1
    return 0

def rule_stack_balance_violation(snip: List[str]) -> int:
    depth = 0
    for b in snip:
        op, rs, rt, rd, sh, fn, imm, tgt = decode(b)
        if op in {0x08, 0x09} and rs == sp_reg and rt == sp_reg:
            k = signed16(imm)
            if k % 4 != 0:
                continue
            if k < 0: depth += (-k // 4)
            elif k > 0:
                pops = (k // 4)
                if pops > depth:
                    return 1
                depth -= pops
    return 1 if depth != 0 else 0

def rule_base_def_before_use_violation(snip: List[str]) -> int:
    defined = {0}
    for i, b in enumerate(snip):
        op, rs, rt, rd, sh, fn, imm, tgt = decode(b)
        if op in load_ops | store_ops:
            if rs not in defined and rs != sp_reg:
                return 1
        if op == 0 and rd != 0: defined.add(rd)
        elif op in {0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F}:
            if rt != 0: defined.add(rt)
        elif op in load_ops:
            if rt != 0: defined.add(rt)
        if op in {0x08,0x09} and rs == sp_reg and rt == sp_reg:
            defined.add(sp_reg)
    return 0

#extra rules
def rule_jump_reg_defined_violation(snip: List[str]) -> int:
    defined = {0}
    for i, b in enumerate(snip):
        op, rs, rt, rd, sh, fn, imm, tgt = decode(b)
        if op == 0 and rd != 0: defined.add(rd)
        if op in {0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F} and rt != 0: defined.add(rt)
        if op in load_ops and rt != 0: defined.add(rt)
        if op in {0x08,0x09} and rs == sp_reg and rt == sp_reg: defined.add(sp_reg)
        if op == 0 and fn in {0x08, 0x09}:
            if rs not in defined:
                return 1
    return 0

def rule_call_return_pair_violation(snip: List[str]) -> int:
    call_idx = None
    jr_idx   = None
    for i, b in enumerate(snip):
        op, rs, rt, rd, sh, fn, imm, tgt = decode(b)
        if call_idx is None and ((op == 0x03) or (op == 0 and fn == 0x09)):
            call_idx = i
        if jr_idx is None and (op == 0 and fn == 0x08 and rs == ra_reg):
            jr_idx = i
    if call_idx is None and jr_idx is None:
        return 0
    if call_idx is not None and (jr_idx is None or jr_idx <= call_idx):
        return 1
    if jr_idx is not None and (call_idx is None or call_idx >= jr_idx):
        return 1
    return 0

def rule_word_alignment_violation(snip: List[str]) -> int:
    for b in snip:
        op, rs, rt, rd, sh, fn, imm, tgt = decode(b)
        if op in load_ops | store_ops:
            if (signed16(imm) % 4) != 0:
                return 1
    return 0

#combine rules
rule_funcs = [
    ("store_load",             rule_store_load_violation),
    ("branch_inbounds",        rule_branch_inbounds_violation),
    ("stack_balance",          rule_stack_balance_violation),
    ("base_def_before_use",    rule_base_def_before_use_violation),
    ("jump_reg_defined",       rule_jump_reg_defined_violation),
    ("call_return_pair",       rule_call_return_pair_violation),
    ("word_alignment",         rule_word_alignment_violation),
]
active_rules = [name for name, _ in rule_funcs if use_rules[name]]
num_rules    = len(active_rules)

def violations_vector(snip: List[str]) -> List[int]:
    vec = []
    for name, fn in rule_funcs:
        if use_rules[name]:
            vec.append(fn(snip))
    return vec

#corruption targets
def corrupt_store_load(snip: List[str]) -> List[str]:
    s = snip[:]
    lw_idx = None; lw_rs = None; lw_imm = None
    for i,b in enumerate(s):
        op, rs, rt, rd, sh, fn, imm, _ = decode(b)
        if op in load_ops:
            lw_idx, lw_rs, lw_imm = i, rs, imm
            break
    if lw_idx is None: return s
    sw_idx = None
    for i,b in enumerate(s[:lw_idx]):
        op, rs, *_ , imm, _ = decode(b)
        if op in store_ops and rs == lw_rs and imm == lw_imm:
            sw_idx = i; break
    if sw_idx is None:
        w = int(s[lw_idx], 2)
        new_rs = (lw_rs + 3) & 0x1F
        w = (w & ~(0x1F << 21)) | ((new_rs & 0x1F) << 21)
        s[lw_idx] = format(w & 0xFFFFFFFF, "032b")
        return s
    mode = random.random()
    if mode < 0.34:
        del s[sw_idx]; s.append(s[-1])
    elif mode < 0.67:
        inst = s.pop(sw_idx)
        if sw_idx < lw_idx: lw_idx -= 1
        s.insert(min(lw_idx+1, len(s)), inst)
    else:
        w = int(s[sw_idx], 2)
        if random.random() < 0.5:
            imm = (lw_imm + (4 if random.random()<0.5 else -4)) & 0xFFFF
            w = (w & ~0xFFFF) | imm
        else:
            new_rs = (lw_rs + random.randint(1,3)) & 0x1F
            w = (w & ~(0x1F << 21)) | ((new_rs & 0x1F) << 21)
        s[sw_idx] = format(w & 0xFFFFFFFF, "032b")
    return s

def corrupt_branch_out_of_bounds(snip: List[str]) -> List[str]:
    s = snip[:]; T = len(s)
    br_idx = None
    for i,b in enumerate(s):
        op, rs, rt, rd, sh, fn, imm, _ = decode(b)
        if op in br_i_ops or op == regm:
            br_idx = i; break
    if br_idx is None: return s
    w = int(s[br_idx], 2)
    new_imm = ( (T + 2) & 0xFFFF ) if random.random() < 0.5 else ( (-T - 2) + 0x10000 ) & 0xFFFF
    w = (w & ~0xFFFF) | new_imm
    s[br_idx] = format(w & 0xFFFFFFFF, "032b")
    return s

def corrupt_stack_imbalance(snip: List[str]) -> List[str]:
    s = snip[:]
    for i,b in enumerate(s):
        op, rs, rt, rd, sh, fn, imm, _ = decode(b)
        if op in {0x08,0x09} and rs==sp_reg and rt==sp_reg and signed16(imm) <= -4 and (signed16(imm) % 4 == 0):
            s.insert(i, b)
            if len(s) > len(snip): s.pop()
            return s
    for i,b in enumerate(s):
        op, rs, rt, rd, sh, fn, imm, _ = decode(b)
        if op in {0x08,0x09} and rs==sp_reg and rt==sp_reg and signed16(imm) >= 4 and (signed16(imm) % 4 == 0):
            del s[i]; s.append(s[-1])
            return s
    return s

def corrupt_base_def_use(snip: List[str]) -> List[str]:
    s = snip[:]
    defined = {0}
    for i,b in enumerate(s):
        op, rs, rt, rd, sh, fn, imm, _ = decode(b)
        if op in load_ops | store_ops and rs in defined and rs != sp_reg:
            new_rs = random.choice([r for r in range(1,32) if r not in defined and r != sp_reg])
            w = int(b, 2)
            w = (w & ~(0x1F << 21)) | ((new_rs & 0x1F) << 21)
            s[i] = format(w & 0xFFFFFFFF, "032b")
            return s
        if op in {0x08,0x09} and rs==sp_reg and rt==sp_reg: defined.add(sp_reg)
        if op == 0 and rd != 0: defined.add(rd)
        if op in {0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F} and rt != 0: defined.add(rt)
        if op in load_ops and rt != 0: defined.add(rt)
    return s

def corrupt_jump_reg_defined(snip: List[str]) -> List[str]:
    s = snip[:]
    defined = {0}
    jr_idx = None; jr_fn = None; jr_rs = None
    for i,b in enumerate(s):
        op, rs, rt, rd, sh, fn, imm, _ = decode(b)
        if op == 0 and rd != 0: defined.add(rd)
        if op in {0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F} and rt != 0: defined.add(rt)
        if op in load_ops and rt != 0: defined.add(rt)
        if op in {0x08,0x09} and rs==sp_reg and rt==sp_reg: defined.add(sp_reg)
        if op == 0 and fn in {0x08,0x09} and jr_idx is None:
            jr_idx, jr_fn, jr_rs = i, fn, rs
    if jr_idx is None: return s
    w = int(s[jr_idx], 2)
    undef = [r for r in range(1,32) if r not in defined]
    if not undef: undef = [2]
    new_rs = random.choice(undef)
    w = (w & ~(0x1F << 21)) | ((new_rs & 0x1F) << 21)
    s[jr_idx] = format(w & 0xFFFFFFFF, "032b")
    return s

def corrupt_call_return_pair(snip: List[str]) -> List[str]:
    s = snip[:]
    call_idx = None
    jr_idx   = None
    for i,b in enumerate(s):
        op, rs, rt, rd, sh, fn, imm, _ = decode(b)
        if call_idx is None and ((op == 0x03) or (op == 0 and fn == 0x09)):
            call_idx = i
        if jr_idx is None and (op == 0 and fn == 0x08 and rs == ra_reg):
            jr_idx = i
    if call_idx is not None and jr_idx is not None and jr_idx > call_idx:
        inst = s.pop(jr_idx)
        s.insert(max(0, call_idx-1), inst)
        return s
    for i,b in enumerate(s):
        op, rs, rt, rd, sh, fn, imm, _ = decode(b)
        if op == 0x02:
            w = int(b, 2)
            w = (w & ~(0x3F << 26)) | (0x03 << 26)
            s[i] = format(w & 0xFFFFFFFF, "032b")
            return s
    return s

def corrupt_word_alignment(snip: List[str]) -> List[str]:
    s = snip[:]
    for i,b in enumerate(s):
        op, rs, rt, rd, sh, fn, imm, _ = decode(b)
        if op in load_ops | store_ops:
            delta = 2 if random.random() < 0.5 else -2
            new_imm = ( (signed16(imm) + delta) & 0xFFFF )
            w = int(b, 2)
            w = (w & ~0xFFFF) | new_imm
            s[i] = format(w & 0xFFFFFFFF, "032b")
            return s
    return s

corrupters = {
    "store_load":          corrupt_store_load,
    "branch_inbounds":     corrupt_branch_out_of_bounds,
    "stack_balance":       corrupt_stack_imbalance,
    "base_def_before_use": corrupt_base_def_use,
    "jump_reg_defined":    corrupt_jump_reg_defined,
    "call_return_pair":    corrupt_call_return_pair,
    "word_alignment":      corrupt_word_alignment,
}

#token making
def tokenize(bits: str) -> Tuple[int, ...]:
    op, rs, rt, rd, sh, fn, imm, tgt = decode(bits)
    is_lw    = int(op in load_ops)
    is_sw    = int(op in store_ops)
    is_br    = int(op in br_i_ops or op == regm)
    touch_sp = int(rs == sp_reg or rt == sp_reg)
    imm_small= int(abs(signed16(imm)) <= 16)
    is_jump  = int(op in j_op or (op == 0 and fn in {0x08,0x09}))
    is_call  = int(op == 0x03 or (op == 0 and fn == 0x09))
    uses_ra  = int((op == 0 and fn == 0x08 and rs == ra_reg) or rt == ra_reg or rd == ra_reg)
    return (op % op_max, is_lw, is_sw, is_br, touch_sp, imm_small, is_jump, is_call, uses_ra)

#dataset
@dataclass
class Sample:
    toks: List[Tuple[int, ...]]
    rule_vec: List[int]

class MultiRuleDataset(Dataset):
    def __init__(self, pos_snips: List[List[str]], neg_per_pos: float = 1.0):
        self.samples: List[Sample] = []
        for sn in pos_snips:
            self.samples.append(Sample([tokenize(b) for b in sn], violations_vector(sn)))
        target_neg = int(len(pos_snips) * neg_per_pos)
        active = [r for r in active_rules]
        for _ in range(target_neg):
            base = random.choice(pos_snips)
            r = random.choice(active)
            bad = corrupters[r](base)
            self.samples.append(Sample([tokenize(b) for b in bad], violations_vector(bad)))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        s = self.samples[i]
        X = torch.tensor(s.toks, dtype=torch.long)
        y = torch.tensor(s.rule_vec, dtype=torch.float32).unsqueeze(0)
        return X, y

def collate(batch):
    seqs, ys = zip(*batch)
    lengths = torch.tensor([t.shape[0] for t in seqs], dtype=torch.long)
    T = max(lengths).item()
    padded = []
    for t in seqs:
        pad = torch.zeros((T - t.shape[0], t.shape[1]), dtype=torch.long)
        padded.append(torch.cat([t, pad], dim=0))
    X = torch.stack(padded, dim=0)
    Y = torch.cat(ys, dim=0)
    return X, lengths, Y

#BiGRU model
class TinyBiGRU(nn.Module):
    def __init__(self, emb_dim=32, hidden=128, num_rules: int = 7):
        super().__init__()
        self.num_rules = num_rules
        self.emb_op = nn.Embedding(op_max, emb_dim)
        self.rnn = nn.GRU(input_size=emb_dim + 8, hidden_size=hidden,
                          num_layers=1, batch_first=True, bidirectional=True)
        self.head_rules = nn.Linear(hidden*2, num_rules)
        self.head_valid = nn.Linear(hidden*2, 1)

    def forward(self, X: torch.LongTensor, lengths: torch.LongTensor):
        op = X[:,:,0]
        flags = X[:,:,1:].float()
        e = torch.cat([self.emb_op(op), flags], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        logits_rules = self.head_rules(h_cat)
        logit_valid  = self.head_valid(h_cat)
        return logits_rules, logit_valid

#data building
def build():
    benign  = load_lines(benign_file)
    malware = load_lines(malware_file)
    all_lines = benign + malware

    cut = int(len(all_lines) * train_frac)
    train_lines, val_lines = all_lines[:cut], all_lines[cut:]

    pos_train = windows(train_lines, snip_length, sliding_stride)
    pos_val   = windows(val_lines,   snip_length, sliding_stride)

    if not pos_train or not pos_val:
        raise RuntimeError("Not enough snippets; reduce snip_length or provide more data.")

    train_set = MultiRuleDataset(pos_train, neg_per_pos=neg_pos_weight)
    val_set   = MultiRuleDataset(pos_val,   neg_per_pos=neg_pos_weight)
    return train_set, val_set

#training
def train():
    train_set, val_set = build()
    tr_loader = DataLoader(train_set, batch_size=batch, shuffle=True,  collate_fn=collate)
    va_loader = DataLoader(val_set,   batch_size=batch, shuffle=False, collate_fn=collate)

    model = TinyBiGRU(num_rules=num_rules).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr)
    bce   = nn.BCEWithLogitsLoss()

    def run(loader, train_mode=True):
        model.train(train_mode)
        tot_loss = 0.0; n = 0
        rule_correct = 0; rule_total = 0
        valid_correct = 0; valid_total = 0
        with torch.set_grad_enabled(train_mode):
            for X, lengths, Y in loader:
                X, lengths, Y = X.to(device), lengths.to(device), Y.to(device)
                logits_rules, logit_valid = model(X, lengths)
                target_valid = (Y.sum(dim=1, keepdim=True) == 0).float()
                loss = bce(logits_rules, Y) + 0.5 * bce(logit_valid, target_valid)
                if train_mode:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                tot_loss += loss.item() * X.size(0); n += X.size(0)

                preds_rules = (torch.sigmoid(logits_rules) >= 0.5).float()
                rule_correct += (preds_rules == Y).sum().item()
                rule_total   += Y.numel()
                preds_valid = (torch.sigmoid(logit_valid) >= 0.5).float()
                valid_correct += (preds_valid == target_valid).sum().item()
                valid_total   += target_valid.numel()

        return (tot_loss / max(1,n),
                rule_correct / max(1,rule_total),
                valid_correct / max(1,valid_total))

    print(f"Device: {device} | Active rules: {active_rules}")
    for ep in range(1, epochs+1):
        tr_loss, tr_rule_acc, tr_valid_acc = run(tr_loader, True)
        va_loss, va_rule_acc, va_valid_acc = run(va_loader, False)
        print(f"Epoch {ep:02d} | loss {tr_loss:.4f}  rule_acc {tr_rule_acc:.3f}  valid_acc {tr_valid_acc:.3f} "
              f"| val_loss {va_loss:.4f}  val_rule_acc {va_rule_acc:.3f}  val_valid_acc {va_valid_acc:.3f}")

    torch.save(model.state_dict(), "semantic_critic_multirule.pt")
    print("Saved: semantic_critic_multirule.pt")

if __name__ == "__main__":
    train()