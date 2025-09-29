
import os
import random
from typing import List, Tuple

import torch
import torch.nn as nn

#parameters
model   = "Semantic_Critic\semantic_critic_multirule.pt"
input_files  = ["Dataset\benign_train_1_cleaned.txt", "Dataset\malware_train_1_cleaned.txt"]
snip_length  = 5 #must match training
num_samples = 10  #how many snips to print
rand = True
threshold = 0.5  #rule validity thresh
rule_checker = True

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 1337
random.seed(seed)
torch.manual_seed(seed)

#MIPS commands
op_max   = 64
sp_reg   = 29
ra_reg   = 31
load_ops = {0x23}
store_ops= {0x2B}
br_ops = {0x04, 0x05, 0x06, 0x07}
regm   = 0x01
j_op     = {0x02, 0x03}

#MIPS rules
rules = [
    "store_load",
    "branch_inbounds",
    "stack_balance",
    "base_def_before_use",
    "jump_reg_defined",
    "call_return_pair",
    "word_alignment",
]

reg = [
    "$zero", "$at", "$v0", "$v1",
    "$a0", "$a1", "$a2", "$a3",
    "$t0", "$t1", "$t2", "$t3", "$t4", "$t5", "$t6", "$t7",
    "$s0", "$s1", "$s2", "$s3", "$s4", "$s5", "$s6", "$s7",
    "$t8", "$t9", "$k0", "$k1", "$gp", "$sp", "$fp", "$ra",
]

i_op = {
    0x04: "BEQ", 0x05: "BNE", 0x06: "BLEZ", 0x07: "BGTZ",
    0x08: "ADDI", 0x09: "ADDIU", 0x0A: "SLTI", 0x0B: "SLTIU",
    0x0C: "ANDI", 0x0D: "ORI",  0x0E: "XORI", 0x0F: "LUI",
    0x20: "LB",  0x21: "LH",   0x23: "LW",   0x24: "LBU",
    0x25: "LHU", 0x28: "SB",   0x29: "SH",   0x2B: "SW",
}

j_op_names = {0x02: "J", 0x03: "JAL"}
regm_rt = {0: "BLTZ", 1: "BGEZ", 16: "BLTZAL", 17: "BGEZAL"}
special = {
    0x00: "SLL",  0x02: "SRL",  0x03: "SRA",
    0x04: "SLLV", 0x06: "SRLV", 0x07: "SRAV",
    0x08: "JR",   0x09: "JALR",
    0x0C: "SYSCALL", 0x0D: "BREAK",
    0x10: "MFHI", 0x11: "MTHI", 0x12: "MFLO", 0x13: "MTLO",
    0x18: "MULT", 0x19: "MULTU", 0x1A: "DIV",  0x1B: "DIVU",
    0x20: "ADD",  0x21: "ADDU", 0x22: "SUB",  0x23: "SUBU",
    0x24: "AND",  0x25: "OR",   0x26: "XOR",  0x27: "NOR",
    0x2A: "SLT",  0x2B: "SLTU",
}

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
            if b and is_bin32(b):
                xs.append(b)
    return xs

def windows(lines: List[str], k: int, stride: int=1) -> List[List[str]]:
    out = []
    for i in range(0, len(lines) - k + 1, stride):
        out.append(lines[i:i+k])
    return out

#disasmebler
def disasm(bits: str, idx_in_snippet: int, snippet_len: int) -> str:
    op, rs, rt, rd, sh, fn, imm, tgt = decode(bits)
    s_imm = signed16(imm)

    if op == 0x00:  #R-type case
        m = special.get(fn)
        if m is None:
            return f".word 0x{int(bits,2):08X}"
        if m in {"SLL", "SRL", "SRA"}:
            return f"{m} {reg[rd]}, {reg[rt]}, {sh}"
        if m in {"SLLV", "SRLV", "SRAV"}:
            return f"{m} {reg[rd]}, {reg[rt]}, {reg[rs]}"
        if m == "JR":
            return f"{m} {reg[rs]}"
        if m == "JALR":
            return f"{m} {reg[rd]}, {reg[rs]}"
        if m in {"MFHI", "MFLO"}:
            return f"{m} {reg[rd]}"
        if m in {"MTHI", "MTLO"}:
            return f"{m} {reg[rs]}"
        if m in {"MULT","MULTU","DIV","DIVU"}:
            return f"{m} {reg[rs]}, {reg[rt]}"
        if m in {"ADD","ADDU","SUB","SUBU","AND","OR","XOR","NOR","SLT","SLTU"}:
            return f"{m} {reg[rd]}, {reg[rs]}, {reg[rt]}"
        if m in {"SYSCALL","BREAK"}:
            return m
        return f"{m} ?"

    if op == regm:
        m = regm_rt.get(rt, f"REGIMM({rt})")
        target = idx_in_snippet + 1 + s_imm
        return f"{m} {reg[rs]}, L{target}"

    if op in j_op_names:
        return f"{j_op_names[op]} 0x{tgt:07X}"

    if op in i_op:
        m = i_op[op]
        if m in {"BEQ","BNE"}:
            target = idx_in_snippet + 1 + s_imm
            return f"{m} {reg[rs]}, {reg[rt]}, L{target}"
        if m in {"BLEZ","BGTZ"}:
            target = idx_in_snippet + 1 + s_imm
            return f"{m} {reg[rs]}, L{target}"
        if m in {"ADDI","ADDIU","SLTI","SLTIU","ANDI","ORI","XORI"}:
            return f"{m} {reg[rt]}, {reg[rs]}, {s_imm}"
        if m == "LUI":
            return f"{m} {reg[rt]}, 0x{imm:04X}"
        if m in {"LB","LH","LW","LBU","LHU","SB","SH","SW"}:
            return f"{m} {reg[rt]}, {s_imm}({reg[rs]})"

    return f".word 0x{int(bits,2):08X}"

#tokenizing
def tokenize(bits: str):
    op, rs, rt, rd, sh, fn, imm, tgt = decode(bits)
    is_lw    = int(op in load_ops)
    is_sw    = int(op in store_ops)
    is_br    = int(op in br_ops or op == regm)
    touch_sp = int(rs == sp_reg or rt == sp_reg)
    imm_small= int(abs(signed16(imm)) <= 16)
    is_jump  = int(op in j_op or (op == 0 and fn in {0x08,0x09})) 
    is_call  = int(op == 0x03 or (op == 0 and fn == 0x09))        
    uses_ra  = int((op == 0 and fn == 0x08 and rs == ra_reg) or rt == ra_reg or rd == ra_reg)
    return (op % op_max, is_lw, is_sw, is_br, touch_sp, imm_small, is_jump, is_call, uses_ra)

def collate_token_seqs(token_seqs: List[List[Tuple[int, ...]]]):
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

#rule checkers
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
        if op in br_ops or op == regm:
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
                pops = k // 4
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
        if op in {0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F} and rt != 0: defined.add(rt)
        if op in load_ops and rt != 0: defined.add(rt)
        if op in {0x08,0x09} and rs == sp_reg and rt == sp_reg: defined.add(sp_reg)
    return 0

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

rule_checkers = [
    rule_store_load_violation,
    rule_branch_inbounds_violation,
    rule_stack_balance_violation,
    rule_base_def_before_use_violation,
    rule_jump_reg_defined_violation,
    rule_call_return_pair_violation,
    rule_word_alignment_violation,
]

#LSTM model
class TinyBiGRU(nn.Module):
    def __init__(self, emb_dim=32, hidden=128, num_rules: int = 7):
        super().__init__()
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
        return self.head_rules(h_cat), self.head_valid(h_cat)

def load_model():
    model = TinyBiGRU(num_rules=len(rules)).to(device)
    state = torch.load(model, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

#printing
def fmt_prob(p: float) -> str:
    return f"{p*100:.1f}%"

def print_report_for_file(model, path: str, k_samples: int):
    print(f"\n{os.path.basename(path)}")
    lines = load_lines(path)
    snips = windows(lines, snip_length, 1)
    if not snips:
        print("No snippets found.")
        return
    idxs = list(range(len(snips)))
    if rand:
        random.shuffle(idxs)
    idxs = idxs[:k_samples]

    token_seqs = [[tokenize(b) for b in snips[i]] for i in idxs]
    X, lengths = collate_token_seqs(token_seqs)
    X, lengths = X.to(device), lengths.to(device)

    with torch.no_grad():
        logits_rules, logit_valid = model(X, lengths)
        probs_rules = torch.sigmoid(logits_rules).cpu().numpy()
        probs_valid = torch.sigmoid(logit_valid).cpu().numpy()

    for row, i in enumerate(idxs):
        snippet = snips[i]
        model_valid = probs_valid[row, 0]
        pred_valid  = model_valid >= threshold
        print(f"\nSnippet #{i} (lines {i}..{i+snip_length-1})")
        print(f"  Model: validity = {fmt_prob(model_valid)}  => {'PASS' if pred_valid else 'FAIL'}")
        for r, name in enumerate(rules):
            p_vi = probs_rules[row, r]
            pred_violate = p_vi >= threshold
            extra = ""
            if rule_checker:
                gt_vi = rule_checkers[r](snippet)
                extra = f" | rule_checker: {'VIOL' if gt_vi==1 else 'OK '}"
            print(f"    - {name:20s} p(violation)={fmt_prob(p_vi)} - {'VIOL' if pred_violate else 'OK  '}{extra}")

        #decoded assembly
        for j, bits in enumerate(snippet):
            asm = disasm(bits, j, snip_length)
            print(f"      [{i+j:04d}] {bits}    ; {asm}")

def main():
    print(f"Device: {device}")
    model = load_model()
    for path in input_files:
        if os.path.exists(path):
            print_report_for_file(model, path, num_samples)
        else:
            print(f"(skip) file not found: {path}")

if __name__ == "__main__":
    main()
