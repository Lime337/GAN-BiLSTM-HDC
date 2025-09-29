registers = {
    0: "$zero", 1: "$at", 2: "$v0", 3: "$v1",
    4: "$a0", 5: "$a1", 6: "$a2", 7: "$a3",
    8: "$t0", 9: "$t1", 10: "$t2", 11: "$t3",
    12: "$t4", 13: "$t5", 14: "$t6", 15: "$t7",
    16: "$s0", 17: "$s1", 18: "$s2", 19: "$s3",
    20: "$s4", 21: "$s5", 22: "$s6", 23: "$s7",
    24: "$t8", 25: "$t9", 28: "$gp", 29: "$sp",
    30: "$fp", 31: "$ra"
}


def binary_to_mips(binary_code):
    if len(binary_code) != 32 or not set(binary_code).issubset({'0', '1'}):
        return "invalid MIPS code"

    try:
        instruction_int = int(binary_code, 2)
    except ValueError:
        return "invalid MIPS code"

    opcode = (instruction_int >> 26) & 0x3F



    #R-Type
    if opcode == 0:
        rs = (instruction_int >> 21) & 0x1F
        rt = (instruction_int >> 16) & 0x1F
        rd = (instruction_int >> 11) & 0x1F
        shamt = (instruction_int >> 6) & 0x1F
        func_code = instruction_int & 0x3F

        r_funcs = {
            32: "add", 33: "addu", 34: "sub", 35: "subu",
            36: "and", 37: "or", 38: "xor", 39: "nor",
            42: "slt", 43: "sltu",
            0: "sll", 2: "srl", 3: "sra",
            8: "jr", 9: "jalr", 12: "syscall"
        }

        if func_code in r_funcs:
            instr = r_funcs[func_code]
            if instr in ["sll", "srl", "sra"]:
                return f"{instr} {safe_reg(rd)}, {safe_reg(rt)}, {shamt}"
            elif instr == "jr":
                return f"{instr} {safe_reg(rs)}"
            elif instr == "jalr":
                if(rd == 0): 
                    rd = 31
                return f"{instr} {safe_reg(rd)}{safe_reg(rs)}"
            elif instr == "syscall":
                return instr
            else:
                return f"{instr} {safe_reg(rd)}, {safe_reg(rs)}, {safe_reg(rt)}"
        else:
            return "invalid MIPS code"

    #I-Type
    i_instructions = {
        8: "addi", 9: "addiu",
        12: "andi", 13: "ori", 14: "xori",
        10: "slti", 11: "sltiu",
        35: "lw", 43: "sw",
        4: "beq", 5: "bne",
        15: "lui"
    }

    if opcode in i_instructions:
        rs = (instruction_int >> 21) & 0x1F
        rt = (instruction_int >> 16) & 0x1F
        imm = instruction_int & 0xFFFF
        if imm & 0x8000:
            imm = -((~imm + 1) & 0xFFFF)

        instr = i_instructions[opcode]
        if instr in ["addi", "addiu", "andi", "ori", "xori", "slti", "sltiu"]:
            return f"{instr} {safe_reg(rt)}, {safe_reg(rs)}, {imm}"
        elif instr in ["lw", "sw"]:
            return f"{instr} {safe_reg(rt)}, {imm}({safe_reg(rs)})"
        elif instr in ["beq", "bne"]:
            return f"{instr} {safe_reg(rs)}, {safe_reg(rt)}, {imm}"
        elif instr == "lui":
            return f"{instr} {safe_reg(rt)}, {imm}"

    #J-Type
    if opcode in [2, 3]:
        addr = instruction_int & 0x3FFFFFF
        return ("j" if opcode == 2 else "jal") + f" {addr}"

    return "invalid MIPS code"


def safe_reg(idx: int) -> str:
    return registers.get(idx, f"$r{idx}")

def convert_file_to_mips(file_path):
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    num_valid_lines = 0
    print(f"\nConverting {file_path}:\n")
    for i, line in enumerate(lines, start=1):
        binary = line.strip()
        result = binary_to_mips(binary)
        if(result != ("invalid MIPS code")):
            num_valid_lines += 1
           
        print(f"Line {i}: {result}")

    return num_valid_lines
