from Assembler.verify_instruction_binary import binary_to_mips
def analyze_file(filename):
    total = 0
    valid = 0
    invalid = 0
    cleaned_lines = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            result = binary_to_mips(line)
            if result == "invalid MIPS code":
                invalid += 1
            else:
                valid += 1
                cleaned_lines.append(line)

    if total == 0:
        return filename, 0, 0, 0, []

    valid_pct = (valid / total) * 100
    invalid_pct = (invalid / total) * 100
    return valid_pct, invalid_pct, total, cleaned_lines


def write_clean_file(original_filename, cleaned_lines):
    out_name = f"{original_filename}_cleaned.txt"
    with open(out_name, "w") as f:
        for line in cleaned_lines:
            f.write(line + "\n")
    print(f"Cleaned file written to: {out_name}")
    return out_name


if __name__ == "__main__":
    files = ["malware_train_1", "benign_train_1"]
    for fname in files:
        valid_pct, invalid_pct, total, cleaned_lines = analyze_file(fname + ".txt")
        print(f"File: {fname}")
        print(f"  Total lines: {total}")
        print(f"  Valid: {valid_pct:.2f}%")
        print(f"  Invalid: {invalid_pct:.2f}%\n")
        outname = write_clean_file(fname, cleaned_lines)

        valid_pct, invalid_pct, total, cleaned_lines = analyze_file(outname)
        print(f"File: {outname}")
        print(f"Total lines: {total}")
        print(f"Valid: {valid_pct:.2f}%")
        print(f"Invalid: {invalid_pct:.2f}%\n")