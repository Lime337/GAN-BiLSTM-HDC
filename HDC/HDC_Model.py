from itertools import islice
import numpy as np
import time
import tracemalloc

#load in data
def load_lines(path: str, n: int = 27_000) -> list[str]:
    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in islice(f, n):
            s = raw.strip()
            if s:
                lines.append(s)
    return lines

#check for binary values
def bitstr_to_array(s: str) -> np.ndarray:
    a = np.frombuffer(s.encode("ascii"), dtype=np.uint8) - ord('0')
    if a.size == 0 or (a.min() < 0) or (a.max() > 1):
        raise ValueError("Input contains non-binary chars.")
    return a.astype(np.uint8)

#expand lines to match d size 
def expand_bitstring(s: str, k: int) -> str:
    if k < 1:
        raise ValueError("k must be >= 1")
    arr = bitstr_to_array(s)
    exp = np.repeat(arr, k)
    return ''.join('1' if v else '0' for v in exp)

#expand batch
def expand_batch(lines: list[str], k: int) -> np.ndarray:
    if k < 1:
        raise ValueError("k must be >= 1")
    X = np.stack([bitstr_to_array(s) for s in lines], axis=0)
    X_exp = np.repeat(X, k, axis=1)
    return X_exp

#seperate testing
def split_train_val(X: np.ndarray, n_val: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return X[train_idx], X[val_idx]

#bipolar conversion
def to_bipolar(X: np.ndarray) -> np.ndarray:
    return (X.astype(np.int8) * 2) - 1

#generate position vectors
def generate_position_hvs(num_positions: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=(num_positions, dim)).astype(np.int8)

#binding function
def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b

#accumulate function
def bundle_bound_vectors(X: np.ndarray, position_hvs: np.ndarray) -> np.ndarray:
    assert X.shape[0] == position_hvs.shape[0], "Mismatch in number of vectors"
    bound = bind(X, position_hvs)
    acc = bound.sum(axis=0, dtype=np.int32)
    hv = np.sign(acc).astype(np.int8)
    hv[acc == 0] = 1
    return hv

#chunk bundle with position vectors
def chunk_and_bundle_with_position(X: np.ndarray, chunk_size: int, position_hvs: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    X_shuffled = X[idx]

    N, D = X_shuffled.shape
    assert N % chunk_size == 0, "Number of samples must be divisible by chunk_size"

    num_chunks = N // chunk_size
    hvs = []
    for i in range(num_chunks):
        chunk = X_shuffled[i*chunk_size:(i+1)*chunk_size]
        chunk_bipolar = to_bipolar(chunk)
        chunk_pos = position_hvs
        hv = bundle_bound_vectors(chunk_bipolar, chunk_pos)
        hvs.append(hv)
    return np.stack(hvs, axis=0)

#cosine similarity and prediction
def cosine_prediction(X: np.ndarray, hv_a: np.ndarray, hv_b: np.ndarray, tie: str = "random", seed: int = 0):
    X32 = X.astype(np.int32, copy=False)
    hv_a32 = hv_a.astype(np.int32, copy=False)
    hv_b32 = hv_b.astype(np.int32, copy=False)
    X_norms = np.linalg.norm(X32, axis=1)
    hv_a_norm = np.linalg.norm(hv_a32)
    hv_b_norm = np.linalg.norm(hv_b32)
    dots_a = X32 @ hv_a32
    dots_b = X32 @ hv_b32
    sim_a  = dots_a / (X_norms * hv_a_norm + 1e-12)
    sim_b  = dots_b / (X_norms * hv_b_norm + 1e-12)
    preds = (sim_b > sim_a).astype(np.int8)
    ties = (sim_a == sim_b)
    if np.any(ties):
        rng = np.random.default_rng(seed)
        preds[ties] = rng.integers(0, 2, size=ties.sum(), dtype=np.int8)
    return preds, sim_a, sim_b

#loading data
benign_class_path = "output\generated_benign_semvalid.txt"
malware_class_path = "output\generated_malware_semvalid.txt"
benign_class_lines = load_lines(benign_class_path)
malware_class_lines = load_lines(malware_class_path)
k = 32
benign_class_exp = expand_batch(benign_class_lines, k)
malware_class_exp = expand_batch(malware_class_lines, k)
print(f"Benign: {len(benign_class_lines)} lines")
print(f"Malware: {len(malware_class_lines)} lines")

benign_train, benign_val = split_train_val(benign_class_exp, n_val=5400, seed=41)
malware_train, malware_val = split_train_val(malware_class_exp, n_val=5400, seed=41)
print("Benign train:", benign_train.shape, "val:", benign_val.shape)
print("Malware train:", malware_train.shape, "val:", malware_val.shape)

position_hvs_10 = generate_position_hvs(10, dim=1024, seed=99)

benign_train_group_hvs = chunk_and_bundle_with_position(benign_train, chunk_size=10, position_hvs=position_hvs_10, seed=40)
malware_train_group_hvs = chunk_and_bundle_with_position(malware_train, chunk_size=10, position_hvs=position_hvs_10, seed=41)

#CPU tracking
tracemalloc.start()
t_start = time.perf_counter()

benign_class_vector = bundle_bound_vectors(benign_train_group_hvs, np.ones_like(benign_train_group_hvs))
malware_class_vector = bundle_bound_vectors(malware_train_group_hvs, np.ones_like(malware_train_group_hvs))

t_end = time.perf_counter()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print("Training Time:", t_end - t_start)
print(f"Training CPU Memory Used: {current / 1024 / 1024:.2f} MB")
print(f"Training CPU Peak Memory: {peak / 1024 / 1024:.2f} MB")

#validation
benign_val_hvs = chunk_and_bundle_with_position(benign_val, chunk_size=10, position_hvs=position_hvs_10, seed=44)
malware_val_hvs = chunk_and_bundle_with_position(malware_val, chunk_size=10, position_hvs=position_hvs_10, seed=45)

#infernece CPU tracking
tracemalloc.start()
start_inf = time.perf_counter()

preds_benign, _, _ = cosine_prediction(benign_val_hvs, benign_class_vector, malware_class_vector)
preds_mal, _, _ = cosine_prediction(malware_val_hvs, benign_class_vector, malware_class_vector)

end_inf = time.perf_counter()
current_inf, peak_inf = tracemalloc.get_traced_memory()
tracemalloc.stop()

print("Inference Time:", end_inf - start_inf)
print(f"Inference CPU Memory Used: {current_inf / 1024 / 1024:.2f} MB")
print(f"Inference CPU Peak Memory: {peak_inf / 1024 / 1024:.2f} MB")

#final evaluation
y_true_ben = np.zeros(benign_val_hvs.shape[0], dtype=np.int8)
y_true_mal = np.ones(malware_val_hvs.shape[0], dtype=np.int8)
y_true = np.concatenate([y_true_ben, y_true_mal])
y_pred = np.concatenate([preds_benign, preds_mal])

acc_ben = (preds_benign == 0).mean()
acc_mal = (preds_mal == 1).mean()
acc_all = (y_pred == y_true).mean()

cm = np.zeros((2,2), dtype=int)
cm[0,0] = (preds_benign == 0).sum()
cm[0,1] = (preds_benign == 1).sum()
cm[1,0] = (preds_mal == 0).sum()
cm[1,1] = (preds_mal == 1).sum()

print(f"Benign val accuracy:  {acc_ben:.4f}")
print(f"Malware val accuracy: {acc_mal:.4f}")
print(f"Overall accuracy:     {acc_all:.4f}")
print("Confusion matrix [[TN, FP],[FN, TP]]:\n", cm)

TP = cm[1,1]
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
prec = TP / (TP + FP + 1e-12)
rec  = TP / (TP + FN + 1e-12)
f1   = 2 * prec * rec / (prec + rec + 1e-12)

print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")