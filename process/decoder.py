# process/decoder.py
import numpy as np
from process.mapping import base_to_binary

def logistic_sequence(x0, r, L, burn=100):
    for _ in range(burn):
        x0 = r * x0 * (1 - x0)
    seq = []
    x = x0
    for _ in range(L):
        x = r * x * (1 - x)
        seq.append(x)
    return seq

def decode_from_partitions(subseq, img_info, seeds):
    h, w, chs = img_info
    total = h * w
    mapping = {'P1': (0,7), 'P2': (1,6), 'P3': (2,5), 'P4': (3,4)}
    for c in range(chs):
        for i in range(4):
            key = f"C{c}_P{i+1}"; parts = subseq[key]
            xs = logistic_sequence(seeds[i], 3.99, len(parts))
            for j in reversed(range(len(parts))):
                k = int(xs[j] * len(parts))
                parts[j], parts[k] = parts[k], parts[j]
            subseq[key] = parts
    decrypted = np.zeros((h, w, chs), dtype=np.uint8) if chs > 1 else np.zeros((h, w), dtype=np.uint8)
    for c in range(chs):
        planes = {i: np.zeros(total, dtype=np.uint8) for i in range(8)}
        for key, parts in subseq.items():
            if not key.startswith(f"C{c}_"): continue
            low, high = mapping[key.split('_')[1]]
            bases = ''.join(parts)
            for idx, base in enumerate(bases[:total]):
                b0, b1 = base_to_binary.get(base, '00')
                planes[low][idx] = int(b1)
                planes[high][idx] = int(b0)
        pix = np.zeros(total, dtype=np.uint8)
        for i in range(8): pix |= (planes[i] << i)
        if chs == 1:
            decrypted = pix.reshape((h, w))
        else:
            decrypted[:, :, c] = pix.reshape((h, w))
    return decrypted