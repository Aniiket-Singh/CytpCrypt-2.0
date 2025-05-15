# process/decoder.py
import numpy as np
from process.mapping import base_to_binary

def logistic_sequence(x0, r, L, burn=100):
    for _ in range(burn): x0 = r * x0 * (1 - x0)
    seq, x = [], x0
    for _ in range(L): x = r * x * (1 - x); seq.append(x)
    return seq

def unshuffle_channels(arr, seed):
    h, w, chs = arr.shape
    flat = arr.reshape(-1, chs)
    xs = logistic_sequence(seed, 3.99, flat.shape[0])
    for i, x in enumerate(xs):
        k = int(x * chs)
        flat[i] = np.roll(flat[i], -k)
    return flat.reshape((h, w, chs))

def decode_from_partitions(subseq, img_info, seeds, enc_arr=None):
    h, w, chs = img_info
    total = h * w
    # reconstruct bitplanes from subseq
    planes = {i: np.zeros((chs, total), dtype=np.uint8) for i in range(8)}
    mapping = {'P1': (0,7), 'P2': (1,6), 'P3': (2,5), 'P4': (3,4)}
    for key, parts in subseq.items():
        c = int(key[1])
        grp = key.split('_')[1]
        low, high = mapping[grp]
        bases = ''.join(parts)[:total]
        for idx, base in enumerate(bases):
            b0, b1 = base_to_binary.get(base, '00')
            planes[low][c, idx] = int(b1)
            planes[high][c, idx] = int(b0)
    # combine and rebuild
    decrypted = np.zeros((h, w, chs), dtype=np.uint8)
    for c in range(chs):
        pix = np.zeros(total, dtype=np.uint8)
        for i in range(8):
            pix |= (planes[i][c] << i)
        decrypted[:, :, c] = pix.reshape((h, w))
    return decrypted
