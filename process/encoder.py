# process/encoder.py
import numpy as np
from PIL import Image
from process.mapping import binary_to_base

def logistic_sequence(x0, r, L, burn=100):
    for _ in range(burn): x0 = r * x0 * (1 - x0)
    seq = []
    x = x0
    for _ in range(L):
        x = r * x * (1 - x)
        seq.append(x)
    return seq

def split_channels(arr):
    return [arr] if arr.ndim == 2 else [arr[:,:,i] for i in range(arr.shape[2])]

def bitplanes(chan):
    return [(chan >> i) & 1 for i in range(8)]

def combine_planes(low, high):
    flat = ((high << 1) | low).flatten()
    return [binary_to_base[format(v, '02b')] for v in flat]

def partition(seq, length=2):
    return [''.join(seq[i:i+length]) for i in range(0, len(seq), length)]

def intra_plane_transform(parts, seed):
    xs = logistic_sequence(seed, 3.99, len(parts))
    for j in range(len(parts)):
        k = int(xs[j] * len(parts))
        parts[j], parts[k] = parts[k], parts[j]
    return parts

def encode_and_partition(path, seeds):
    img = Image.open(path)
    arr = np.array(img)
    channels = split_channels(arr)
    h, w = channels[0].shape
    pairs = [(0,7), (1,6), (2,5), (3,4)]
    subseq = {}
    for c, chan in enumerate(channels):
        planes = bitplanes(chan)
        for i, (l, hp) in enumerate(pairs):
            seq = combine_planes(planes[l], planes[hp])
            parts = partition(seq, 2)
            parts = intra_plane_transform(parts, seeds[i])
            subseq[f"C{c}_P{i+1}"] = parts
    return subseq, (h, w, len(channels)), img.mode

def reconstruct_encrypted_image(subseq, img_info):
    from process.mapping import base_to_binary
    import numpy as np
    h, w, chs = img_info
    total = h * w
    encrypted = np.zeros((h, w, chs), dtype=np.uint8) if chs>1 else np.zeros((h, w), dtype=np.uint8)
    mapping = {'P1':(0,7), 'P2':(1,6), 'P3':(2,5), 'P4':(3,4)}
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
        if chs==1:
            encrypted = pix.reshape((h,w))
        else:
            encrypted[:,:,c] = pix.reshape((h,w))
    return encrypted