# process/decoder.py
import numpy as np
from process.mapping import base_to_binary

def decode_from_partitions(subseq, img_info):
    h,w,chs = img_info
    total = h*w
    channels = []
    mapping = {'P1':(0,7),'P2':(1,6),'P3':(2,5),'P4':(3,4)}
    for c in range(chs):
        planes = {i:np.zeros(total,dtype=np.uint8) for i in range(8)}
        for key, seqs in subseq.items():
            if not key.startswith(f"C{c}_"): continue
            p = key.split('_')[1]
            low,high = mapping[p]
            bases = list(''.join(seqs))
            for i,b in enumerate(bases):
                b0,b1 = base_to_binary[b]
                planes[low][i] = int(b1)
                planes[high][i] = int(b0)
        pix = np.zeros(total,dtype=np.uint8)
        for i in range(8): pix |= planes[i]<<i
        channels.append(pix.reshape((h,w)))
    if chs==1:
        return channels[0]
    return np.stack(channels, axis=2)

# # process/decoder.py
# import numpy as np
# from process.mapping import base_to_binary

# def decode_from_partitions(subseq, img_info):
#     h, w, chs = img_info
#     total = h * w
#     channels = []
#     mapping = {'P1': (0, 7), 'P2': (1, 6), 'P3': (2, 5), 'P4': (3, 4)}
#     for c in range(chs):
#         planes = {i: np.zeros(total, dtype=np.uint8) for i in range(8)}
#         for key, seqs in subseq.items():
#             if not key.startswith(f"C{c}_"): continue
#             p = key.split('_')[1]
#             if p not in mapping: continue
#             low, high = mapping[p]
#             bases = ''.join(seqs)
#             for idx in range(len(bases)):
#                 base = bases[idx]
#                 if base in base_to_binary:
#                     b0, b1 = base_to_binary[base]
#                     planes[low][idx] = int(b1)
#                     planes[high][idx] = int(b0)

#         pix = np.zeros(total, dtype=np.uint8)
#         for i in range(8):
#             pix |= planes[i] << i

#         channels.append(pix.reshape((h, w)))

#     if chs == 1:
#         return channels[0]
#     return np.stack(channels, axis=2)


# def reconstruct_encrypted_image(subseq, img_info):
#     h, w, chs = img_info
#     total = h * w
#     channels = []
#     mapping = {'P1': (0, 7), 'P2': (1, 6), 'P3': (2, 5), 'P4': (3, 4)}

#     for c in range(chs):
#         planes = {i: np.zeros(total, dtype=np.uint8) for i in range(8)}
#         for key, seqs in subseq.items():
#             if not key.startswith(f"C{c}_"): continue
#             p = key.split('_')[1]
#             if p not in mapping: continue
#             low, high = mapping[p]
#             bases = ''.join(seqs)
#             for idx in range(len(bases)):
#                 base = bases[idx]
#                 if base in base_to_binary:
#                     b0, b1 = base_to_binary[base]
#                     planes[high][idx] = int(b1)
#                     planes[low][idx] = int(b0)

#         pix = np.zeros(total, dtype=np.uint8)
#         for i in range(8):
#             pix |= planes[7 - i] << i  # Reverse the bit-plane order

#         channels.append(pix.reshape((h, w)))

#     if chs == 1:
#         return channels[0]
#     return np.stack(channels, axis=2)
