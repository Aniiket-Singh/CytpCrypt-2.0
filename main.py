# main.py
import os
from PIL import Image
from process.encoder import encode_and_partition
from process.decoder import decode_from_partitions

SEEDS = [0.12, 0.34, 0.56, 0.78]

def process_samples(samples_dir='samples', outputs_dir='outputs'):
    os.makedirs(outputs_dir, exist_ok=True)
    for fname in os.listdir(samples_dir):
        if not fname.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff')):
            continue
        path = os.path.join(samples_dir, fname)
        subseq_dict, img_info, mode = encode_and_partition(path, SEEDS)
        base = os.path.splitext(fname)[0]
        for key, seqs in subseq_dict.items():
            with open(os.path.join(outputs_dir, f"{base}_{key}.txt"), 'w') as f:
                f.write("\n".join(seqs))
        arr = decode_from_partitions(subseq_dict, img_info, SEEDS)
        out = Image.fromarray(arr, mode)
        out.save(os.path.join(outputs_dir, f"decoded_{fname}"))

if __name__ == '__main__':
    process_samples()