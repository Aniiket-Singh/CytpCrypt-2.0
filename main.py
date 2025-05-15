# main.py
import os
from PIL import Image
from process.encoder import encode_and_partition, reconstruct_encrypted_image
from process.decoder import decode_from_partitions, unshuffle_channels

# Four seeds for intra-plane P1-P4 and inter-channel shuffle
SEEDS = {
    'intra': [0.12, 0.34, 0.56, 0.78],
    'shuffle': 0.42
}


def process_samples(samples_dir='samples', outputs_dir='outputs'):
    os.makedirs(outputs_dir, exist_ok=True)
    for fname in os.listdir(samples_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        path = os.path.join(samples_dir, fname)

        # Encode + partition + intra-plane transform
        subseq_dict, img_info, mode = encode_and_partition(path, SEEDS['intra'])

        # Reconstruct encrypted array
        enc_arr = reconstruct_encrypted_image(subseq_dict, img_info)

        # Inter-channel shuffle
        from process.encoder import shuffle_channels
        enc_shuffled = shuffle_channels(enc_arr, SEEDS['shuffle'])

        # Save encrypted image
        Image.fromarray(enc_shuffled, mode).save(
            os.path.join(outputs_dir, f"encrypted_{fname}"))

        # Reverse inter-channel shuffle, then decode
        dec_shuffled = unshuffle_channels(enc_shuffled, SEEDS['shuffle'])
        dec_arr = decode_from_partitions(subseq_dict, img_info, SEEDS['intra'], dec_shuffled)
        Image.fromarray(dec_arr, mode).save(
            os.path.join(outputs_dir, f"decoded_{fname}"))

if __name__ == '__main__':
    process_samples()