# process/mapping.py
"""
Binary-to-DNA and DNA-to-binary mapping utilities.
"""
# 2-bit binary -> DNA base
binary_to_base = {
    '00': 'A',
    '01': 'T',
    '10': 'G',
    '11': 'C'
}
# Reverse mapping
base_to_binary = {v: k for k, v in binary_to_base.items()}