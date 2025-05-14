# process/encoder.py
import numpy as np
from PIL import Image
from process.mapping import binary_to_base

def split_channels(arr):
    if arr.ndim==2: return [arr]
    return [arr[:,:,i] for i in range(arr.shape[2])]

def bitplanes(chan):
    return [(chan>>i)&1 for i in range(8)]

def combine(low,high):
    flat=((high<<1)|low).flatten()
    return [binary_to_base[format(v,'02b')] for v in flat]

def partition(seq,n):
    return [''.join(seq[i:i+n]) for i in range(0,len(seq),n)]

def logistic(x0,r,L,burn=100):
    for _ in range(burn): x0=r*x0*(1-x0)
    xs=[]
    x=x0
    for _ in range(L): x=r*x*(1-x); xs.append(x)
    return xs

def encode_and_partition(path,seeds):
    img=Image.open(path)
    arr=np.array(img)
    chans=split_channels(arr)
    h,w=chans[0].shape
    pairs=[(0,7),(1,6),(2,5),(3,4)]
    lengths=[128,64,32,8]
    out={}
    for c,chan in enumerate(chans):
        planes=bitplanes(chan)
        for i,(l,hp) in enumerate(pairs):
            seq=combine(planes[l],planes[hp])
            parts=partition(seq,lengths[i])
            L=len(parts)
            xs=logistic(seeds[i],3.99,L)
            for j in range(L):
                k=int(xs[j]*L)
                parts[j],parts[k]=parts[k],parts[j]
            out[f"C{c}_P{i+1}"]=parts
    return out,(h,w,len(chans)),img.mode