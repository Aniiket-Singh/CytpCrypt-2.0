# process/decoder.py
import numpy as np
from process.mapping import base_to_binary

def decode_from_partitions(subseq,img_info,seeds):
    h,w,chs=img_info
    total=h*w
    mapping={'P1':(0,7),'P2':(1,6),'P3':(2,5),'P4':(3,4)}
    # reverse swaps
    for i in range(4):
        key_pattern=f"_P{i+1}"
        L=len(subseq[f"C0_P{i+1}"])
        xs=logistic(seeds[i],3.99,L)
        for j in reversed(range(L)):
            k=int(xs[j]*L)
            for c in range(chs):
                key=f"C{c}_P{i+1}"
                subseq[key][j],subseq[key][k]=subseq[key][k],subseq[key][j]
    channels=[]
    for c in range(chs):
        planes={i:np.zeros(total,dtype=np.uint8) for i in range(8)}
        for key,parts in subseq.items():
            if not key.startswith(f"C{c}_"): continue
            p=key.split('_')[1]
            low,high=mapping[p]
            bases=''.join(parts)
            for idx,base in enumerate(bases):
                b0,b1=base_to_binary.get(base, '00')
                planes[low][idx]=int(b1)
                planes[high][idx]=int(b0)
        pix=np.zeros(total,dtype=np.uint8)
        for i in range(8): pix |= planes[i]<<i
        channels.append(pix.reshape((h,w)))
    if chs==1: return channels[0]
    return np.stack(channels,axis=2)

def logistic(x0,r,L,burn=100):
    for _ in range(burn): x0=r*x0*(1-x0)
    xs=[]
    x=x0
    for _ in range(L): x=r*x*(1-x); xs.append(x)
    return xs

