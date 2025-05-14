# process/encoder.py
import numpy as np
from PIL import Image
from process.mapping import binary_to_base, base_to_binary

def split_channels(arr):
    return [arr] if arr.ndim==2 else [arr[:,:,i] for i in range(arr.shape[2])]

def bitplanes(chan):
    return [(chan>>i)&1 for i in range(8)]

def combine(low,high):
    return [binary_to_base[format(val,'02b')] for val in ((high<<1)|low).flatten()]

def partition(seq,length):
    return [''.join(seq[i:i+length]) for i in range(0,len(seq),length)]

def logistic(x0,r,L,burn=100):
    for _ in range(burn): x0=r*x0*(1-x0)
    xs=[]; x=x0
    for _ in range(L): x=r*x*(1-x); xs.append(x)
    return xs

def encode_and_partition(path,seeds):
    img=Image.open(path)
    arr=np.array(img)
    chans=split_channels(arr)
    h,w=chans[0].shape
    pairs=[(0,7),(1,6),(2,5),(3,4)]
    lengths=[128,64,32,8]
    subseq={}
    for c,chan in enumerate(chans):
        planes=bitplanes(chan)
        for i,(l,hp) in enumerate(pairs):
            key=f"C{c}_P{i+1}"
            seq=combine(planes[l],planes[hp])
            parts=partition(seq,lengths[i])
            xs=logistic(seeds[i],3.99,len(parts))
            for j in range(len(parts)):
                k=int(xs[j]*len(parts))
                parts[j],parts[k]=parts[k],parts[j]
            subseq[key]=parts
    return subseq,(h,w,len(chans)),img.mode


def reconstruct_encrypted_image(subseq,img_info):
    h,w,chs=img_info
    total=h*w
    channels=[]
    mapping={'P1':(0,7),'P2':(1,6),'P3':(2,5),'P4':(3,4)}
    for c in range(chs):
        bitplanes_arr=[np.zeros(total,dtype=np.uint8) for _ in range(8)]
        for pkey,parts in subseq.items():
            if not pkey.startswith(f"C{c}_"): continue
            grp=pkey.split('_')[1]
            low,high=mapping[grp]
            bases=''.join(parts)
            for idx,base in enumerate(bases):
                bstr=base_to_binary.get(base,'00')
                bitplanes_arr[low][idx]=int(bstr[1])
                bitplanes_arr[high][idx]=int(bstr[0])
        pix=np.zeros(total,dtype=np.uint8)
        for i in range(8): pix |= bitplanes_arr[i]<<i
        channels.append(pix.reshape((h,w)))
    if chs==1: return Image.fromarray(channels[0])
    import numpy as _np
    return Image.fromarray(_np.stack(channels,axis=2))