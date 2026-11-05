# -*- coding: utf-8 -*-
"""Render RGB input, one GT and model semantic predictions with a shared view per plant.
Dependency: numpy, matplotlib. Only binary_little_endian vertex PLY supported.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TYPE_MAP = {'char':'i1','uchar':'u1','short':'<i2','ushort':'<u2','int':'<i4','uint':'<u4','float':'<f4','double':'<f8',
            'int8':'i1','uint8':'u1','int16':'<i2','uint16':'<u2','int32':'<i4','uint32':'<u4','float32':'<f4','float64':'<f8'}

def read_ply(path, require_rgb=False):
    names, types = [], []
    with open(path,'rb') as f:
        header=[]
        while True:
            line=f.readline()
            if not line: raise ValueError(f'Missing end_header: {path}')
            s=line.decode('ascii',errors='replace').strip()
            header.append(s)
            if s=='end_header': break
        if 'format binary_little_endian 1.0' not in header:
            raise ValueError(f'Only binary_little_endian PLY is supported: {path}')
        n=None; vertex=False
        for h in header:
            p=h.split()
            if p[:2]==['element','vertex']:
                n=int(p[2]);vertex=True
            elif p[:1]==['element']:
                vertex=False
            elif vertex and p[:1]==['property']:
                if len(p)!=3 or p[1] not in TYPE_MAP: raise ValueError(f'Unsupported vertex property: {h}')
                names.append(p[2]);types.append(TYPE_MAP[p[1]])
        if n is None: raise ValueError(f'No vertex element: {path}')
        data=np.fromfile(f,dtype=np.dtype(list(zip(names,types))),count=n)
    if len(data)!=n: raise ValueError(f'PLY truncated: {path}')
    xyz=np.stack([data[c].astype(np.float64) for c in ('x','y','z')],axis=1)
    if not np.isfinite(xyz).all(): raise ValueError(f'Nonfinite XYZ: {path}')
    if all(c in data.dtype.names for c in ('red','green','blue')):
        rgb=np.stack([data[c] for c in ('red','green','blue')],axis=1).astype(np.float64)
        if rgb.max()>1.5: rgb/=255.0
        rgb=np.clip(rgb,0,1)
    elif require_rgb:
        raise ValueError(f'Original input has no RGB colors: {path}')
    else:
        rgb=np.full((len(xyz),3),0.3)
    return xyz,rgb

def unit(v):
    n=np.linalg.norm(v)
    if n<1e-10: raise ValueError('Cannot construct view: zero vector')
    return v/n

def camera_from_gt(xyz,rgb,flip=False):
    stem=(rgb[:,1]>rgb[:,0]) & (rgb[:,1]>rgb[:,2])
    leaf=rgb[:,0]>rgb[:,1]
    if stem.sum()<20 or leaf.sum()<20: raise ValueError('GT must have green stem and orange leaf points')
    s,l=xyz[stem],xyz[leaf]
    _,v=np.linalg.eigh(np.cov(s.T));up=unit(v[:,-1])
    if np.dot(up,l.mean(0)-s.mean(0))<0:up=-up
    q=l-l.mean(0);q-=np.outer(q@up,up)
    _,v=np.linalg.eigh(np.cov(q.T));right=unit(v[:,-1]-np.dot(v[:,-1],up)*up)
    if right[0]<0:right=-right
    if flip:right=-right
    return right,up

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,required=True,help='One GT semantic PLY')
    p.add_argument('--raw',type=Path,required=True,help='Original RGB point cloud PLY')
    p.add_argument('--files',type=Path,nargs='+',required=True,help='Prediction PLY files, with distinct stems')
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--width',type=int,default=900)
    p.add_argument('--height',type=int,default=1200)
    p.add_argument('--point-size',type=float,default=3.0)
    p.add_argument('--margin',type=int,default=55)
    p.add_argument('--flip-horizontal',action='store_true')
    a=p.parse_args()
    paths=[a.reference,a.raw,*a.files]
    for f in paths:
        if not f.is_file():p.error(f'File not found: {f}')
    if len(set(f.stem for f in paths))!=len(paths):
        p.error('Duplicate file stems would overwrite PNGs; rename a source or use a different name')
    if a.width<=2*a.margin or a.height<=2*a.margin or a.point_size<=0:p.error('Invalid output dimensions/point size')
    clouds=[read_ply(f,require_rgb=(f==a.raw)) for f in paths]
    right,up=camera_from_gt(*clouds[0],flip=a.flip_horizontal)
    ref_xyz=clouds[0][0]
    # Original PLY may be in a different frame than the semantic inputs.
    # Never silently depict them with an inconsistent comparison scale.
    raw_xyz=clouds[1][0]
    ref_span=np.ptp(ref_xyz,axis=0);raw_span=np.ptp(raw_xyz,axis=0)
    if np.any(np.abs(raw_span-ref_span)>np.maximum(ref_span,1e-5)*0.10):
        print('[WARNING] Original RGB scan has different XYZ extents from GT (e.g. preprocessed/cropped).')
        print('          Images retain native coordinates; inspect original PNG before publication.')
    uv=[np.column_stack((xyz@right,xyz@up)) for xyz,_ in clouds]
    union=np.concatenate(uv,axis=0)
    lo,hi=union.min(0),union.max(0)
    span=np.maximum(hi-lo,1e-8)
    scale=min((a.width-2*a.margin)/span[0],(a.height-2*a.margin)/span[1])
    center=(hi+lo)*0.5
    xlim=(center[0]-a.width/(2*scale),center[0]+a.width/(2*scale))
    ylim=(center[1]-a.height/(2*scale),center[1]+a.height/(2*scale))
    a.out_dir.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(a.width/150,a.height/150),dpi=150,facecolor='white')
    ax=fig.add_axes([0,0,1,1],facecolor='white');ax.set_xlim(*xlim);ax.set_ylim(*ylim)
    ax.set_aspect('equal');ax.axis('off')
    print('View right:',np.round(right,5),'up:',np.round(up,5))
    for i,(f,(_,rgb),proj) in enumerate(zip(paths,clouds,uv)):
        name=('GT.png' if i==0 else 'Input_RGB.png' if i==1 else f.stem+'.png')
        sc=ax.scatter(proj[:,0],proj[:,1],c=rgb,s=a.point_size,marker='.',linewidths=0,rasterized=True)
        dest=a.out_dir/name
        fig.savefig(dest,dpi=150,facecolor='white',pad_inches=0)
        sc.remove();print('[PNG]',dest)
    plt.close(fig)
    print(f'[DONE] {len(paths)} PNGs in {a.out_dir}')

if __name__=='__main__': main()
