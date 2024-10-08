from __future__ import division
import pstats,cProfile
import numpy,time
import solver as a
from core import *

# Autogenerated with SMOP version 0.1.1
# demo.py solver.m
def solver(ai=None,af=None,w=None,*args,**kwargs):
    nargout = kwargs["nargout"] if kwargs else None
    varargin = cellarray(args)
    nargin = 3-[ai,af,w].count(None)+len(args)

    rand(1,2,3)
    nBlocks=max(ai[:])
    m,n=size(ai,nargout=2)
    I=matlabarray(cat(0,1,0,- 1))
    J=matlabarray(cat(1,0,- 1,0))
    a=copy(ai)
    mv=matlabarray([])
    while not isequal(af,a):

        bid=ceil(rand() * nBlocks)
        i,j=find(a == bid,nargout=2)
        r=ceil(rand() * 4)
        ni=i + I[r]
        nj=j + J[r]
        if (ni < 1) or (ni > m) or (nj < 1) or (nj > n):
            continue
        if a[ni,nj] > 0:
            continue
        ti,tj=find(af == bid,nargout=2)
        d=(ti - i) ** 2 + (tj - j) ** 2
        dn=(ti - ni) ** 2 + (tj - nj) ** 2
        if (d < dn) and (rand() > 0.05):
            continue
        a[ni,nj]=bid
        a[i,j]=0
        mv[end() + 1,cat(1,2)]=cat(bid,r)

    return mv
def rand(*args,**kwargs):
    nargout = kwargs["nargout"] if kwargs else None
    varargin = cellarray(args)
    nargin = 0-[].count(None)+len(args)

    global s1,s2,s3
    if nargin != 0:
        r=0
        s1=varargin[1]
        s2=varargin[2]
        s3=varargin[3]
    else:
        r,s1,s2,s3=r8_random(s1,s2,s3,nargout=4)
    return r
