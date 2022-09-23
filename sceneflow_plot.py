import numpy as np
import matplotlib.pyplot as plt



def plotFlow3D(flow3D, rmax=-1, gmax=-1, bmax=-1):
    if rmax != -1:
        rmax = np.max(np.abs(flow3D[...,0]))
    if gmax != -1:
        gmax = np.max(np.abs(flow3D[...,1]))
    if bmax != -1:
        bmax = np.max(np.abs(flow3D[...,2]))
    vis = flow3D.copy()
    vis[...,0] /= 2*rmax
    vis[...,1] /= 2*gmax
    vis[...,2] /= 2*bmax

    vis += 0.5

    return vis
