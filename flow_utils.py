import numpy as np


def inv_project(depths, intrinsics):
    """ Pinhole camera inverse-projection """

    ht, wd = depths.shape
    
    fx, fy, cx, cy = intrinsics

    y, x = np.meshgrid(np.arange(ht), np.arange(wd))

    X = depths * ((x.T - cx) / fx)
    Y = depths * ((y.T - cy) / fy)
    Z = depths

    return np.stack([X, Y, Z], axis=-1)


def backproject_flow3d(flow2d, depth0, depth1, intrinsics):
    """ compute 3D flow from 2D flow + depth change """

    ht, wd = flow2d.shape[0:2]

    fx, fy, cx, cy = intrinsics
    
    y0, x0 = np.meshgrid(np.arange(ht), np.arange(wd))
    x0 = x0.T
    y0 = y0.T

    x1 = x0 + flow2d[...,0]
    y1 = y0 + flow2d[...,1]

    X0 = depth0 * ((x0 - cx) / fx)
    Y0 = depth0 * ((y0 - cy) / fy)
    Z0 = depth0

    X1 = depth1 * ((x1 - cx) / fx)
    Y1 = depth1 * ((y1 - cy) / fy)
    Z1 = depth1

    flow3d = np.stack([X1-X0, Y1-Y0, Z1-Z0], axis=-1)
    return flow3d
