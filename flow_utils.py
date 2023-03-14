import numpy as np


def project(Xs, intrinsics):
    """ Pinhole camera projection """
    X, Y, Z = Xs[:,:,0], Xs[:,:,1], Xs[:,:,2]
    fx, fy, cx, cy = intrinsics

    x = fx * (X / Z) + cx
    y = fy * (Y / Z) + cy
    d = 1.0 / Z

    coords = np.stack([x, y, d], axis=-1)
    return coords


def inv_project(depths, intrinsics):
    """ Pinhole camera inverse-projection """

    ht, wd = depths.shape
    
    fx, fy, cx, cy = intrinsics

    y, x = np.meshgrid(np.arange(ht), np.arange(wd))

    X = depths * ((x.T - cx) / fx)
    Y = depths * ((y.T - cy) / fy)
    Z = depths

    return np.stack([X, Y, Z], axis=-1)


def backproject_flow3d(flow2d, depth0, depth1, intrinsics, T=None, return_2d=False):
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

    point0 = np.stack([X0, Y0, Z0], axis=-1)

    X1 = depth1 * ((x1 - cx) / fx)
    Y1 = depth1 * ((y1 - cy) / fy)
    Z1 = depth1
    point1 = np.stack([X1, Y1, Z1], axis=-1)

    if T is not None:
        point1_hom = np.dstack((point1, np.ones((ht,wd))))
        point1_hom[:,:,:3] *= 0.1
        point1_hom = np.einsum('ijk,lk->ijl', point1_hom, T)
        point1_hom[:,:,:3] /= 0.1
        point1 = point1_hom[:,:,:3] / point1_hom[:,:,3, np.newaxis]

    flow3d = point1-point0

    if not return_2d:
        return flow3d

    x0 = project(point0, intrinsics)
    x1 = project(point1, intrinsics)

    flow2d = x1-x0

    return flow3d, flow2d


def backproject_flow3d_target(flow2d, depth1, intrinsics):
    """ compute 3D flow from 2D flow + depth change """

    ht, wd = flow2d.shape[0:2]

    fx, fy, cx, cy = intrinsics
    
    y0, x0 = np.meshgrid(np.arange(ht), np.arange(wd))
    x0 = x0.T
    y0 = y0.T

    x1 = x0 + flow2d[...,0]
    y1 = y0 + flow2d[...,1]

    X1 = depth1 * ((x1 - cx) / fx)
    Y1 = depth1 * ((y1 - cy) / fy)
    Z1 = depth1
    point1 = np.stack([X1, Y1, Z1], axis=-1)
    return point1


def undo_motioncompensation(flow3d, depth, intrinsics, T):
    X0 = inv_project(depth, intrinsics)
    X1 = X0 + flow3d

    ht, wd, _ = X1.shape
    point1_hom = np.dstack((X1, np.ones((ht,wd))))
    point1_hom[:,:,:3] *= 0.1
    point1_hom = np.einsum('ijk,lk->ijl', point1_hom, T)
    point1_hom[:,:,:3] /= 0.1
    X1 = point1_hom[:,:,:3] / point1_hom[:,:,3, np.newaxis]

    return X1 - X0


def getFlow3D(disp0, disp1, flow, intrinsics):
    depth0 = intrinsics[0] / disp0
    depth1 = intrinsics[0] / disp1
    return backproject_flow3d(flow, depth0, depth1, intrinsics)


def induced_flow(flow3d, depth, intrinsics, min_depth=0.1, T=None):
    """ Compute 2d and 3d flow fields """

    X0 = inv_project(depth, intrinsics)
    X1 = X0 + flow3d


    if T is not None:
        ht, wd, _ = X1.shape
        point1_hom = np.dstack((X1, np.ones((ht,wd))))
        point1_hom[:,:,:3] *= 0.1
        point1_hom = np.einsum('ijk,lk->ijl', point1_hom, T)
        point1_hom[:,:,:3] /= 0.1
        X1 = point1_hom[:,:,:3] / point1_hom[:,:,3, np.newaxis]

    x0 = project(X0, intrinsics)
    x1 = project(X1, intrinsics)

    flow2d = x1 - x0

    valid = (X0[...,-1] > min_depth) & (X1[...,-1] > min_depth)
    return flow2d, valid
