import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d
from flow_utils import inv_project


def colorplot(disp, scale=None):
    """same colorplot as KITTI"""
    # TODO: implement as matplotlib colormap
    initial_steps = [114, 185, 114, 174, 114, 185, 114]
    colors = [[0,0,0], [0,0,1], [1,0,0], [1,0,1], [0,1,0], [0,1,1], [1,1,0], [1,1,1]]
    steps = [0.0] + [sum(initial_steps[:i+1]) / float(sum(initial_steps)) for i in range(7)]

    h,w = disp.shape
    result = np.zeros((h,w,3),dtype=np.uint8)

    if scale is None:
        scale = np.nanmax(disp)
    disp_ = disp / scale
    disp_ = np.clip(disp_, 0, 1)

    for i in range(len(steps)-1):
        col1 = np.asarray(colors[i], dtype=np.uint8)
        col2 = np.asarray(colors[i+1], dtype=np.uint8)
        alpha = (disp_ - steps[i]) / (steps[i+1]-steps[i])
        alpha = alpha[:,:,np.newaxis]
        interpol = ((1-alpha) * col1 + alpha*col2) * 255
        result[(disp_>=steps[i]) & (disp_<steps[i+1]),:] = np.asarray(np.floor(interpol), dtype=np.uint8)[(disp_>=steps[i]) & (disp_<steps[i+1]),:]

    return result


# def plot_3D(disp, intrinsics):
#     depth_0 = (intrinsics[0] / disp)
#     points_3D = inv_project(depth_0, intrinsics)
#     # flatten
#     points_3D = points_3D.reshape((-1,3))
#     points_3D = points_3D[~np.isnan(points_3D[:,0]) & ~np.isnan(points_3D[:,1]) & ~np.isnan(points_3D[:,2])]
#     pcl = o3d.geometry.PointCloud()
#     pcl.points = o3d.utility.Vector3dVector(points_3D.reshape((-1,3)))
#     o3d.visualization.draw_geometries([pcl])
