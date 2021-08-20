import open3d as o3d
import numpy as np
from flow_utils import inv_project, backproject_flow3d
import matplotlib.pyplot as plt


def sceneflow_plot3D(disp_0, disp_1, flow, intrinsics, image1=None, crop_top=0, T=None):
    pcl, line_set = getPointCloud(disp_0, disp_1, flow, intrinsics, image1=image1, crop_top=crop_top, T=T)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcl)
    vis.add_geometry(line_set)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.15, 0.15, 0.15])
    vis.run()
    vis.destroy_window()


def getPointCloud(disp_0, disp_1, flow, intrinsics, image1=None, crop_top=0, T=None):
    if image1 is not None:
        image1 = np.asarray(image1, dtype=np.float64)
        image1 /= 255.0

    depth_0 = (intrinsics[0] / disp_0)
    depth_1 = (intrinsics[0] / disp_1)

    points1_3D = inv_project(depth_0, intrinsics)

    flow_3D = backproject_flow3d(flow, depth_0, depth_1, intrinsics)

    points2_3D = points1_3D + flow_3D

    points1_3D[:crop_top,:,:] = np.nan
    
    ##

    if T is not None:
        T = np.linalg.inv(T)
        print(T)
        print(points1_3D[300,600])
        print(points2_3D[300,600])
        points2_hom = np.dstack((points2_3D, np.ones((depth_0.shape[0], depth_0.shape[1], 1))))
        vec = points2_hom[300,600]
        print(vec)
        #print(T @ vec)

        #points2_hom[:,:,:3] *= 7.425204778189303
        # points2_hom[:,:,:3] *= 0.01695
        points2_hom = np.einsum('ijk,lk->ijl', points2_hom, T)
        #points2_hom[:,:,:3] /= 7.425204778189303
       #  points2_hom[:,:,:3] /= 0.01695

        print(points2_hom[300,600])
        points2_3D = points2_hom[:,:,:3] / points2_hom[:,:,3, np.newaxis]
        print(points2_3D[300,600])

    ##

    #print(points1_3D[216,904])
    #print(points1_3D[220,950])


    points1_3D = points1_3D.reshape((-1,3))
    points2_3D = points2_3D.reshape((-1,3))
    valid = ~np.isnan(points1_3D[:,0]) & ~np.isnan(points1_3D[:,1]) & ~np.isnan(points1_3D[:,2]) & ~np.isnan(points2_3D[:,0]) & ~np.isnan(points2_3D[:,1]) & ~np.isnan(points2_3D[:,2])
    points1_3D = points1_3D[valid]
    points2_3D = points2_3D[valid]
    if image1 is not None:
        image1 = image1.reshape((-1,3))
        image1 = image1[valid]

    points = np.vstack((points1_3D, points2_3D))
    lines = [[i, i+len(points1_3D)] for i in range(len(points1_3D))]

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points1_3D)
    if image1 is not None:
        pcl.colors = o3d.utility.Vector3dVector(image1)

    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points), lines=o3d.utility.Vector2iVector(lines))

    return pcl, line_set


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