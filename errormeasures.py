import numpy as np


def compute_EE(flow, gt):
    flow_comp = flow[..., :2]
    gt_comp = gt[..., :2]

    diff = flow_comp - gt_comp
    diff = np.square(diff)
    comp = np.sum(diff, axis=-1)
    comp = np.sqrt(comp)

    mask = gt[..., -1] == 0
    comp[mask] = 0.0
    return comp, (gt[..., -1] != 0).sum()


def compute_AEE(flow, gt):
    ee, pix_cnt = compute_EE(flow, gt)
    return ee.sum() / float(pix_cnt)


def compute_BP(flow, gt):
    ee, pix_cnt = compute_EE(flow, gt)

    pix_err = ee >= 3.0

    vec_length = np.sqrt(np.square(gt[..., 0]) + np.square(gt[..., 1]))
    err_map = vec_length * 0.05
    pix_err_rel = ee >= err_map

    total_mask = pix_err  # & pix_err_rel # Pixel relative error is mentioned on Kitti webpage

    return total_mask.sum() / float(pix_cnt)
