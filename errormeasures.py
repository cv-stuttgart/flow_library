import numpy as np
import math


def compute_AAE(flow, gt):
    """compute the average angular error (AAE) in degrees between the estimated flow field and the groundtruth flow field
    flow: estimated flow
    gt: groundtruth flow
    return: AAE in [deg]
    """
    arg = flow[:,:,0] * gt[:,:,0] + flow[:,:,1] * gt[:,:,1] + 1
    arg /= np.sqrt(flow[:,:,0]**2+flow[:,:,1]**2+1) * np.sqrt(gt[:,:,0]**2+gt[:,:,1]**2+1)

    np.clip(arg, -1.0, 1.0)

    return np.nansum(np.arccos(arg, where=~np.isnan(arg))) / np.count_nonzero(~np.isnan(arg)) / (2*np.pi) * 360.0


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
    """compute the average endpoint error (AEE) between the estimated flow field and the groundtruth flow field
    flow: estimated flow
    gt: groundtruth flow
    """
    ee, pix_cnt = compute_EE(flow, gt)
    return ee.sum() / float(pix_cnt)


def compute_BP(flow, gt):
    """compute the bad pixel error (BP) between the estimated flow field and the groundtruth flow field.
    The bad pixel error is defined as the percentage of pixels whose endpoint error exceeds 3.
    flow: estimated flow
    gt: groundtruth flow
    return: BP error as percentage [0;100]
    """
    ee, pix_cnt = compute_EE(flow, gt)

    pix_err = ee >= 3.0

    vec_length = np.sqrt(np.square(gt[..., 0]) + np.square(gt[..., 1]))
    err_map = vec_length * 0.05
    pix_err_rel = ee >= err_map

    total_mask = pix_err  # & pix_err_rel # Pixel relative error is mentioned on Kitti webpage

    return 100 * total_mask.sum() / float(pix_cnt)


def printAllErrorMeasures(flow, gt):
    for err, name in zip([compute_AAE, compute_AEE, compute_BP], ["AAE", "AEE", "BP"]):
        print(f"{name:3s}: {err(flow,gt):.2f}")
