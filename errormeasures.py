import numpy as np


def compute_AAE(flow, gt):
    """compute the average angular error (AAE) in degrees between the estimated flow field and the groundtruth flow field
    flow: estimated flow
    gt: groundtruth flow
    return: AAE in [deg]
    """
    arg = flow[:, :, 0] * gt[:, :, 0] + flow[:, :, 1] * gt[:, :, 1] + 1

    # number of valid pixels:
    count = np.count_nonzero(~np.isnan(arg))

    arg /= np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2 + 1) * np.sqrt(gt[:, :, 0]**2 + gt[:, :, 1]**2 + 1)

    # set nan values to 1 since arccos(1)=0
    arg = np.nan_to_num(arg, nan=1.0)

    # clip to the arccos range [-1;1]
    arg[arg > 1.0] = 1.0
    arg[arg < -1.0] = -1.0

    angular_error = np.arccos(arg)

    return np.sum(angular_error) / count / (2 * np.pi) * 360.0


def compute_EE(flow, gt):
    """compute the endpoint error for every pixel location
    flow: estimated flow
    gt: ground truth flow
    return: 2D np array with pixel-wise endpoint error or nan if no groundtruth is present
    """
    diff = np.square(flow - gt)
    comp = np.sum(diff, axis=-1)
    comp = np.sqrt(comp)

    return comp


def compute_AEE(flow, gt, ee=None):
    """compute the average endpoint error (AEE) between the estimated flow field and the groundtruth flow field
    flow: estimated flow
    gt: groundtruth flow
    ee: precomputed endpoint error
    """
    if ee is None:
        ee = compute_EE(flow, gt)
    count = np.count_nonzero(~np.isnan(ee))
    return np.nansum(ee) / count


def compute_BP(flow, gt, useKITTI15=False, ee=None):
    """compute the bad pixel error (BP) between the estimated flow field and the groundtruth flow field.
    The bad pixel error is defined as the percentage of valid pixels.
    Valid pixel are generally defined as those whose endpoint is smaller than 3px.
    An extension to this definition used for the KITTI15 dataset is that a pixel is valid if
    the endpoint error is smaller than 3px OR the is less than 5% of the groundtruth vector length.
    This extension has an influence if the groundtruth vector lenth is > 60px.
    flow: estimated flow
    gt: groundtruth flow
    useKITTI15: boolean flag if the KITTI15 calculation method should be used (gives better results)
    ee: precomputed endpoint error
    return: BP error as percentage [0;100]
    """
    if ee is None:
        ee = compute_EE(flow, gt)

    # number of valid pixels:
    count = np.count_nonzero(~np.isnan(ee))

    # set the ee of nan pixels to zero
    ee = np.nan_to_num(ee, nan=0.0)
    abs_err = ee >= 3.0

    if useKITTI15:
        gt_vec_length = np.sqrt(np.square(gt[..., 0]) + np.square(gt[..., 1]))
        rel_err = ee >= 0.05 * gt_vec_length

        bp_mask = abs_err & rel_err
    else:
        bp_mask = abs_err

    return 100 * np.sum(bp_mask) / count


def compute_BP_KITTI15(flow, gt, ee=None):
    return compute_BP(flow, gt, useKITTI15=True, ee=ee)


def printAllErrorMeasures(flow, gt):
    for err, name in zip([compute_AAE, compute_AEE, compute_BP], ["AAE", "AEE", "BP"]):
        print(f"{name:3s}: {err(flow,gt):.2f}")


def getAllErrorMeasures(flow, gt):
    result = {}
    result["AAE"] = compute_AAE(flow, gt)

    # precompute EE
    ee = compute_EE(flow, gt)
    for err, name in zip([compute_AEE, compute_BP, compute_BP_KITTI15], ["AEE", "BP", "BPKITTI"]):
        result[name] = err(flow, gt, ee=ee)
    return result
