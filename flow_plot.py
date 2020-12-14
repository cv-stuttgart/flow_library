import matplotlib
import numpy as np
import flow_errors


def colorplot_dark(flow, auto_scale=True, max_scale=-1, transform=None, return_max=False):
    """
    color-codes a flow input using the color-coding by [Bruhn 2006]
    """
    # prevents nan warnings
    nan = np.isnan(flow[:, :, 0]) | np.isnan(flow[:, :, 1])
    flow[nan, :] = 0

    flow_gradientmag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    if auto_scale:
        max_scale = flow_gradientmag.max()

    hue = -np.arctan2(flow[:, :, 1], flow[:, :, 0]) % (2 * np.pi) / (2 * np.pi) * 360
    hue[hue < 90] *= 60 / 90
    hue[(hue < 180) & (hue >= 90)] = (hue[(hue < 180) & (hue >= 90)] - 90) * 60 / 90 + 60
    hue[hue >= 180] = (hue[hue >= 180] - 180) * 240 / 180 + 120
    hue /= 360
    if transform is None:
        value = flow_gradientmag / float(max_scale)
    elif transform == "log":
        # map the range [0-max_scale] to [1-10]:
        value = 9 * flow_gradientmag / float(max_scale) + 1
        # log10:
        value = np.log10(value)
    elif transform == "loglog":
        # map the range [0-max_scale] to [1-10]:
        value = 9 * flow_gradientmag / float(max_scale) + 1
        # log10:
        value = np.log10(value)
        value = 9 * value + 1
        value = np.log10(value)
    else:
        raise ValueError("wrong value for parameter transform")
    value[value > 1.0] = 1.0
    sat = np.ones((flow.shape[0], flow.shape[1]))
    hsv = np.stack((hue, sat, value), axis=-1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv) * 255

    rgb[nan, :] = 0
    rgb = rgb.astype(np.uint8)

    # reset flow
    flow[nan, :] = np.nan

    if return_max:
        return rgb, max_scale
    else:
        return rgb


def colorplot_light(flow, auto_scale=True, max_scale=-1, return_max=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    # adapted from https://github.com/tomrunia/OpticalFlow_Visualization

    assert flow.ndim == 3, 'input flow must have three dimensions'
    assert flow.shape[2] == 2, 'input flow must have shape [H,W,2]'

    nan = np.isnan(flow[:, :, 0]) | np.isnan(flow[:, :, 1])
    flow[nan, :] = 0

    u = flow[:,:,0]
    v = flow[:,:,1]
    # scale flow by maxvalue
    rad = np.sqrt(np.square(u) + np.square(v))
    if auto_scale:
        max_scale = rad.max()
    epsilon = 1e-5
    u = u / (max_scale + epsilon)
    v = v / (max_scale + epsilon)

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = get_Middlebury_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        flow_image[:,:,i] = np.floor(255 * col)
        flow_image[nan, i] = 0
    if return_max:
        return flow_image, max_scale
    else:
        return flow_image


def errorplot(flow, gt):
    colors = [
        (0.1875, [49, 53, 148]),
        (0.375, [69, 116, 180]),
        (0.75, [115, 173, 209]),
        (1.5, [171, 216, 233]),
        (3, [223, 242, 248]),
        (6, [254, 223, 144]),
        (12, [253, 173, 96]),
        (24, [243, 108, 67]),
        (48, [215, 48, 38]),
        (np.inf, [165, 0, 38])
    ]

    ee = flow_errors.compute_EE(flow, gt)

    nan = np.isnan(ee)
    ee = np.nan_to_num(ee)
    result = np.zeros((ee.shape[0], ee.shape[1], 3), dtype=np.uint8)

    for threshold, color in reversed(colors):
        result[ee < threshold, :] = color

    # set nan values to black
    result[nan, :] = [0, 0, 0]

    return result


def errorplot_Fl(flow, gt):
    ee = flow_errors.compute_EE(flow, gt)
    nan = np.isnan(ee)
    ee = np.nan_to_num(ee)
    result = np.zeros((ee.shape[0], ee.shape[1], 3), dtype=np.uint8)

    abs_err = ee >= 3.0

    gt_vec_length = np.sqrt(np.square(gt[..., 0]) + np.square(gt[..., 1]))
    rel_err = ee >= 0.05 * gt_vec_length

    bp_mask = abs_err & rel_err

    result[:,:,:] = (0, 255, 0)
    result[bp_mask,:] = (255, 0, 0)

    result[nan, :] = [0,0,0]
    return result


def get_Middlebury_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """
    # used from https://github.com/tomrunia/OpticalFlow_Visualization

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel
