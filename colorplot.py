import matplotlib
import numpy as np
import math
import errormeasures


def colorplot(flow, max_scale=1, auto_scale=False, transform=None):
    """
    color-codes a flow input using the color-coding by [Bruhn 2006]
    """
    # prevents nan warnings
    nan = np.isnan(flow[:, :, 0]) | np.isnan(flow[:, :, 1])
    flow[nan, :] = 0

    flow_gradientmag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    if auto_scale:
        max_scale = flow_gradientmag.max()

    hue = -np.arctan2(flow[:, :, 1], flow[:, :, 0]) % (2 * math.pi) / (2 * math.pi) * 360
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
    rgb = matplotlib.colors.hsv_to_rgb(hsv)

    rgb[nan, :] = 0

    # reset flow
    flow[nan, :] = np.nan

    if auto_scale:
        return rgb, max_scale
    else:
        return rgb


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

    ee = errormeasures.compute_EE(flow, gt)

    nan = np.isnan(ee)
    ee = np.nan_to_num(ee)
    result = np.zeros((ee.shape[0], ee.shape[1], 3), dtype=int)

    for threshold, color in reversed(colors):
        result[ee < threshold, :] = color

    # set nan values to black
    result[nan, :] = [0, 0, 0]

    return result
