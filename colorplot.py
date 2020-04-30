from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import math
import sys


def colorplot(flow, max_scale=1):
    """
    color-codes a flow input using the color-coding by [Bruhn 2006]
    """
    #prevents nan warnings
    nan = np.isnan(flow[:,:,0]) | np.isnan(flow[:,:,1])
    flow[nan,:]=0

    hue = -np.arctan2(flow[:,:,1], flow[:,:,0]) % (2*math.pi) / (2*math.pi) * 360
    hue[hue<90] *= 60/90
    hue[(hue<180) & (hue >=90)] = (hue[(hue<180) & (hue >=90)] - 90) * 60/90 + 60
    hue[hue>=180] = (hue[hue>=180] - 180) * 240/180 + 120
    hue /= 360
    value = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2) / float(max_scale)
    value[value>1.0] = 1.0
    sat = np.ones((flow.shape[0], flow.shape[1]))
    hsv = np.stack((hue,sat,value), axis=-1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)

    rgb[nan,:] = 0

    # reset flow
    flow[nan,:] = np.nan
    
    return rgb
