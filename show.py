#! /usr/bin/python3

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import colorplot
import flowIO
import sys


def showFlow(filepath):
    flow = flowIO.readFlowFile(filepath)

    fig, ax = plt.subplots()
    fig.canvas.set_window_title(filepath)
    plt.subplots_adjust(left=0, right=1)

    rgb_vis, max_scale = colorplot.colorplot(flow, auto_scale=True)
    plt.axis("off")
    ax_implot = plt.imshow(rgb_vis, interpolation="nearest")

    axslider = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(axslider, "max", valmin=0, valmax=200, valinit=max_scale, closedmin=False)

    def update(val):
        val = slider.val
        ax_implot.set_data(colorplot.colorplot(flow, max_scale=val))
        fig.canvas.draw_idle()

    def format_coord(x, y):
        i = int(x + 0.5)
        j = int(y + 0.5)
        if i >= 0 and i < flow.shape[1] and j >= 0 and j < flow.shape[0]:
            return "pos: ({: 4d},{: 4d}), flow: ({: 4.2f}, {: 4.2f}) ".format(i, j, flow[j, i,0], flow[j, i, 1])

        return 'x=%1.4f, y=%1.4f' % (x, y)

    ax.format_coord = format_coord

    slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        showFlow(sys.argv[1])
