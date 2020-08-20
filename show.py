#! /usr/bin/python3

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import colorplot
import flowIO
import datasets
import errormeasures
import sys
import os


def showFlow(filepath):
    flow = flowIO.readFlowFile(filepath)

    dir_name = os.path.dirname(filepath)
    dir_entries = [os.path.join(dir_name, i) for i in sorted(os.listdir(dir_name))]

    fig, ax = plt.subplots()
    fig.canvas.set_window_title(filepath)
    plt.subplots_adjust(left=0, right=1)

    rgb_vis, max_scale = colorplot.colorplot(flow, auto_scale=True)
    plt.axis("off")
    ax_implot = plt.imshow(rgb_vis, interpolation="nearest")

    axslider = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(axslider, "max", valmin=0, valmax=200, valinit=max_scale, closedmin=False)

    def updateEverything():
        nonlocal flow
        fig.canvas.set_window_title(filepath)
        flow = flowIO.readFlowFile(filepath)
        gt = datasets.findGroundtruth(filepath)
        if gt:
            gt_flow = flowIO.readFlowFile(gt)
            errors = errormeasures.getAllErrorMeasures(flow, gt_flow)
            fig.suptitle(f"AAE: {errors['AAE']:.3f}, AEE: {errors['AEE']:.3f}, BP: {errors['BP']:.3f}, BPKITTI: {errors['BPKITTI']:.3f}")
        ax_implot.set_data(colorplot.colorplot(flow, max_scale=slider.val))
        fig.canvas.draw_idle()

    def update(val):
        val = slider.val
        ax_implot.set_data(colorplot.colorplot(flow, max_scale=val))
        fig.canvas.draw_idle()

    def format_coord(x, y):
        i = int(x + 0.5)
        j = int(y + 0.5)
        if i >= 0 and i < flow.shape[1] and j >= 0 and j < flow.shape[0]:
            return "pos: ({: 4d},{: 4d}), flow: ({: 4.2f}, {: 4.2f}) ".format(i, j, flow[j, i, 0], flow[j, i, 1])

        return 'x=%1.4f, y=%1.4f' % (x, y)

    def keypress(event):
        nonlocal filepath
        if event.key not in ["left", "right"]:
            return
        idx = dir_entries.index(filepath)
        if event.key == "left" and idx > 0:
            filepath = dir_entries[idx - 1]
            updateEverything()
        elif event.key == "right" and idx < len(dir_entries) - 1:
            filepath = dir_entries[idx + 1]
            updateEverything()

    ax.format_coord = format_coord

    fig.canvas.mpl_connect('key_press_event', keypress)

    slider.on_changed(update)

    updateEverything()

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        showFlow(sys.argv[1])
