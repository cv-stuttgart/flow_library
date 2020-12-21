#! /usr/bin/python3

import sys
import os

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import numpy as np

import flow_plot
import flow_IO
import flow_datasets
import flow_errors


def getFlowVis(flow, vistype="Color Light", auto_scale=False, max_scale=-1, gt=None, return_max=False):
    if vistype == "Color Light":
        return flow_plot.colorplot_light(flow, auto_scale=auto_scale, max_scale=max_scale, return_max=return_max)
    if vistype == "Color Dark":
        return flow_plot.colorplot_dark(flow, auto_scale=auto_scale, max_scale=max_scale, return_max=return_max)
    elif vistype == "Color Log":
        return flow_plot.colorplot_dark(flow, auto_scale=auto_scale, transform="log", max_scale=max_scale, return_max=return_max)
    elif vistype == "Color LogLog":
        return flow_plot.colorplot_dark(flow, auto_scale=auto_scale, transform="loglog", max_scale=max_scale, return_max=return_max)
    elif vistype == "Error":
        if gt is None:
            return np.zeros((flow.shape[0], flow.shape[1]))
        else:
            return flow_plot.errorplot(flow, gt)
    elif vistype == "Error Fl":
        if gt is None:
            return np.zeros((flow.shape[0], flow.shape[1]))
        else:
            return flow_plot.errorplot_Fl(flow, gt)


def maximizeWindow():
    backend = plt.get_backend().lower()
    mng = plt.get_current_fig_manager()
    if backend == "tkagg":
        mng.window.state('zoomed')
    elif backend == "wxagg":
        mng.frame.Maximize(True)
    elif backend == "qt4agg" or backend == "qt5agg":
        mng.window.showMaximized()


def showFlow(filepath):
    flow = flow_IO.readFlowFile(filepath)
    gt_flow = None

    dir_name = os.path.dirname(filepath)
    dir_entries = [os.path.join(dir_name, i) for i in sorted(os.listdir(dir_name))]

    fig, ax = plt.subplots()
    maximizeWindow()
    fig.canvas.set_window_title(filepath)
    plt.subplots_adjust(left=0, right=1, bottom=0.2)

    rgb_vis, max_scale = getFlowVis(flow, auto_scale=True, return_max=True)
    plt.axis("off")
    ax_implot = plt.imshow(rgb_vis, interpolation="nearest")

    axslider = plt.axes([0.05, 0.085, 0.6, 0.03])
    axbuttons = plt.axes([0.7, 0.005, 0.25, 0.195], frame_on=False, aspect='equal')
    slider = Slider(axslider, "max", valmin=0, valmax=200, valinit=max_scale, closedmin=False)
    buttons = RadioButtons(axbuttons, ["Color Light", "Color Dark", "Color Log", "Color LogLog", "Error", "Error Fl"])

    def updateEverything():
        nonlocal flow
        nonlocal gt_flow
        fig.canvas.set_window_title(filepath)
        flow = flow_IO.readFlowFile(filepath)
        gt = None
        try:
            gt = flow_datasets.findGroundtruth(filepath)
        except Exception as e:
            print(e)
        if gt:
            gt_flow = flow_IO.readFlowFile(gt)
            errors = flow_errors.getAllErrorMeasures(flow, gt_flow)
            fig.suptitle(f"AEE: {errors['AEE']:.3f}, Fl: {errors['Fl']:.3f}")
        colorvis = getFlowVis(flow, vistype=buttons.value_selected, max_scale=slider.val, gt=gt_flow)
        ax_implot.set_data(colorvis)
        fig.canvas.draw_idle()

    def update(val):
        val = slider.val
        colorvis = getFlowVis(flow, vistype=buttons.value_selected, max_scale=val, gt=gt_flow)
        ax_implot.set_data(colorvis)
        fig.canvas.draw_idle()

    def format_coord(x, y):
        i = int(x + 0.5)
        j = int(y + 0.5)
        if i >= 0 and i < flow.shape[1] and j >= 0 and j < flow.shape[0]:
            return "pos: ({: 4d},{: 4d}), flow: ({: 4.2f}, {: 4.2f}) ".format(i, j, flow[j, i, 0], flow[j, i, 1])

        return 'x=%1.4f, y=%1.4f' % (x, y)

    def key_press_event(event):
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

    fig.canvas.mpl_connect('key_press_event', key_press_event)
    slider.on_changed(update)
    buttons.on_clicked(update)

    updateEverything()

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        showFlow(sys.argv[1])
    else:
        print(f"Usage:\n  {sys.argv[0]} <flowfile>")
        sys.exit(1)
