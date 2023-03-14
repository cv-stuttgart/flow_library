#! /usr/bin/python3

import sys
import os

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import numpy as np

import flow_plot
import flow_IO
import disp_plot


def maximizeWindow():
    backend = plt.get_backend().lower()
    mng = plt.get_current_fig_manager()
    if backend == "tkagg":
        mng.window.state('zoomed')
    elif backend == "wxagg":
        mng.frame.Maximize(True)
    elif backend == "qt4agg" or backend == "qt5agg" or backend == "qtagg":
        mng.window.showMaximized()


def getdisp0(filepath):
    disp0 = None
    filepath2 = os.path.join(os.path.dirname(filepath), "..", "disp_0", os.path.basename(filepath))
    if os.path.exists(filepath2):
        disp0 = flow_IO.readDispFile(filepath2)
    return disp0


def showDisp(filepath):
    filepath = os.path.abspath(filepath)
    disp = flow_IO.readDispFile(filepath)
    disp0 = getdisp0(filepath)

    dir_name = os.path.dirname(filepath)
    dir_entries = [os.path.join(dir_name, i) for i in sorted(os.listdir(dir_name))]

    fig, ax = plt.subplots()
    maximizeWindow()
    plt.get_current_fig_manager().set_window_title(filepath)
    plt.subplots_adjust(left=0, right=1, bottom=0.2)

    plt.axis("off")
    ax_implot = plt.imshow(disp, interpolation="nearest")

    axbuttons = plt.axes([0.7, 0.005, 0.25, 0.195], frame_on=False, aspect='equal')
    buttons = RadioButtons(axbuttons, ["KITTI", "turbo", "plasma", "change"])

    def updateEverything():
        nonlocal disp
        nonlocal disp0

        plt.get_current_fig_manager().set_window_title(filepath)
        disp = flow_IO.readDispFile(filepath)
        disp0 = getdisp0(filepath)

        if buttons.value_selected == "KITTI":
            ax_implot.set_data(disp_plot.colorplot(disp))
        elif buttons.value_selected == "turbo":
            ax_implot.set_data(disp)
            ax_implot.set(cmap="turbo", clim=(0,100))
        elif buttons.value_selected == "plasma":
            ax_implot.set_data(disp)
            ax_implot.set(cmap="plasma", clim=(0,100))
        elif buttons.value_selected == "change":
            ax_implot.set_data(disp-disp0)
            ax_implot.set(cmap="plasma", clim=(-10,10))

        fig.canvas.draw_idle()

    def update(val):
        if buttons.value_selected == "KITTI":
            ax_implot.set_data(disp_plot.colorplot(disp))
        elif buttons.value_selected == "turbo":
            ax_implot.set_data(disp)
            ax_implot.set(cmap="turbo", clim=(0,100))
        elif buttons.value_selected == "plasma":
            ax_implot.set_data(disp)
            ax_implot.set(cmap="plasma", clim=(0,100))
        elif buttons.value_selected == "change":
            ax_implot.set_data(disp-disp0)
            ax_implot.set(cmap="plasma", clim=(-10,10))
        fig.canvas.draw_idle()

    def format_coord(x, y):
        i = int(x + 0.5)
        j = int(y + 0.5)
        if i >= 0 and i < disp.shape[1] and j >= 0 and j < disp.shape[0]:
            return "pos: ({: 4d},{: 4d}), disp: {: 4.2f} ".format(i, j, disp[j, i])

        return 'x=%1.4f, y=%1.4f' % (x, y)

    def key_press_event(event):
        nonlocal filepath
        if event.key not in ["left", "right", "home", "end", "pagedown", "pageup"]:
            return
        idx = dir_entries.index(filepath)
        if event.key == "left" and idx - 1 >= 0:
            filepath = dir_entries[idx - 1]
            updateEverything()
        elif event.key == "right" and idx + 1 < len(dir_entries):
            filepath = dir_entries[idx + 1]
            updateEverything()
        elif event.key == "home":
            filepath = dir_entries[0]
            updateEverything()
        elif event.key == "end":
            filepath = dir_entries[-1]
            updateEverything()
        elif event.key == "pagedown" and idx + 10 < len(dir_entries):
            filepath = dir_entries[idx + 10]
            updateEverything()
        elif event.key == "pageup" and idx - 10 >= 0:
            filepath = dir_entries[idx - 10]
            updateEverything()

    ax.format_coord = format_coord

    fig.canvas.mpl_connect('key_press_event', key_press_event)
    buttons.on_clicked(update)

    updateEverything()

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        showDisp(sys.argv[1])
    else:
        print(f"Usage:\n  {sys.argv[0]} <dispfile>")
        sys.exit(1)
