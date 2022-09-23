import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plotInteractive(image1_L, image1_R, image2_L, image2_R, disp1, disp2, flow):
    fig, axs = plt.subplots(2, 2, sharex="all", sharey="all")
    axs[0,0].imshow(image1_L)
    axs[0,1].imshow(image1_R)
    axs[1,0].imshow(image2_L)
    axs[1,1].imshow(image2_R)

    def onclick(event):
        if event.inaxes == axs[0,0]:
            # remove everything:
            [[p.remove() for p in reversed(ax.patches)] for ax in [axs[0,0],axs[0,1],axs[1,0],axs[1,1]]]

            # reference frame:
            circle = patches.Circle((event.xdata, event.ydata), radius=0.5, color="red")
            event.inaxes.add_patch(circle)

            x = int(round(2*event.xdata))
            y = int(round(2*event.ydata))
            # right frame:
            circle = patches.Circle((event.xdata-disp1[y,x], event.ydata), radius=0.5, color="red")
            axs[0,1].add_patch(circle)
            plt.draw()
            # t+1 left frame:
            circle = patches.Circle((event.xdata+flow[y,x,0], event.ydata+flow[y,x,1]), radius=0.5, color="red")
            axs[1,0].add_patch(circle)
            plt.draw()
            # t+1 right frame:
            circle = patches.Circle((event.xdata+flow[y,x,0]-disp2[y,x], event.ydata+flow[y,x,1]), radius=0.5, color="red")
            axs[1,1].add_patch(circle)
            plt.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


def plotInteractive4way(image1_L, image1_R, image2_L, image2_R,
        disp1_left, disp1_left_fr2, disp2_FW_left, flow_FW_left, disp2_BW_left, flow_BW_left,
        disp1_right, disp1_right_fr2, disp2_FW_right, flow_FW_right, disp2_BW_right, flow_BW_right):

    fig, axs = plt.subplots(2, 2, sharex="all", sharey="all")
    axs[0,0].imshow(image1_L)
    axs[0,1].imshow(image1_R)
    axs[1,0].imshow(image2_L)
    axs[1,1].imshow(image2_R)

    all_axs = [axs[0,0],axs[0,1],axs[1,0],axs[1,1]]

    def onclick(event):
        if event.inaxes in all_axs:
            # remove everything:
            [[p.remove() for p in reversed(ax.patches)] for ax in all_axs]
        else:
            return

        # reference frame:
        circle = patches.Circle((event.xdata, event.ydata), radius=0.5, color="red")
        event.inaxes.add_patch(circle)

        x = int(round(2*event.xdata))
        y = int(round(2*event.ydata))


        if event.inaxes == axs[0,0]:
            # right frame:
            circle = patches.Circle((event.xdata-disp1_left[y,x], event.ydata), radius=0.5, color="red")
            axs[0,1].add_patch(circle)
            plt.draw()
            # t+1 left frame:
            circle = patches.Circle((event.xdata+flow_FW_left[y,x,0], event.ydata+flow_FW_left[y,x,1]), radius=0.5, color="red")
            axs[1,0].add_patch(circle)
            plt.draw()
            # t+1 right frame:
            circle = patches.Circle((event.xdata+flow_FW_left[y,x,0]-disp2_FW_left[y,x], event.ydata+flow_FW_left[y,x,1]), radius=0.5, color="red")
            axs[1,1].add_patch(circle)
        elif event.inaxes == axs[0,1]:
            # left frame:
            circle = patches.Circle((event.xdata+disp1_right[y,x], event.ydata), radius=0.5, color="red")
            axs[0,0].add_patch(circle)
            plt.draw()
            # t+1 right frame:
            circle = patches.Circle((event.xdata+flow_FW_right[y,x,0], event.ydata+flow_FW_right[y,x,1]), radius=0.5, color="red")
            axs[1,1].add_patch(circle)
            plt.draw()
            # t+1 left frame:
            circle = patches.Circle((event.xdata+flow_FW_right[y,x,0]+disp2_FW_right[y,x], event.ydata+flow_FW_right[y,x,1]), radius=0.5, color="red")
            axs[1,0].add_patch(circle)
        if event.inaxes == axs[1,0]:
            # right frame:
            circle = patches.Circle((event.xdata-disp1_left_fr2[y,x], event.ydata), radius=0.5, color="red")
            axs[1,1].add_patch(circle)
            plt.draw()
            # t left frame:
            circle = patches.Circle((event.xdata+flow_BW_left[y,x,0], event.ydata+flow_BW_left[y,x,1]), radius=0.5, color="red")
            axs[0,0].add_patch(circle)
            plt.draw()
            # t right frame:
            circle = patches.Circle((event.xdata+flow_BW_left[y,x,0]-disp2_BW_left[y,x], event.ydata+flow_BW_left[y,x,1]), radius=0.5, color="red")
            axs[0,1].add_patch(circle)
        elif event.inaxes == axs[1,1]:
            # left frame:
            circle = patches.Circle((event.xdata+disp1_right_fr2[y,x], event.ydata), radius=0.5, color="red")
            axs[1,0].add_patch(circle)
            plt.draw()
            # t right frame:
            circle = patches.Circle((event.xdata+flow_BW_right[y,x,0], event.ydata+flow_BW_right[y,x,1]), radius=0.5, color="red")
            axs[0,1].add_patch(circle)
            plt.draw()
            # t left frame:
            circle = patches.Circle((event.xdata+flow_BW_right[y,x,0]+disp2_FW_right[y,x], event.ydata+flow_BW_right[y,x,1]), radius=0.5, color="red")
            axs[0,0].add_patch(circle)
        plt.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


def plot3DFlow(flow3D, rmax=-1, gmax=-1, bmax=-1):
    if rmax != -1:
        rmax = np.max(np.abs(flow3D[...,0]))
    if gmax != -1:
        gmax = np.max(np.abs(flow3D[...,1]))
    if bmax != -1:
        bmax = np.max(np.abs(flow3D[...,2]))
    vis = flow3D.copy()
    vis[...,0] /= 2*rmax
    vis[...,1] /= 2*gmax
    vis[...,2] /= 2*bmax

    vis += 0.5

    return vis
