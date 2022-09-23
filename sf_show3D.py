#! /usr/bin/python3

import sys

from PIL import Image
import numpy as np

import sceneflow_plot3D
import flow_IO
import flow_datasets


def main(filepath):
    disp0path, disp1path, flowpath = flow_datasets.sf_findCorrespondingFiles(filepath)
    disp0 = flow_IO.readDispFile(disp0path)
    disp1 = flow_IO.readDispFile(disp1path)
    flow = flow_IO.readFlowFile(flowpath)

    seq = int(flowpath[-13:-7])
    
    mode = None
    if "testing" in filepath:
        mode = "testing"
    elif "training" in filepath:
        mode = "training"
    else:
        raise ValueError("I don't know if this file belongs to the training or testing split")

    intrinsics = flow_datasets.getIntrinsics_KITTI(seq, mode)

    if mode == "training":
        image1 = flow_datasets.getKITTI15Train()[f"{seq:06d}"]["images"][0]
    else:
        image1 = flow_datasets.getKITTI15Test()[f"{seq:06d}"]["images"][0]

    sceneflow_plot3D.sceneflow_plot3D(disp0, disp1, flow, intrinsics, np.asarray(Image.open(image1)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage:\n    {sys.argv[0]} <filepath>")
    else:
        main(sys.argv[1])
