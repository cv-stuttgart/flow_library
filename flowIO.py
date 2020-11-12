import struct
import numpy as np
from scipy.io import loadmat
import png
import cv2


def readFlowFile(filepath):
    """read flow files in flo, mat or png format. The resulting flow has shape height x width x 2.
    For positions where there is no groundtruth available, the flow is set to np.nan
    returns: flow with shape height x width x 2
    """
    if filepath.endswith(".flo"):
        return readFloFlow(filepath)
    elif filepath.endswith(".mat"):
        return readMatFlow(filepath)
    elif filepath.endswith(".png"):
        return readPngFlow(filepath)
    else:
        raise(f"readFlowFile: Unknown file format for {filepath}")


def writeFlowFile(flow, filepath):
    """write optical flow to file. Supports ".flo" and ".png" (KITTI) file format.
    """
    if filepath.endswith(".flo"):
        return writeFloFlow(flow, filepath)
    elif filepath.endswith(".png"):
        return writePngFlow(flow, filepath)
    else:
        raise(f"writeFlowFile: Unknown file format for {filepath}")


def readMatFlow(filepath):
    mat = loadmat(filepath)
    u = mat["u"]
    v = mat["v"]
    flow = np.stack([u, v], axis=-1)
    return flow


def readPngFlow(filepath):
    flow_object = png.Reader(filename=filepath)
    flow_direct = flow_object.asDirect()
    flow_data = list(flow_direct[2])
    (w, h) = flow_direct[3]['size']
    flow = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
    flow[invalid_idx, 0] = np.nan
    flow[invalid_idx, 1] = np.nan
    return flow[:, :, :2]


# ========================= FLO FORMAT =========================

"""
".flo" file format used for optical flow evaluation

Stores 2-band float image for horizontal (u) and vertical (v) flow components.
Floats are stored in little-endian order.
A flow value is considered "unknown" if either |u| or |v| is greater than 1e9.

 bytes  contents

 0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
         (just a sanity check that floats are represented correctly)
 4-7     width as an integer
 8-11    height as an integer
 12-end  data (width*height*2*4 bytes total)
         the float values for u and v, interleaved, in row order, i.e.,
         u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...
"""

# first four bytes, should be the same in little endian
TAG_FLOAT = 202021.25  # check for this when READING the file
TAG_STRING = "PIEH"    # use this when WRITING the file

# the "official" threshold - if the absolute value of either
# flow component is greater, it's considered unknown
UNKNOWN_FLOW_THRESH = 1e9

# value to use to represent unknown flow
UNKNOWN_FLOW = 1e10


def readFloFlow(filename):
    """read optical flow from file stored in .flo file format
    filename: path to file where to read from
    returns: flow as a numpy array with shape height x width x 2
    """
    if (filename is None):
        raise IOError("read flo file: empty filename")

    if not filename.endswith(".flo"):
        raise IOError(f"read flo file ({filename}): extension .flo expected")

    with open(filename, "rb") as stream:
        tag = struct.unpack("f", stream.read(4))[0]
        width = struct.unpack("i", stream.read(4))[0]
        height = struct.unpack("i", stream.read(4))[0]

        if tag != TAG_FLOAT:  # simple test for correct endian-ness
            raise IOError(f"read flo file({filename}): wrong tag (possibly due to big-endian machine?)")

        # another sanity check to see that integers were read correctly (99999 should do the trick...)
        if width < 1 or width > 99999:
            raise IOError(f"read flo file({filename}): illegal width {width}")

        if height < 1 or height > 99999:
            raise IOError(f"read flo file({filename}): illegal height {height}")

        nBands = 2
        flow = []

        n = nBands * width
        for _ in range(height):
            data = stream.read(n * 4)
            if data is None:
                raise IOError(f"read flo file({filename}): file is too short")
            data = np.asarray(struct.unpack(f"{n}f", data))
            data = data.reshape((width, nBands))
            flow.append(data)

        if stream.read(1) != b'':
            raise IOError(f"read flo file({filename}): file is too long")

        flow = np.asarray(flow)
        # unknown values are set to nan
        flow[np.abs(flow) > UNKNOWN_FLOW_THRESH] = np.nan

        return flow


def writeFloFlow(flow, filename):
    """
    write optical flow in .flo format to file
    flow: optical flow with shape height x width x 2
    filename: optical flow file path to be saved
    """
    if not filename:
        raise ValueError("write flo file: empty filename")

    if not filename.endswith(".flo"):
        raise IOError(f"write flo file {filename}: expected .flo file extension")

    if len(flow.shape) != 3 or flow.shape[2] != 2:
        raise IOError(f"write flo file {filename}: expected shape height x width x 2 but received {flow.shape}")

    if flow.shape[0] > flow.shape[1]:
        print(f"write flo file {filename}: Warning: Are you writing an upright image? Expected shape height x width x 2, got {flow.shape}")

    height, width, nBands = flow.shape

    with open(filename, "wb") as f:
        if f is None:
            raise IOError(f"write flo file {filename}: file could not be opened")

        # write header
        result = f.write(TAG_STRING.encode("ascii"))
        result += f.write(struct.pack('i', width))
        result += f.write(struct.pack('i', height))
        if result != 12:
            raise IOError(f"write flo file {filename}: problem writing header")

        # write content
        n = nBands * width
        for i in range(height):
            data = flow[i, :, :].flatten()
            data[np.isnan(data)] = UNKNOWN_FLOW
            result = f.write(struct.pack(f"{n}f", *data))
            if result != n * 4:
                raise IOError(f"write flo file {filename}: problem writing row {i}")


def writePngFlow(flow, filename):
    flow = 64.0 * flow + 2**15
    valid = np.ones([flow.shape[0], flow.shape[1], 1])
    flow = np.concatenate([flow, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, flow[..., ::-1])
