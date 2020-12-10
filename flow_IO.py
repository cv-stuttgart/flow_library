import struct
import numpy as np
import png

FLO_TAG_FLOAT = 202021.25  # first 4 bytes in flo file; check for this when READING the file
FLO_TAG_STRING = "PIEH"    # first 4 bytes in flo file; use this when WRITING the file
FLO_UNKNOWN_FLOW_THRESH = 1e9 # flo format threshold for unknown values
FLO_UNKNOWN_FLOW = 1e10 # value to use to represent unknown flow in flo file format


def readFlowFile(filepath):
    """read flow files in flo, mat or png format. The resulting flow has shape height x width x 2.
    For positions where there is no groundtruth available, the flow is set to np.nan.
    Supports flo (Sintel), png (KITTI) and npy (numpy) file format.
    filepath: path to the flow file
    returns: flow with shape height x width x 2
    """
    if filepath.endswith(".flo"):
        return readFloFlow(filepath)
    elif filepath.endswith(".png"):
        return readPngFlow(filepath)
    elif filepath.endswith(".npy"):
        return readNpyFlow(filepath)
    else:
        raise ValueError(f"readFlowFile: Unknown file format for {filepath}")


def writeFlowFile(flow, filepath):
    """write optical flow to file. Supports flo (Sintel), png (KITTI) and npy (numpy) file format.
    flow: optical flow with shape height x width x 2. Invalid values should be represented as np.nan
    filepath: file path where to write the flow
    """
    if not filepath:
        raise ValueError("writeFlowFile: empty filepath")

    if len(flow.shape) != 3 or flow.shape[2] != 2:
        raise IOError(f"writeFlowFile {filepath}: expected shape height x width x 2 but received {flow.shape}")

    if flow.shape[0] > flow.shape[1]:
        print(f"write flo file {filepath}: Warning: Are you writing an upright image? Expected shape height x width x 2, got {flow.shape}")

    if filepath.endswith(".flo"):
        return writeFloFlow(flow, filepath)
    elif filepath.endswith(".png"):
        return writePngFlow(flow, filepath)
    elif filepath.endswith(".npy"):
        return writeNpyFlow(flow, filepath)
    else:
        raise ValueError(f"writeFlowFile: Unknown file format for {filepath}")


def readFloFlow(filepath):
    """read optical flow from file stored in .flo file format as used in the Sintel dataset (Butler et al., 2012)
    filepath: path to file where to read from
    returns: flow as a numpy array with shape height x width x 2
    ---
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
    if (filepath is None):
        raise IOError("read flo file: empty filename")

    if not filepath.endswith(".flo"):
        raise IOError(f"read flo file ({filepath}): extension .flo expected")

    with open(filepath, "rb") as stream:
        tag = struct.unpack("f", stream.read(4))[0]
        width = struct.unpack("i", stream.read(4))[0]
        height = struct.unpack("i", stream.read(4))[0]

        if tag != FLO_TAG_FLOAT:  # simple test for correct endian-ness
            raise IOError(f"read flo file({filepath}): wrong tag (possibly due to big-endian machine?)")

        # another sanity check to see that integers were read correctly (99999 should do the trick...)
        if width < 1 or width > 99999:
            raise IOError(f"read flo file({filepath}): illegal width {width}")

        if height < 1 or height > 99999:
            raise IOError(f"read flo file({filepath}): illegal height {height}")

        nBands = 2
        flow = []

        n = nBands * width
        for _ in range(height):
            data = stream.read(n * 4)
            if data is None:
                raise IOError(f"read flo file({filepath}): file is too short")
            data = np.asarray(struct.unpack(f"{n}f", data))
            data = data.reshape((width, nBands))
            flow.append(data)

        if stream.read(1) != b'':
            raise IOError(f"read flo file({filepath}): file is too long")

        flow = np.asarray(flow)
        # unknown values are set to nan
        flow[np.abs(flow) > FLO_UNKNOWN_FLOW_THRESH] = np.nan

        return flow


def writeFloFlow(flow, filepath):
    """
    write optical flow in .flo format to file as used in the Sintel dataset (Butler et al., 2012)
    flow: optical flow with shape height x width x 2
    filepath: optical flow file path to be saved
    ---
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

    height, width, nBands = flow.shape

    with open(filepath, "wb") as f:
        if f is None:
            raise IOError(f"write flo file {filepath}: file could not be opened")

        # write header
        result = f.write(FLO_TAG_STRING.encode("ascii"))
        result += f.write(struct.pack('i', width))
        result += f.write(struct.pack('i', height))
        if result != 12:
            raise IOError(f"write flo file {filepath}: problem writing header")

        # write content
        n = nBands * width
        for i in range(height):
            data = flow[i, :, :].flatten()
            data[np.isnan(data)] = FLO_UNKNOWN_FLOW
            result = f.write(struct.pack(f"{n}f", *data))
            if result != n * 4:
                raise IOError(f"write flo file {filepath}: problem writing row {i}")


def readPngFlow(filepath):
    """read optical flow from file stored in png file format as used in the KITTI 12 (Geiger et al., 2012) and KITTI 15 (Menze et al., 2015) dataset.
    filepath: path to file where to read from
    returns: flow as a numpy array with shape height x width x 2. Invalid values are represented as np.nan
    """
    # adapted from https://github.com/liruoteng/OpticalFlowToolkit
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


def writePngFlow(flow, filename):
    """write optical flow to file png file format as used in the KITTI 12 (Geiger et al., 2012) and KITTI 15 (Menze et al., 2015) dataset.
    flow: optical flow in shape height x width x 2, invalid values should be represented as np.nan
    filepath: path to file where to write to
    """
    flow = 64.0 * flow + 2**15
    width = flow.shape[1]
    height = flow.shape[0]
    valid_map = np.ones([flow.shape[0], flow.shape[1], 1])
    valid_map[np.isnan(flow[:,:,0]) | np.isnan(flow[:,:,1])] = 0
    flow = np.nan_to_num(flow)
    flow = np.concatenate([flow, valid_map], axis=-1).astype(np.uint16)
    flow = np.reshape(flow, (-1, width*3))
    with open(filename, "wb") as f:
        writer = png.Writer(width=width, height=height, bitdepth=16, greyscale=False)
        writer.write(f, flow)


def readNpyFlow(filepath):
    """read numpy array from file.
    filepath: file to read from
    returns: numpy array
    """
    return np.load(filepath)


def writeNpyFlow(flow, filepath):
    """write numpy array to file.
    flow: flow as numpy array to write
    filepath: file to write to
    """
    np.save(filepath, flow)