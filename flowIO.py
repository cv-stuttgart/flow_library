# read and write our .flo flow file format

import struct
import numpy as np
from scipy.io import loadmat
import png

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
        print("Unknown file format for", filepath)


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


# ".flo" file format used for optical flow evaluation
#
# Stores 2-band float image for horizontal (u) and vertical (v) flow components.
# Floats are stored in little-endian order.
# A flow value is considered "unknown" if either |u| or |v| is greater than 1e9.
#
#  bytes  contents
#
#  0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
#          (just a sanity check that floats are represented correctly)
#  4-7     width as an integer
#  8-11    height as an integer
#  12-end  data (width*height*2*4 bytes total)
#          the float values for u and v, interleaved, in row order, i.e.,
#          u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...

# first four bytes, should be the same in little endian
TAG_FLOAT = 202021.25  # check for this when READING the file
TAG_STRING = "PIEH"    # use this when WRITING the file

# the "official" threshold - if the absolute value of either
# flow component is greater, it's considered unknown
UNKNOWN_FLOW_THRESH = 1e9

# value to use to represent unknown flow
UNKNOWN_FLOW = 1e10

# return whether flow vector is unknown
# def unknown_flow(u, v):
#     return (fabs(u) >  UNKNOWN_FLOW_THRESH) or (fabs(v) >  UNKNOWN_FLOW_THRESH) or isnan(u) or isnan(v)

# bool unknown_flow(float *f) {
#     return unknown_flow(f[0], f[1]);
# }


# read a flow file into 2-band image
def readFloFlow(filename):

    if (filename is None):
        raise "ReadFlowFile: empty filename"

    if not filename.endswith(".flo"):
        raise f"ReadFlowFile ({filename}): extension .flo expected"

    with open(filename, "rb") as stream:

        tag = struct.unpack("f", stream.read(4))[0]
        width = struct.unpack("i", stream.read(4))[0]
        height = struct.unpack("i", stream.read(4))[0]

        if tag != TAG_FLOAT:  # simple test for correct endian-ness
            raise f"ReadFlowFile({filename}): wrong tag (possibly due to big-endian machine?)"

        # another sanity check to see that integers were read correctly (99999 should do the trick...)
        if width < 1 or width > 99999:
            raise f"ReadFlowFile({filename}): illegal width {width}"

        if height < 1 or height > 99999:
            raise f"ReadFlowFile({filename}): illegal height {height}"

        nBands = 2
        flow = []

        n = nBands * width
        for _ in range(height):
            # float* ptr = &img.Pixel(0, y, 0);
            data = stream.read(n * 4)
            if data is None:
                raise f"ReadFlowFile({filename}): file is too short"
            data = np.asarray(struct.unpack(f"{n}f", data))
            data = data.reshape((width, nBands))
            flow.append(data)

        if stream.read(1) != b'':
            raise f"ReadFlowFile({filename}): file is too long"

        flow = np.asarray(flow)
        return flow


def writeFloFlow(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()


# # write a 2-band image into flow file
# void WriteFlowFile(CFloatImage img, const char* filename)
# {
#     if (filename == NULL)
#     throw CError("WriteFlowFile: empty filename");

#     char *dot = strrchr(filename, '.');
#     if (dot == NULL)
#     throw CError("WriteFlowFile: extension required in filename '%s'", filename);

#     if (strcmp(dot, ".flo") != 0)
#     throw CError("WriteFlowFile: filename '%s' should have extension '.flo'", filename);

#     CShape sh = img.Shape();
#     int width = sh.width, height = sh.height, nBands = sh.nBands;

#     if (nBands != 2)
#     throw CError("WriteFlowFile(%s): image must have 2 bands", filename);

#     FILE *stream = fopen(filename, "wb");
#     if (stream == 0)
#         throw CError("WriteFlowFile: could not open %s", filename);

#     # write the header
#     fprintf(stream, TAG_STRING);
#     if ((int)fwrite(&width,  sizeof(int),   1, stream) != 1 ||
#     (int)fwrite(&height, sizeof(int),   1, stream) != 1)
#     throw CError("WriteFlowFile(%s): problem writing header", filename);

#     # write the rows
#     int n = nBands * width;
#     for (int y = 0; y < height; y++) {
#     float* ptr = &img.Pixel(0, y, 0);
#     if ((int)fwrite(ptr, sizeof(float), n, stream) != n)
#         throw CError("WriteFlowFile(%s): problem writing data", filename);
#    }

#     fclose(stream);
# }
