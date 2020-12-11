import os
import re
from glob import iglob
from itertools import chain


SUPPORTED_DATASETS = ["middlebury", "kitti12", "kitti15", "mpi_sintel"]
SINTEL_TRAIN_SEQUENCES = ["alley_1", "alley_2", "ambush_2", "ambush_4", "ambush_5", "ambush_6", "ambush_7", "bamboo_1", "bamboo_2", "bandage_1", "bandage_2", "cave_2", "cave_4", "market_2", "market_5", "market_6", "mountain_1", "shaman_2", "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"]

def getSintelTrain(sintel_imagetype):
    """Get the MPI Sintel train dataset as a dictionary containing file paths.
    sintel_imagetype: image pass, one of "clean" or "final"
    """
    return getTrainDataset("mpi_sintel", sintel_imagetype=sintel_imagetype)


def getSintelTrainClean():
    """Get the MPI Sintel train dataset as a dictionary containing file paths.
    The image pass "clean" is used.
    """
    return getTrainDataset("mpi_sintel", sintel_imagetype="clean")


def getSintelTrainFinal():
    """Get the MPI Sintel train dataset as a dictionary containing file paths.
    The image pass "final" is used.
    """
    return getTrainDataset("mpi_sintel", sintel_imagetype="final")


def getKITTI15Train(kitti_flowtype="flow_occ"):
    """Get the KITTI 15 train dataset as a dictionary containing file paths.
    kitti_flowtype: The flowtype used for evaluation, one of "flow_occ" (all pixels) or "flow_noc" (non-occluded pixels only)
    """
    return getTrainDataset("kitti15", kitti_flowtype=kitti_flowtype)


def getKITTI12Train(kitti_flowtype="flow_occ"):
    """Get the KITTI 12 train dataset as a dictionary containing file paths.
    kitti_flowtype: The flowtype used for evaluation, one of "flow_occ" (all pixels) or "flow_noc" (non-occluded pixels only)
    """
    return getTrainDataset("kitti12", kitti_flowtype=kitti_flowtype)


def getTrainDataset(dataset_name, sintel_imagetype=None, kitti_flowtype="flow_occ"):
    """List image file paths and groundtruth flow file paths for a dataset referenced by name.
    This method returns a dictionary structured by sequence names, which contain a list "images" and "flows".
    A prerequisite is that the datasets are in the correct folder structure:
    The datasets folder is referenced using the environment variable $DATASETS.
    Inside this folder the datasets "middlebury", "kitti12", "kitti15" or "mpi_sintel" are in their respective folder.
    For example:
    $DATASETS
        > kitti12
            > testing
            > training
                > flow_noc
                > flow_occ
                > image_0
                > image_1
        > mpi_sintel
            > training
                > clean
                > final
                > flow

    dataset_name: one of "middlebury", "kitti12", "kitti15" or "mpi_sintel"
    sintel_imagetype: one of "clean" or "final"
    kitti_flowtype: one of "flow_noc" or "flow_occ"
    returns: dictionary with the sequences as keys, each containing two lists "images" and "flows"
    """
    dataset_basepath = os.getenv("DATASETS")

    if dataset_basepath is None:
        raise ValueError(f"DATASET environment variable not given")

    dataset_basepath = os.path.join(dataset_basepath, dataset_name)

    if not os.path.exists(dataset_basepath):
        raise IOError("Dataset basepath does not exist:", dataset_basepath)

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset {dataset_name} currently not supported. Please choose one of: "+ ", ".join(SUPPORTED_DATASETS))

    if kitti_flowtype not in ["flow_noc", "flow_occ"]:
        raise ValueError("kitti_flowtype must be flow_noc or flow_occ!")

    if dataset_name == "mpi_sintel":
        if sintel_imagetype is None:
            raise ValueError("sintel_imagetype not given, must be final or clean!")

        if sintel_imagetype not in ["final", "clean"]:
            raise ValueError("sintel_imagetype must be final or clean!")


    d = {
    "middlebury":
        {
            "base": "training",
            "image_path": "",
            "flow_path": "",
            "sequences": ["Dimetrodon", "Grove2", "Grove3", "Hydrangea", "RubberWhale", "Urban2", "Urban3", "Venus"],
            "image_format": "{seq}" + os.path.sep + "frame{frame:02d}.png",
            "flow_format": "{seq}" + os.path.sep + "flow{frame:02d}.flo",
            "start_frame": 10,
            "end_frame": 11
        },
    "kitti12":
        {
            "base": "training",
            "image_path": "image_0",
            "flow_path": kitti_flowtype,
            "sequences": [f"{i:06d}" for i in range(194)],
            "image_format": "{seq}_{frame:2d}.png",
            "flow_format": "{seq}_{frame:2d}.png",
            "start_frame": 10,
            "end_frame": 11
        },
    "kitti15":
        {
            "base": "training",
            "image_path": "image_2",
            "flow_path": kitti_flowtype,
            "sequences": [f"{i:06d}" for i in range(200)],
            "image_format": "{seq}_{frame:2d}.png",
            "flow_format": "{seq}_{frame:2d}.png",
            "start_frame": 10,
            "end_frame": 11
        },
    "mpi_sintel":
        {
            "base": "training",
            "image_path": sintel_imagetype,
            "image_datatype": ".png",
            "flow_path": "flow",
            "flow_datatype": ".flo",
            "sequences": SINTEL_TRAIN_SEQUENCES,
            "image_format": "{seq}" + os.path.sep + "frame_{frame:04d}.png",
            "flow_format": "{seq}" + os.path.sep + "frame_{frame:04d}.flo",
            "start_frame": 1,
            "end_frame": [50, 50, 21, 33, 50, 20, 50, 50, 50, 50, 50, 50, 50, 50, 50, 40, 50, 50, 50, 50, 50, 50, 50]
        }
    }

    base_image_path = os.path.join(dataset_basepath, d[dataset_name]["base"], d[dataset_name]["image_path"])
    base_flow_path = os.path.join(dataset_basepath, d[dataset_name]["base"], d[dataset_name]["flow_path"])

    if not os.path.exists(base_image_path):
        raise IOError("image path does not exist:", base_image_path)

    if not os.path.exists(base_flow_path):
        raise IOError("flow path does not exist:", base_flow_path)

    result = {}
    for i, sequence in enumerate(d[dataset_name]["sequences"]):
        if type(d[dataset_name]["end_frame"]) is list:
            end_frame = d[dataset_name]["end_frame"][i]
        else:
            end_frame = d[dataset_name]["end_frame"]

        images = []
        for frame in range(d[dataset_name]["start_frame"], end_frame + 1):
            image_path = d[dataset_name]["image_format"].format(seq=sequence, frame=frame)
            image_path = os.path.join(base_image_path, image_path)
            images.append(image_path)

        flows = []
        for frame in range(d[dataset_name]["start_frame"], end_frame):
            flow_path = d[dataset_name]["flow_format"].format(seq=sequence, frame=frame)
            flow_path = os.path.join(base_flow_path, flow_path)
            flows.append(flow_path)

        result[sequence] = {"flows": flows, "images": images}

    return result


def findGroundtruth(filepath):
    sintel_seq = ["alley_1", "alley_2", "ambush_2", "ambush_4", "ambush_5", "ambush_6", "ambush_7", "bamboo_1", "bamboo_2", "bandage_1", "bandage_2", "cave_2", "cave_4", "market_2", "market_5", "market_6", "mountain_1", "shaman_2", "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"]

    sequence = None
    for sq in sintel_seq:
        if sq in filepath:
            sequence = sq

    if sequence is not None:
        # might be sintel
        m = re.search(r"frame_(\d\d\d\d)", filepath)
        if m:
            framenum = int(m.group(1))
            try:
                return getTrainDataset("mpi_sintel")[sequence]["flows"][framenum - 1]
            except Exception as e:
                print(e)
    else:
        # could be kitti
        if "kitti15" in filepath.lower() or "kitti_15" in filepath.lower() or "kitti-15" in filepath.lower():
            m = re.search(r"(\d\d\d\d\d\d)_10", filepath)
            if m:
                sequence = m.group(1)
                try:
                    return getTrainDataset("kitti15", kitti_flowtype="flow_occ")[sequence]["flows"][0]
                except Exception as e:
                    print(e)
    return None


def getSintelTest(sintel_imagetype):
    assert(sintel_imagetype in ["clean", "final"])
    basepath = os.getenv("DATASETS", "")
    basepath = os.path.join(basepath, "mpi_sintel", "test", sintel_imagetype)
    result = {}
    for sequence in sorted(os.listdir(basepath)):
        result[sequence] = {"images": []}
        sq_path = os.path.join(basepath, sequence)
        for frame in sorted(os.listdir(sq_path)):
            result[sequence]["images"].append(os.path.join(sq_path, frame))
    return result


def getKITTI15Test():
    basepath = os.getenv("DATASETS", "")
    basepath = os.path.join(basepath, "kitti15", "testing", "image_2")
    result = {}
    for seq in range(200):
        seq_name = f"{seq:06d}"
        images = [os.path.join(basepath, f"{seq_name}_{i}.png") for i in [10,11]]
        result[seq_name] = {"images": images}
    return result


def testDatasetCompleteness(dataset):
    """
    Check if all flow and image files are existing on disk.
    dataset: dataset dictionary containing flow and image paths
    """
    for _, content in dataset.items():
        for flow in content["flows"]:
            if not os.path.exists(flow):
                print("Flow file does not exist", flow)
        for img in content["images"]:
            if not os.path.exists(img):
                print("Image file does not exist", img)


if __name__ == "__main__":
    sintel_clean = getTrainDataset("mpi_sintel", sintel_imagetype="clean")
    testDatasetCompleteness(sintel_clean)

    sintel_final = getTrainDataset("mpi_sintel", sintel_imagetype="final")
    testDatasetCompleteness(sintel_final)

    kitti15 = getTrainDataset("kitti15", kitti_flowtype="flow_occ")
    testDatasetCompleteness(kitti15)

    kitti12 = getTrainDataset("kitti12")
    testDatasetCompleteness(kitti12)

    middlebury = getTrainDataset("middlebury")
    testDatasetCompleteness(middlebury)
