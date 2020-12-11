import os
import re


SUPPORTED_DATASETS = ["middlebury", "kitti12", "kitti15", "mpi_sintel"]
SINTEL_TRAIN_SEQUENCES = ["alley_1", "alley_2", "ambush_2", "ambush_4", "ambush_5", "ambush_6", "ambush_7", "bamboo_1", "bamboo_2", "bandage_1", "bandage_2", "cave_2", "cave_4", "market_2", "market_5", "market_6", "mountain_1", "shaman_2", "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"]
SINTEL_TEST_SEQUENCES = ["ambush_1", "ambush_3", "bamboo_3", "cave_3", "market_1", "market_4", "mountain_2", "PERTURBED_market_3", "PERTURBED_shaman_1", "temple_1", "tiger", "wall"]
SINTEL_TEST_IMG_COUNTS = [23, 41, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]


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
        raise ValueError(f"DATASET environment variable not set")

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


def getSintelTestClean():
    """Get the MPI Sintel test dataset as a dictionary containing image file paths.
    The image pass "clean" is used.
    """
    return getSintelTest("clean")


def getSintelTestFinal():
    """Get the MPI Sintel test dataset as a dictionary containing image file paths.
    The image pass "final" is used.
    """
    return getSintelTest("final")


def getSintelTest(sintel_imagetype):
    """Get the MPI Sintel test dataset as a dictionary containing image file paths.
    sintel_imagetype: one of "clean" or "final
    """
    if sintel_imagetype not in ["clean", "final"]:
        raise ValueError("sintel_imagetype must be clean or final!")

    basepath = os.getenv("DATASETS")
    if basepath is None:
        raise ValueError(f"DATASET environment variable not set")

    basepath = os.path.join(basepath, "mpi_sintel", "test", sintel_imagetype)

    if not os.path.exists(basepath):
        raise IOError("Path does not exist:", basepath)

    result = {}
    for i, sequence in enumerate(SINTEL_TEST_SEQUENCES):
        result[sequence] = {"images": [], "flows": []}
        end = SINTEL_TEST_IMG_COUNTS[i] + 1
        for frame in range(1, end):
            result[sequence]["images"].append(os.path.join(basepath, sequence, f"frame_{frame:04d}.png"))
    return result


def getKITTI15Test():
    """Get the KITTI 15 test dataset as a dictionary containing image file paths.
    """
    basepath = os.getenv("DATASETS")

    if basepath is None:
        raise ValueError(f"DATASET environment variable not set")

    basepath = os.path.join(basepath, "kitti15", "testing", "image_2")

    if not os.path.exists(basepath):
        raise IOError("Path does not exist:", basepath)

    result = {}
    for seq in range(200):
        seq_name = f"{seq:06d}"
        images = [os.path.join(basepath, f"{seq_name}_{i}.png") for i in [10,11]]
        result[seq_name] = {"images": images, "flows": []}
    return result


def getKITTI12Test():
    """Get the KITTI 12 test dataset as a dictionary containing image file paths.
    """
    basepath = os.getenv("DATASETS")

    if basepath is None:
        raise ValueError(f"DATASET environment variable not set")

    basepath = os.path.join(basepath, "kitti12", "testing", "image_0")

    if not os.path.exists(basepath):
        raise IOError("Path does not exist:", basepath)

    result = {}
    for seq in range(195):
        seq_name = f"{seq:06d}"
        images = [os.path.join(basepath, f"{seq_name}_{i}.png") for i in [10,11]]
        result[seq_name] = {"images": images, "flows": []}
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


def findGroundtruth(filepath):
    """Try to automatically find a ground truth flow file for a given filepath.
    returns: path to groundtruth flow or None if not found
    """
    sequence = None
    for sq in SINTEL_TRAIN_SEQUENCES:
        if sq in filepath:
            sequence = sq

    if sequence is not None:
        # might be sintel
        m = re.search(r"frame_(\d\d\d\d)", filepath)
        if m:
            framenum = int(m.group(1))
            return getSintelTrainClean()[sequence]["flows"][framenum - 1]

    else:
        # could be kitti 15
        if "kitti15" in filepath.lower() or "kitti_15" in filepath.lower() or "kitti-15" in filepath.lower():
            m = re.search(r"(\d\d\d\d\d\d)_10", filepath)
            if m:
                sequence = m.group(1)
                return getKITTI15Train()[sequence]["flows"][0]

        # could be kitti 12
        if "kitti12" in filepath.lower() or "kitti_12" in filepath.lower() or "kitti-12" in filepath.lower():
            m = re.search(r"(\d\d\d\d\d\d)_10", filepath)
            if m:
                sequence = m.group(1)
                return getKITTI12Train()[sequence]["flows"][0]

    return None


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

    sintel_clean_test = getSintelTest("clean")
    testDatasetCompleteness(sintel_clean_test)

    sintel_final_test = getSintelTest("final")
    testDatasetCompleteness(sintel_final_test)

    kitti15_test = getKITTI15Test()
    testDatasetCompleteness(kitti15_test)

    kitti12_test = getKITTI12Test()
    testDatasetCompleteness(kitti12_test)
