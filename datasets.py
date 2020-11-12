import os
import re
from glob import iglob
from itertools import chain


SUPPORTED_DATASETS = ["middlebury", "kitti12", "kitti15", "mpi_sintel", "classroom", "viper"]


def getTrainDataset(dataset_name, kitti_flowtype="flow_noc", sintel_imagetype="final"):
    """List image file paths and groundtruth flow file paths for a dataset referenced by name.
    A prerequisite is that the datasets are in the correct folder structure:
        The datasets folder is referenced using the environment variable $DATASETS.
        Inside this folder the datasets "middlebury", "kitti12", "kitti15", "mpi_sintel", 
        "classroom" or "viper" are in their respective folder.
        For example:
        $DATASETS
            > kitti12
                > testing
                > training
                    > flow_noc
                    > flow_occ
                    > image_0
                    > image_1

    dataset_name: one of "middlebury", "kitti12", "kitti15", "mpi_sintel", "classroom" or "viper"
    [optional] kitti_flowtype: one of "flow_noc" or "flow_occ"
    [optional] sintel_imagetype: one of "clean" or "final"
    returns: dictionary with the sequences as keys, each containing two lists "images" and "flows"
    """
    dataset_basepath = os.getenv("DATASETS", "")
    dataset_basepath = os.path.join(dataset_basepath, dataset_name)

    if not os.path.exists(dataset_basepath):
        raise IOError("Dataset basepath does not exist:", dataset_basepath)

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset {dataset_name} currently not supported. Please choose one of: "+ ", ".join(SUPPORTED_DATASETS))

    if kitti_flowtype not in ["flow_noc", "flow_occ"]:
        raise ValueError("kitti_flowtype must be flow_noc or flow_occ!")

    if sintel_imagetype not in ["final", "clean"]:
        raise ValueError("sintel_imagetype must be final or clean!")


    d = {
    "middlebury":
        {
            "base": "training",
            "image_path": "",
            "image_datatype": ".png",
            "flow_path": "",
            "flow_datatype": ".flo",
            "sequences": ["Dimetrodon", "Grove2", "Grove3", "Hydrangea", "RubberWhale", "Urban2", "Urban3", "Venus"]
        },
    "kitti12":
        {
            "base": "training",
            "image_path": "image_0",
            "image_datatype": ".png",
            "flow_path": kitti_flowtype,
            "flow_datatype": ".png",
            "sequences": [f"{i:06d}" for i in range(194)]
        },
    "kitti15":
        {
            "base": "training",
            "image_path": "image_2",
            "image_datatype": ".png",
            "flow_path": kitti_flowtype,
            "flow_datatype": ".png",
            "sequences": [f"{i:06d}" for i in range(200)]
        },
    "mpi_sintel":
        {
            "base": "training",
            "image_path": sintel_imagetype,
            "image_datatype": ".png",
            "flow_path": "flow",
            "flow_datatype": ".flo",
            "sequences": ["alley_1", "alley_2", "ambush_2", "ambush_4", "ambush_5", "ambush_6", "ambush_7", "bamboo_1", "bamboo_2", "bandage_1", "bandage_2", "cave_2", "cave_4", "market_2", "market_5", "market_6", "mountain_1", "shaman_2", "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"]
        },
    "classroom":
        {
            "base": "",
            "image_path": "",
            "image_datatype": ".png",
            "flow_path": "",
            "flow_datatype": ".flo",
            "sequences": [f"sq_{i}" for i in range(1,5)]
        },
    "viper":
        {
            "base": "train",
            "image_path": "img",
            "image_datatype": ".jpg",
            "flow_path": "flow",
            "flow_datatype": ".mat",
            "sequences": [f"{i:03d}" for i in range(1,76)]
        }
    }

    image_path = os.path.join(dataset_basepath, d[dataset_name]["base"], d[dataset_name]["image_path"])
    flow_path = os.path.join(dataset_basepath, d[dataset_name]["base"], d[dataset_name]["flow_path"])

    if not os.path.exists(image_path):
        raise IOError("image path does not exist:", image_path)

    if not os.path.exists(flow_path):
        raise IOError("flow path does not exist:", flow_path)

    result = {}
    for sequence in d[dataset_name]["sequences"]:
        #print(f"===== {sequence} =====")
        image_seq_path = os.path.join(image_path, sequence)
        flow_seq_path = os.path.join(flow_path, sequence)

        flows = []
        for p in chain(iglob(flow_seq_path+"**", recursive=True), iglob(os.path.join(flow_seq_path,"**"), recursive=True)):
            if p.endswith(d[dataset_name]["flow_datatype"]):
                if os.path.isfile(p):
                    flows.append(p)
        
        images = []
        for p in chain(iglob(image_seq_path+"**", recursive=True), iglob(os.path.join(image_seq_path,"**"), recursive=True)):
            if p.endswith(d[dataset_name]["image_datatype"]):
                if os.path.isfile(p):
                    images.append(p)

        flows.sort()
        images.sort()
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
