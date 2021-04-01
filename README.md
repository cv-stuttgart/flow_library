# flow_library
Pure python library providing tools for optical flow computation.

### Features:
* Read and write optical flow files (.flo, .png)
* visualize optical flow fields
* interactive GUI to visualize optical flow files
* compute error measures
* handle optical flow datasets: MPI Sintel, KITTI 12, KITTI 15, Middlebury

![flow visualization](docs/flow_plot.gif)

## Installation
Clone the repository and install the necessary requirements.
In order to use this library, add it to the `PYTHONPATH` environment variable.

```console
git clone git@github.com:cv-stuttgart/flow_library.git
cd flow_library
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

Finally, the library is able to manage dataset filepaths and automatically detect groundtruth flow files if the `DATASET` environment variable is set to the folder containing the desired datasets.
The datasets folder should be structured as follows:
```
$DATASETS
    > kitti15
        > testing
            > image_2
        > training
            > flow_noc
            > flow_occ
            > image_2
    > mpi_sintel
        > test
            > clean
            > final
        > training
            > clean
            > final
            > flow
    > ...
```

## Optical Flow GUI
To use the GUI, right-click an .flo or .png flow file and choose `Open With` > `Other Application`.
There, open the dialog to search for a custom executable, navigate to the `flow_library` folder and select `flow_show.py` (for Windows users: `flow_show.bat`)

In the GUI, you can interactively select the scaling factor (default is the maximum flow vector length) and select the visualization type.
The library tries to automatically detect a groundtruth flow file and shows the average endpoint error (AEE) and the percentage of bad pixels (Fl error).
Use the arrow keys to traverse the current directory.
