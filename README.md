# flow_library
Pure python library providing tools for optical flow computation.

### Features:
* Read and write optical flow files (.flo, .png)
* visualize optical flow fields
* interactive inspection tool for optical flow files
* compute error measures
* handle datasets

![flow visualization](docs/flow_plot.gif)

## Installation
Clone the repository and install the necessary requirements.
In order to use this library, add it to the `PYTHONPATH` environment variable.

```console
git clone git@github.com:cvis-stuttgart/flow_library.git
cd flow_library
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

Finally, the library is able to manage dataset filepaths and automatically detect groundtruth flow files if the `DATASET` environment variable is set to the folder containing the desired datasets.
The datasets folder should then be structured as follows:
```
$DATASETS
    > kitti15
        > testing
        > training
            > flow_noc
            > flow_occ
            > image_2
    > mpi_sintel
        > training
            > clean
            > final
            > flow
```
