# Optimizing DEM generation from spaceborne VHR imagery

[![DOI](https://zenodo.org/badge/600075861.svg)](https://zenodo.org/doi/10.5281/zenodo.10091998)
![CI Workflow](https://github.com/nasa-nccs-hpda/vhr-cnn-dsm/actions/workflows/ci.yml/badge.svg)
![CI to DockerHub ](https://github.com/nasa-nccs-hpda/vhr-cnn-dsm/actions/workflows/dockerhub.yml/badge.svg)
![Code style: PEP8](https://github.com/nasa-nccs-hpda/vhr-cnn-dsm/actions/workflows/lint.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/nasa-nccs-hpda/vhr-cnn-dsm/badge.svg?branch=main)](https://coveralls.io/github/nasa-nccs-hpda/vhr-cnn-dsm?branch=main)

The use of stereo imagery to generate 3D surface structures is important to a variety of earth science investigations. Stereo image analysis generates surface and elevation estimates that are crucial to the monitoring of vegetation, safe satellites landing, disaster monitoring, and many other applications that rely on such data at high resolutions. The general methodology to generate digital surface models (DSM) relies on pairing the stereo imagery and performing expensive geometrical physics-based computations to correlate individual pixels to extract local elevations. The NASA Ames Stereo Pipeline (ASP) software follows this procedure at a production level. This software has been heavily optimized to be parallelized using many CPUs. However, the process of estimating elevations in many cases can take more than 10 hours using multi-node and multi-CPU resources. Given the active migration of compute to the commercial cloud, linear CPU acceleration is not enough to maintain costs at a reasonable level for running this type of software at scale. In this work we tested two avenues to accelerate current DSM generation workflows. The first line of research was to test the viability of using physics-informed machine learning models to replace the physics-based computations done by ASP to accelerate the DSM computation process. The second line of research was to optimize the key process of calculating disparity maps for DSMs using CUDA and OpenACC, decreasing this way the use of many CPU resources. This second line of research was done as part of a co-sponsored NVIDIA and NASA GPU Hackathon. Results show these two lines of research as promising avenues to accelerate the acquisition of near real-time elevations. 

## Dataset Archive

Individual stereo scenes:
  - WV02_20160623_10300100577C7E00_1030010058580000
  - WV01_20130825_1020010024E78600_10200100241E6200
  - WV03_20160616_104001001EBDB400_104001001E13F600
 
The ‘disparity maps’ are found in the first 2 bands of the ‘out-F.tif’ file in each directory:
Band 1: the horizontal disparities (x direction)
Band 2: the vertical disparities (y direction)

## Development Runs using the CNN
  
Which workflow from these tests produces a more accurate DSM

- Predict DSM with stereopair
- Predict DSM with disparity map
- Predict DSM with stereopair + disparity map
 
To do these tests, we are building prediction stacks that have some combo of the mapprojected stereopairs:
stereo pair #1: <catid_1>.r100_<res of left image>m.tif
stereo pair #2: <catid_2>.r100_<res of left image>m.tif
disparity map:  out-F.tif

We used the following steps to map the DSM truth values for training:
- Use the 4m version from the DSM output
- Fill its no-data holes with the 24m version
- Interpolate to fill other holes
- Resample to the same grid as that of the prediction stack

## Downloading the Container

The container for this work can be downloaded from DockerHub. The container is deployed on a weekly basis
to take care of potential OS vulnerabilities. All CPU and GPU dependencies are baked into the container image
for end-to-end processing.

```bash
singularity build --sandbox /lscratch/$USER/container/tensorflow-caney docker://nasanccs/tensorflow-caney:latest
```

## Running the Entire Pipeline

Below is the command to run the entire deep learning pipeline. It includes a preprocessing, training, and 
prediction step.

```bash
singularity exec --env PYTHONPATH="development/vhr-cnn-dsm:development/tensorflow-caney" \
    --nv -B $directories_to_mount_inside_the_container \
    tensorflow-caney-container \
    python vhr-cnn-dsm/vhr_cnn_dsm/view/dsm_pipeline_cli.py 
    -c vhr-cnn-dsm/projects/cnn-dsm/configs/cnn_dsm.yaml \
    -s preprocess train predict
```

## Performing Tests on NCCS Explore

For development purposes and to test new features, you will need to test the software pipeline
using pytest. Below is an example of how to run pytest on NASA's NCCS Explore system. You will
need GPUs to run this test workflow.

```bash
singularity exec --env PYTHONPATH="development/vhr-cnn-dsm:development/tensorflow-caney" \
    --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people \
    tensorflow-caney-container python \
    -m pytest /explore/nobackup/people/jacaraba/development/vhr-cnn-dsm/tests
```

## Disparity Map Calculations - GPU Implementation

THe DSG group has worked to implement a GPU (CUDA) implementation of the block-matching algorithm used to calculate disparity maps through Ames Stereo Pipeline (ASP).

Learn more: https://nasa-nccs-hpda.github.io/blockmatchgpu

Repo: https://github.com/nasa-nccs-hpda/blockmatchgpu

## Citing this Software

```bash
Jordan Alexis Caraballo-Vega, “nasa-nccs-hpda/vhr-cnn-dsm: DOI Generation”. Zenodo, Nov. 09, 2023. doi: 10.5281/zenodo.10091999.
```
