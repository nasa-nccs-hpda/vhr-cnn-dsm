# vhr-cnn-dsm

Very-high Resolution CNN-based DSM

## Dataset Archive

General data dir: /panfs/ccds02/nobackup/people/pmontesa/outASP/cnn_mode_test
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

We will use the following steps to map the DSM truth values for training:
- Use the 4m version from the DSM output ()
- Fill its holes with the 24m version ()
- Interpolate to fill other holes ()
- Resample to the same grid as that of the prediction stack
 
## Individual Tests Details

### Predict DSM with stereopair

- Description:
- Data dir: 
- Output dir: 

### Predict DSM with disparity maps

- Description:
- Data dir: 
- Output dir: 

### Predict DSM with stereopair + disparity maps

- Description:
- Data dir: 
- Output dir: 

## Singularity Commands

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/vhr-cnn-dsm:/explore/nobackup/people/jacaraba/development/tensorflow-caney" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/jacaraba/container/tensorflow-caney python /explore/nobackup/people/jacaraba/development/vhr-cnn-dsm/vhr_cnn_dsm/view/dsm_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/vhr-cnn-dsm/projects/cnn-dsm/configs/cnn_dsm.yaml -s setup
```

## Tests

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/vhr-cnn-dsm:/explore/nobackup/people/jacaraba/development/tensorflow-caney" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/jacaraba/container/tensorflow-caney python -m pytest /explore/nobackup/people/jacaraba/development/vhr-cnn-dsm/tests
```