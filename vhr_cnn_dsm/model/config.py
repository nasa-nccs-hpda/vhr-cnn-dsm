from enum import Enum
from typing import List
from omegaconf import MISSING
from dataclasses import dataclass, field
from tensorflow_caney.model.config.cnn_config import Config


@dataclass
class DSMConfig(Config):

    # experiment name specific to this work
    EXPERIMENTS = ['stereo', 'disparity', 'stereo-disparity']
    EXPERIMENTS_ENUM = Enum("Experiments", {k: k for k in EXPERIMENTS})
    experiment_name: EXPERIMENTS_ENUM = MISSING

    # directories pointing to stereo output
    stereo_dirs: List[str] = field(default_factory=lambda: [])

    # The configurations below are regex to read from the data outputs
    # trying to find stereo, disparity, and dsm outputs from stereo
    # pipeline. Feel free to modify the regex values if the output
    # from your stereo pipeline is different.

    # disparity map read, one file
    disparity_map_regex: str = 'out-F.tif'

    # stereo pair read, two files
    stereo_pair_regex: str = '*r100_*m.tif'

    # low resolution DSM regex, one file
    lowres_dsm_regex: str = 'out-DEM_24m.tif'

    # mid resolution DSM regex, one file
    midres_dsm_regex: str = 'out-DEM_4m.tif'

    # high resolution DSM regex, one file
    highres_dsm_regex: str = 'out-DEM_1m.tif'

    # number of tiles per scene
    n_tiles: int = 10000
