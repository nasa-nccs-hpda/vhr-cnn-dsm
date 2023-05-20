from enum import Enum
from typing import List
from omegaconf import MISSING
from dataclasses import dataclass, field
from tensorflow_caney.model.config.cnn_config import Config


@dataclass
class DSMConfig(Config):

    # experiment name specific to this work
    OPTIONS = ['stereo', 'disparity', 'stereo-disparity']
    OPTIONS_ENUM = Enum("OPTIONS", {k: k for k in OPTIONS})
    experiment_name: OPTIONS_ENUM = 'stereo'

    # directories pointing to stereo output
    stereo_dirs: List[str] = field(default_factory=lambda: [])

    # regex to read from stereo output directory

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
