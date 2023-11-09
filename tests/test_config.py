import sys
import pytest
from omegaconf import OmegaConf
from vhr_cnn_dsm.model.config import DSMConfig


@pytest.mark.parametrize(
    "filename, expected_tile_size, expected_experiment_type",
    [(
        'tests/test_data/test_config.yaml',
        256,
        'cnn-dsm-stereo'
    )]
)
def test_config_yaml(
            filename: str,
            expected_tile_size: str,
            expected_experiment_type: str
        ):
    schema = OmegaConf.structured(DSMConfig)
    conf = OmegaConf.load(filename)

    try:
        conf = OmegaConf.merge(schema, conf)
    except BaseException as err:
        sys.exit(f"ERROR: {err}")

    assert expected_tile_size == conf.tile_size
    assert expected_experiment_type == conf.experiment_type
