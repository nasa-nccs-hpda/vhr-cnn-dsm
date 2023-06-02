import pytest
from vhr_cnn_dsm.model.dsm_pipeline import DSMPipeline


@pytest.mark.parametrize(
    "filename, expected_tile_size," +
    "expected_experiment_type, expected_data_dir",
    [(
        'tests/test_data/test_config.yaml',
        256,
        'cnn-dsm-stereo',
        'regression-test'
    )]
)
def test_pipeline_init(
            filename,
            expected_tile_size,
            expected_experiment_type,
            expected_data_dir
        ):
    pipeline = DSMPipeline(filename)
    assert pipeline.conf.tile_size == expected_tile_size
    assert pipeline.conf.experiment_type == expected_experiment_type
    assert pipeline.conf.data_dir == expected_data_dir
