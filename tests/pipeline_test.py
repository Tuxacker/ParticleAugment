from data.dali_ra import LCRAPipeline
from utils.config_src import get_global_config

def test_create_pipeline():
    config = get_global_config()
    pipeline = LCRAPipeline(config.tests.lmdb_dataset_path, 512, 32, 0, 2, 5)
    pipeline.build()
    assert pipeline is not None