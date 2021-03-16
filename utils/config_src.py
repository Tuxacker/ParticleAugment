from munch import Munch

from utils.path_utils import join_paths, get_parent_dir

global_config_path = join_paths(get_parent_dir(__file__, go_back=1), "project_config.yaml")

with open(global_config_path, "r") as f:
    global_config = Munch.fromYAML(f)

def get_global_config():
    return global_config

def pprint_global_config():
    raise NotImplementedError