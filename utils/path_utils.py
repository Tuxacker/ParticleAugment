from pathlib import Path

def get_fq_fpath(path):
    path_obj = Path(path).expanduser().resolve()
    if path_obj.exists() and path_obj.is_file():
        return str(path_obj)
    else:
        raise FileNotFoundError(str(path_obj))

def get_parent_dir(path, go_back=0):
    path_obj = Path(path).expanduser().resolve().parents[go_back]
    if path_obj.exists() and path_obj.is_dir():
        return str(path_obj)
    else:
        raise FileNotFoundError(str(path_obj))

def join_paths(path, file, check=True):
    path_obj = Path(path).expanduser().resolve()
    if path_obj.exists() and path_obj.is_dir() and check:
        if path_obj.joinpath(file).exists():
            return str(path_obj.joinpath(file))
        else:
            raise FileNotFoundError(str(path_obj.joinpath(file)))
    elif not check:
        return str(path_obj.joinpath(file))
    else:
        raise FileNotFoundError(str(path_obj))

def is_dir(path):
    path_obj = Path(path).expanduser().resolve()
    return path_obj.is_dir()
