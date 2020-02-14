"""A global module containing functions for managing the project."""

__author__ = 'wittawat'

import kcgof
import kcgof.log as log
import os
import pickle

def get_root():
    """Return the full path to the root of the package"""
    return os.path.abspath(os.path.dirname(kcgof.__file__))

def result_folder():
    """Return the full path to the result/ folder containing experimental result 
    files"""
    return _get_key_from_default_config('ex_results_path')

def _get_key_from_default_config(key):
    """
    Return the value in the default config dictionary as identified by the key.
    """
    import kcgof.config as config
    con = config.get_default_config()
    value = con[key]
    return value

def get_torch_dtype():
    """
    Return the default torch dtype from the config.
    """
    return _get_key_from_default_config('torch_dtype')

def get_torch_device():
    """
    Return the default torch device from the config.
    """
    return _get_key_from_default_config('torch_device')

# def data_folder():
#     """
#     Return the full path to the data folder 
#     """
#     import kcgof.config as config
#     data_path = config.resource_configs['data_path']
#     return data_path
#     #return os.path.join(get_root(), 'data')

# def data_file(*relative_path):
#     """
#     Access the file under the data folder. The path is relative to the 
#     data folder
#     """
#     dfolder = data_folder()
#     return os.path.join(dfolder, *relative_path)

def shared_resource_folder(*relative_path):
    """
    Return the full path to the shared resource folder.
    """
    path = _get_key_from_default_config('shared_resource_path')
    if relative_path:
        path = os.path.join(path, *relative_path)
    return path

# def load_data_file(*relative_path):
#     fpath = data_file(*relative_path)
#     return pickle_load(fpath)

def ex_result_folder(ex):
    """Return the full path to the folder containing result files of the specified 
    experiment. 
    ex: a positive integer. """
    rp = result_folder()
    fpath = os.path.join(rp, 'ex%d'%ex )
    if not os.path.exists(fpath):
        create_dirs(fpath)
    return fpath

def create_dirs(full_path):
    """Recursively create the directories along the specified path. 
    Assume that the path refers to a folder. """
    if not os.path.exists(full_path):
        os.makedirs(full_path)

def ex_result_file(ex, *relative_path ):
    """Return the full path to the file identified by the relative path as a list 
    of folders/files under the result folder of the experiment ex. """
    rf = ex_result_folder(ex)
    return os.path.join(rf, *relative_path)

def ex_save_result(ex, result, *relative_path):
    """Save a dictionary object result for the experiment ex. Serialization is 
    done with pickle. 
    EX: ex_save_result(1, result, 'data', 'result.p'). Save under result/ex1/data/result.p 
    EX: ex_save_result(1, result, 'result.p'). Save under result/ex1/result.p 
    """
    fpath = ex_result_file(ex, *relative_path)
    dir_path = os.path.dirname(fpath)
    create_dirs(dir_path)
    # 
    with open(fpath, 'wb') as f:
        # expect result to be a dictionary
        pickle.dump(result, f)

def ex_load_result(ex, *relative_path):
    """Load a result identified by the  path from the experiment ex"""
    fpath = ex_result_file(ex, *relative_path)
    return pickle_load(fpath)

def ex_file_exists(ex, *relative_path):
    """Return true if the result file in under the specified experiment folder
    exists"""
    fpath = ex_result_file(ex, *relative_path)
    return os.path.isfile(fpath)

def pickle_load(fpath):
    if not os.path.isfile(fpath):
        raise ValueError('%s does not exist' % fpath)

    with open(fpath, 'rb') as f:
        # expect a dictionary
        result = pickle.load(f)
    return result

