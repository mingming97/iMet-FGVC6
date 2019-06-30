import os
import sys
import importlib


def cfg_from_file(filepath):
    abspath = os.path.abspath(os.path.expanduser(filepath))
    if not os.path.exists(abspath):
        raise ValueError('File not exists: {}'.format(filepath))
    if not abspath.endswith('.py'):
        raise ValueError('Config file need to be a python file.')
    filename = os.path.basename(abspath)[:-3]
    if '.' in filename:
        raise ValueError('Dots are not allowed in config file name.')
    dirname = os.path.dirname(abspath)
    sys.path.insert(0, dirname)
    module = importlib.import_module(filename)
    sys.path.pop(0)
    cfg = {
        name: value for name, value in module.__dict__.items()
        if not name.startswith('__')
    } 
    return cfg