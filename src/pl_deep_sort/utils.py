from functools import singledispatch
from argparse import Namespace
import os
import yaml
from easydict import EasyDict as edict
# from types import SimpleNamespace

# https://stackoverflow.com/questions/50490856/creating-a-namespace-with-a-dict-of-dicts
@singledispatch
def wrap_namespace(ob):
    return ob

@wrap_namespace.register(dict)
def _wrap_dict(ob):
    return Namespace(**{k: wrap_namespace(v) for k, v in ob.items()})

@wrap_namespace.register(list)
def _wrap_list(ob):
    return [wrap_namespace(v) for v in ob]



# https://github.com/ZQPei/deep_sort_pytorch/blob/master/utils/parser.py
class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """
    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.safe_load(fo.read()))

        super(YamlParser, self).__init__(cfg_dict)

    
    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            self.update(yaml.safe_load(fo.read()))

    
    def merge_from_dict(self, config_dict):
        self.update(config_dict)


def get_config(config_file=None):
    return YamlParser(config_file=config_file)


# if __name__ == "__main__":
#     default_cfg = YamlParser(config_file="./cfgs/default_cfg.yml")
#     print(default_cfg)

#     import ipdb; ipdb.set_trace()