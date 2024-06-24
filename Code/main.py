from utils.generator import CNN_Generator
from utils.discrimator import CNN_Discriminator


def get_from_cfg(cfg_dict: dict, param=None):
    """
    Return item from config dictionary.
    dictionary has to have a key "NAME" that is the name
    of the instance to return and a key "PARAMETERS"
    which is a dict of the required parameters.
    """
    if param is None:
        return eval(cfg_dict["NAME"])(**cfg_dict["PARAMETERS"])
    else:
        return eval(cfg_dict["NAME"])(param, **cfg_dict["PARAMETERS"])
