import yaml


def load_config(config_file_path):
    """
    Load config file in dict
    """
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.CFullLoader)

    return cfg


def load_configs(config_file_path):
    """
    Load config file in dict. Returns each categories.
    """
    cfg = load_config(config_file_path)

    cfg_gen = cfg["GENERATOR"]
    cfg_disc = cfg["DISCRIMINATOR"]
    cfg_data = cfg["DATASET"]
    cfg_train = cfg["TRAIN"]
    cfg_opt_g = cfg_train["OPTIMIZER_GENERATOR"]
    cfg_opt_d = cfg_train["OPTIMIZER_DISCRIMINATOR"]
    cfg_scheduler = cfg_train["SCHEDULER"]
    cfg_logging = cfg_train["LOGGING"]

    return (
        cfg_gen,
        cfg_disc,
        cfg_data,
        cfg_train,
        cfg_opt_g,
        cfg_opt_d,
        cfg_scheduler,
        cfg_logging,
    )
