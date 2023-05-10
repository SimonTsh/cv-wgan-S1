import torch
import os
import random
import numpy as np

from utils.eval import EvaluationGenerator
from utils.config import load_config
from utils.logs import find_run_path, find_config
from utils.generator import *
from main import get_from_cfg
from utils.transform import *


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", "-lm", type=str, default="")
    parser.add_argument("--transform", "-t", type=str, default=None)
    parser.add_argument("-n", type=int, default=8)
    parser.add_argument("-seed", type=int, default=123)
    args = parser.parse_args()

    run_path = find_run_path(args.load_model, toplogdir="./logs")
    config_path = find_config(run_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config_path == None:
        print("Config path not found")
        exit(1)

    print("Run path found :", run_path)
    print("Config loaded :", config_path)

    config = load_config(config_path)
    cfg_gen = config["GENERATOR"]

    # Load model
    generator = get_from_cfg(cfg_gen)
    generator.eval()
    generator.to(device)

    generator_state_path = os.path.join(run_path, cfg_gen["NAME"] + ".pt")
    generator.load_state_dict(torch.load(generator_state_path))

    # Plot figures
    saving_path = os.path.join(run_path, "figures/")
    print(f"Saving figures at {saving_path}")

    test = EvaluationGenerator(generator, saving_path)
    transform = eval(args.transform) if args.transform != None else None

    seed_everything(args.seed)
    test.interpolate(args.n, transform=transform)

    seed_everything(args.seed)
    test.interpolate_circle(args.n, transform=transform)

    seed_everything(args.seed)
    test.plot(args.n, transform=transform, save_points=True)
