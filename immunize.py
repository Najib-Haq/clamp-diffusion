import argparse

from clamp_diffusion.config import load_config
from clamp_diffusion.immunize_training import immunize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    immunize(config)
