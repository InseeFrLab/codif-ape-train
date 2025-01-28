import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

# Get the absolute path to the project root directory (codif-ape-train)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the torch-fastText directory to the Python path
sys.path.append(os.path.join(project_root, "torch-fastText"))

# Now try importing


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # Debugging


if __name__ == "__main__":
    train()
