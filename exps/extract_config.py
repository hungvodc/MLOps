import hydra
import yaml
from hydra import compose, initialize

initialize("config")


def extract_config():
    cfg = compose(config_name="cola_config.yaml")
    return cfg
