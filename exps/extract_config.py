import yaml
import hydra
from hydra import initialize, compose

initialize("config") 
def extract_config():
    cfg = compose(config_name="cola_config.yaml")
    return cfg 
