import onnx
import os
import logging
import torch
import joblib
import colorlog
import numpy as np

from model import ColaModel
from data import DataModule
from extract_config import extract_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()  # Prints logs to the console
formatter = colorlog.ColoredFormatter(
    (
        '[%(log_color)s%(asctime)s] - [%(name)s] - [%(levelname)s:%(reset)s] '
        '[%(log_color)s%(message)s%(reset)s]'
    ),
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    reset=True,
    style='%'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

config = extract_config()



def convert_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = config['pretrain_model']
    logger.info(f"Loading pre-trained model from: {model_path}")
    cola_model = joblib.load(model_path)

    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    input_batch = next(iter(data_model.train_dataloader()))
    input_sample = {
        "input_ids": input_batch["input_ids"][0].unsqueeze(0),
        "attention_mask": input_batch["attention_mask"][0].unsqueeze(0),
    }
    print(input_batch["attention_mask"])
    # Export the model
    logger.info(f"Converting the model into ONNX format")
    torch.onnx.export(
        cola_model,  # model being run
        (
            input_sample["input_ids"].to(device),
            input_sample["attention_mask"].to(device),
        ),  # model input (or a tuple for multiple inputs)
        config["onnx_converter_model"],  # where to save the model (can be a file or file-like object)
        export_params=True,
        opset_version=10,
        input_names=["input_ids", "attention_mask"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input_ids": {0: "batch_size"},  # variable length axes
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    onnx_path = config["onnx_converter_model"]
    logger.info(
        f"Model converted successfully. ONNX format model is at: {onnx_path}"
    )

if __name__ == "__main__":
    convert_model()