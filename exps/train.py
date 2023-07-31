import logging

import colorlog
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from data import DataModule
from extract_config import extract_config
from model import ColaModel
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

wandb.login()

config = extract_config()

new_config = dict(
    epochs=config["num_epochs"],
    learning_rate=config["learning_rate"],
    save_frequency=config["save_frequency"],
    save_path=config["save_path"],
    model_name=config["model_name"],
    batch_size=config["batch_size"],
    save_model=config["save_model"],
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()  # Prints logs to the console
formatter = colorlog.ColoredFormatter(
    (
        "[%(log_color)s%(asctime)s] - [%(name)s] - [%(levelname)s:%(reset)s] "
        "[%(log_color)s%(message)s%(reset)s]"
    ),
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
    reset=True,
    style="%",
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def load_data():
    cola_data = DataModule()
    cola_data.prepare_data()
    cola_data.setup()
    train_dataloader = cola_data.train_dataloader()
    val_dataloader = cola_data.val_dataloader()
    return train_dataloader, val_dataloader


def load_model():
    cola_model = ColaModel()
    optimizer = cola_model.configure_optimizers()
    return cola_model, optimizer


def confusion_matrix_func(preds, target):
    target_res, preds_res = [], []
    for eles in target:
        if type(eles) is list:
            for ele in eles:
                target_res.append(ele)
    for eles in preds:
        if type(eles) is list:
            for ele in eles:
                preds_res.append(ele)
    # preds_res = [ele for sub in preds for ele in sub]
    data = confusion_matrix(target_res, preds_res)
    df_cm = pd.DataFrame(
        data, columns=np.unique(target_res), index=np.unique(target_res)
    )
    df_cm.index.name = "Actual"
    df_cm.columns.name = "Predicted"
    plt.figure(figsize=(10, 10))
    plot = sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})  #
    wandb.log({"Confusion Matrix": wandb.Image(plot)})


def main():
    with wandb.init(project="cola_model", config=new_config):
        logger.warning(f"Training config: {wandb.config}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dataloader, val_dataloader = load_data()
        cola_model, optimizer = load_model()
        cola_model.to(device)
        wandb.watch(cola_model, log="all", log_freq=40)
        for epoch in range(wandb.config["epochs"]):
            cola_model.train()
            (
                total_acc_train,
                total_loss_train,
                total_train_pre_score,
                total_train_rec_score,
                total_train_f1,
            ) = (0, 0, 0, 0, 0)
            for train_input in tqdm(train_dataloader):
                cola_model.zero_grad()
                (
                    loss,
                    acc,
                    train_pre_score,
                    train_rec_score,
                    train_f1,
                ) = cola_model.training_step(train_input)
                total_loss_train += loss
                total_acc_train += acc
                total_train_pre_score += train_pre_score
                total_train_rec_score += train_rec_score
                total_train_f1 += train_f1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cola_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            cola_model.eval()

            (
                total_val_acc,
                total_val_loss,
                total_val_pre_score,
                total_val_rec_score,
                total_val_f1,
            ) = (0, 0, 0, 0, 0)
            preds_res, target_res = [], []
            for val_input in tqdm(val_dataloader):
                with torch.no_grad():
                    (
                        val_loss,
                        val_acc,
                        val_pre_score,
                        val_rec_score,
                        val_f1,
                        preds,
                        target,
                    ) = cola_model.validation_step(val_input)
                    total_val_loss += val_loss
                    total_val_acc += val_acc
                    total_val_pre_score += val_pre_score
                    total_val_rec_score += val_rec_score
                    total_val_f1 += val_f1
                    preds_res.append(preds)
                    target_res.append(target)
            # print(target_res)
            wandb.log(
                {
                    "train_loss": total_loss_train / len(train_dataloader),
                    "train_acc": total_acc_train / len(train_dataloader),
                    "train_precision": total_train_pre_score / len(train_dataloader),
                    "train_recall": total_val_rec_score / len(train_dataloader),
                    "train_f1": total_train_f1 / len(train_dataloader),
                    "val_loss": total_val_loss / len(val_dataloader),
                    "val_acc": total_val_acc / len(val_dataloader),
                    "val_precision": total_val_pre_score / len(val_dataloader),
                    "val_recall": total_val_rec_score / len(val_dataloader),
                    "val_f1": total_val_f1 / len(val_dataloader),
                }
            )
            logger.info(
                f"train_loss: {total_loss_train / len(train_dataloader)} || train_acc: {total_acc_train / len(train_dataloader)} || val_loss: {total_val_loss / len(val_dataloader)} || val_acc: {total_val_acc / len(val_dataloader)}"
            )
            if (
                epoch % wandb.config["save_frequency"] == 0
                and wandb.config["save_model"] is True
            ):
                save_path = wandb.config["save_path"]
                save_path += f"cola_epoch_{epoch}"
                wandb.unwatch()
                joblib.dump(cola_model, save_path)
            if epoch == wandb.config["epochs"] - 1:
                confusion_matrix_func(preds_res, target_res)


if __name__ == "__main__":
    main()
