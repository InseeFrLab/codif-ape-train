"""
Launch this script with default params. using `uv run -m src.train` from the root of the repository.
"""

import os
import sys

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torchTextClassifiers import ModelConfig, torchTextClassifiers
from torchTextClassifiers.model import TextClassificationModule

from src.datasets import TextClassificationDataModule
from src.utils.build_trainers import build_lightning_trainer
from src.utils.data import CATEGORICAL_FEATURES, mappings
from src.utils.evaluation import run_evaluation
from src.utils.logger import get_logger
from src.utils.mlflow import (
    create_or_restore_experiment,
    init_and_log_wrapper,
)

logger = get_logger(name=__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    # cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    create_or_restore_experiment(cfg.mlflow.experiment_name)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():
        # Log config
        # log_dict(cfg_dict)
        # log_hydra_config(cfg)

        ##### Data #########

        logger.info("Starting data preparation...")

        data_module = TextClassificationDataModule(
            revision=cfg.revision,
            batch_size=cfg.training_config.batch_size,
            tokenizer_cfg=cfg.tokenizer,
        )
        data_module.prepare_data()
        tokenizer = data_module.tokenizer
        Y = data_module.Y

        ###### Model #####

        cfg.model_config.num_classes = max(mappings[Y].values()) + 1

        categorical_vocab_sizes = []
        for feature in CATEGORICAL_FEATURES:
            if feature == "SRF":
                categorical_vocab_sizes.append(5)
            else:
                categorical_vocab_sizes.append(max(mappings[feature].values()) + 1)
        cfg.model_config.categorical_vocabulary_sizes = categorical_vocab_sizes
        logger.info("Number of classes: " + str(cfg.model_config.num_classes))
        logger.info("categorical_vocab_sizes " + str(categorical_vocab_sizes))

        # PyTorch model
        ttc = torchTextClassifiers(
            tokenizer=tokenizer,
            model_config=ModelConfig.from_dict(
                OmegaConf.to_container(cfg.model_config, resolve=True)
            ),
        )
        model = ttc.pytorch_model

        model = model.to(device)
        logger.info(model)

        # Lightning
        loss = hydra.utils.instantiate(cfg.training_config.loss).to(device)
        mlflow.log_param("loss_name", cfg.training_config.loss._target_.split(".")[-1])

        optimizer = hydra.utils.get_class(cfg.training_config.optimizer._target_)
        optimizer_params = {
            k: v for k, v in cfg.training_config.optimizer.items() if k != "_target_"
        }
        scheduler = hydra.utils.get_class(cfg.training_config.scheduler._target_)
        scheduler_params = (
            {k: v for k, v in cfg.training_config.scheduler.items() if k != "_target_"}
            if cfg.training_config.scheduler
            else {}
        )

        module = TextClassificationModule(
            model=model,
            loss=loss,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
        )
        logger.info(module)

        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param("num_trainable_parameters", num_trainable)

        ###### Trainer #####

        trainer = build_lightning_trainer(
            patience_early_stopping=cfg.training_config.patience_early_stopping,
            num_epochs=cfg.training_config.num_epochs,
        )

        mlflow.pytorch.autolog()
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision("medium")

        trainer.fit(module, datamodule=data_module)

        mlflow.log_param("number_of_training_observations", len(data_module.train_dataset))

        # Load the "best" weights (minimizing the val_loss)
        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        checkpoint = torch.load(best_ckpt_path, weights_only=False)
        module.load_state_dict(checkpoint["state_dict"])

        # Log wrapper
        run_id = mlflow.active_run().info.run_id
        logged_pth_path = f"runs:/{run_id}/model/data/model.pth"
        init_and_log_wrapper(
            cfg=cfg, logged_pth_path=logged_pth_path, pre_tokenizer=data_module.pre_tokenizer
        )

        ########## Evaluation ##########

        logger.info("Starting evaluation...")

        zipped = zip(
            [data_module.df_val, data_module.df_test],
            [data_module.val_dataloader(), data_module.test_dataloader()],
            ["val", "test"],
        )

        run_evaluation(
            trainer=trainer,
            module=module,
            revision=cfg.revision,
            Y=Y,
            zipped_data=zipped,
        )


if __name__ == "__main__":
    logger.info("GPU available: " + str(torch.cuda.is_available()))
    for i in range(len(sys.argv)):
        if sys.argv[-1] == "":  # Hydra may get an empty string
            logger.info("Removing empty string argument")
            sys.argv = sys.argv[:-1]  # Remove it
        else:
            break

    # Merge all the args into one
    args = " ".join(sys.argv)
    train()
