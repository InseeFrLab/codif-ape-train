import logging
import os
import sys

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf

from utils.evaluation import run_evaluation
from utils.mappings import mappings
from utils.mlflow import create_or_restore_experiment, log_dict

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    create_or_restore_experiment(cfg.mlflow.experiment_name)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():
        # Log config
        log_dict(cfg_dict)

        ##### Data #########

        logger.info("Starting data preparation...")

        data_module = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)
        data_module.setup()
        tokenizer = data_module.tokenizer
        Y = data_module.Y

        mlflow.log_param("number_of_training_observations", len(data_module.train_dataset))

        ###### Model #####

        num_classes = max(mappings[Y].values()) + 1

        categorical_vocab_sizes = []
        for feature in cfg.data.categorical_features:
            if feature == "SRF":
                categorical_vocab_sizes.append(5)
            else:
                categorical_vocab_sizes.append(max(mappings[feature].values()) + 1)

        logger.info("Number of classes: " + str(num_classes))
        logger.info("categorical_vocab_sizes " + str(categorical_vocab_sizes))

        if cfg.model.model_name == "torchFastText":
            # for torchFastText only, we add the number of words in the vocabulary
            # In general, tokenizer.num_tokens == num_rows is a invariant
            num_rows = tokenizer.num_tokens + tokenizer.get_nwords() + 1
            padding_idx = num_rows - 1

        # PyTorch model
        model_params = OmegaConf.to_container(cfg.model.model_params, resolve=True)

        model = hydra.utils.instantiate(
            {"_target_": cfg.model._target_, "_convert_": "partial"},
            tokenizer=tokenizer,
            num_rows=num_rows,
            num_classes=num_classes,
            categorical_vocabulary_sizes=categorical_vocab_sizes,
            padding_idx=padding_idx,
            **model_params,
        )

        model = model.to(device)
        logger.info(model)

        # Lightning
        loss = hydra.utils.instantiate(cfg.loss).to(device)
        mlflow.log_param("loss_name", cfg.loss._target_.split(".")[-1])

        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

        module = hydra.utils.instantiate(
            cfg.model.module,
            model=model,
            loss=loss,
            optimizer=optimizer,
            optimizer_params=None,
            scheduler=scheduler,
            scheduler_params=None,
        )
        logger.info(module)

        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param("num_trainable_parameters", num_trainable)

        ###### Trainer #####
        trainer = hydra.utils.instantiate(cfg.model.trainer)

        if cfg_dict["model"]["preprocessor"] == "PyTorch":
            mlflow.pytorch.autolog()
            torch.cuda.empty_cache()
            torch.set_float32_matmul_precision("medium")

        trainer.fit(module, datamodule=data_module)

        ########## Evaluation ##########

        logger.info("Starting evaluation...")

        zipped = zip(
            [data_module.df_val, data_module.df_test],
            [data_module.val_dataloader, data_module.test_dataloader],
            ["val", "test"],
        )

        run_evaluation(
            trainer=trainer,
            module=module,
            revision=cfg_dict["data"]["revision"],
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
