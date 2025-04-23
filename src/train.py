import logging
import os
import sys

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torchFastText.datasets import FastTextModelDataset

from framework_classes import (
    DATASETS,
    LOSSES,
    MODELS,
    MODULES,
    OPTIMIZERS,
    SCHEDULERS,
    TOKENIZERS,
    TRAINERS,
)
from utils.data import get_processed_data, get_Y
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
    create_or_restore_experiment(cfg_dict["mlflow"]["experiment_name"])
    mlflow.set_experiment(cfg_dict["mlflow"]["experiment_name"])

    with mlflow.start_run():
        # Log config
        log_dict(cfg_dict)

        ##### Data #########
        Y = get_Y(revision=cfg_dict["data"]["revision"])
        df_train, df_val, df_test = get_processed_data(revision=cfg_dict["data"]["revision"])

        df_train = df_train.sample(frac=0.001)
        df_val = df_val.sample(frac=0.01)
        df_test = df_test.sample(frac=0.01)

        mlflow.log_param("number_of_training_observations", df_train.shape[0])

        train_text, train_categorical_variables = (
            df_train[cfg_dict["data"]["text_feature"]].values,
            df_train[cfg_dict["data"]["categorical_features"]].values,
        )
        val_text, val_categorical_variables = (
            df_val[cfg_dict["data"]["text_feature"]].values,
            df_val[cfg_dict["data"]["categorical_features"]].values,
        )
        test_text, test_categorical_variables = (
            df_test[cfg_dict["data"]["text_feature"]].values,
            df_test[cfg_dict["data"]["categorical_features"]].values,
        )

        ###### Tokenizer ######

        tokenizer = TOKENIZERS[cfg_dict["tokenizer"]["tokenizer_name"]](
            **cfg_dict["tokenizer"], training_text=train_text
        )
        logger.info(tokenizer)

        ###### Dataset ######

        if cfg_dict["dataset"] is not None:
            similarity_coefficients = cfg_dict["dataset"].get("similarity_coefficients", None)

            dataset_class = DATASETS[cfg_dict["dataset"]["dataset_name"]]

            train_dataset = dataset_class(
                texts=train_text,
                categorical_variables=train_categorical_variables,
                tokenizer=tokenizer,
                outputs=df_train[Y].values,
                revision=cfg_dict["data"]["revision"],
                similarity_coefficients=similarity_coefficients,
            )
            val_dataset = dataset_class(
                texts=val_text,
                categorical_variables=val_categorical_variables,
                tokenizer=tokenizer,
                outputs=df_val[Y].values,
                revision=cfg_dict["data"]["revision"],
                similarity_coefficients=similarity_coefficients,
            )

            test_dataset = dataset_class(
                texts=test_text,
                categorical_variables=test_categorical_variables,
                tokenizer=tokenizer,
                outputs=df_test[Y].values,
                revision=cfg_dict["data"]["revision"],
                similarity_coefficients=similarity_coefficients,
            )

            if isinstance(train_dataset, FastTextModelDataset):
                train_dataloader = train_dataset.create_dataloader(
                    **cfg_dict["model"]["train_params"]
                )
                val_dataloader = val_dataset.create_dataloader(**cfg_dict["model"]["train_params"])
                test_dataloader = test_dataset.create_dataloader(
                    **cfg_dict["model"]["train_params"]
                )

        ###### Model #####
        num_classes = max(mappings[Y].values()) + 1

        categorical_vocab_sizes = []
        for feature in cfg_dict["data"]["categorical_features"]:
            if feature == "SRF":
                categorical_vocab_sizes.append(5)
            else:
                categorical_vocab_sizes.append(max(mappings[feature].values()) + 1)

        logger.info("Number of classes: " + str(num_classes))
        logger.info("categorical_vocab_sizes " + str(categorical_vocab_sizes))

        if cfg_dict["model"]["model_name"] == "torchFastText":
            # for torchFastText only, we add the number of words in the vocabulary
            # In general, tokenizer.num_tokens == num_rows is a invariant
            num_rows = tokenizer.num_tokens + tokenizer.get_nwords() + 1
            padding_idx = num_rows - 1

        # PyTorch model
        model = MODELS[cfg_dict["model"]["model_name"]](
            **cfg_dict["model"]["model_params"],
            tokenizer=tokenizer,
            num_rows=num_rows,
            num_classes=num_classes,
            categorical_vocabulary_sizes=categorical_vocab_sizes,
            padding_idx=padding_idx,
        )

        model = model.to(device)
        logger.info(model)

        # Lightning
        loss = LOSSES[cfg_dict["model"]["train_params"]["loss_name"]]().to(device)
        optimizer = OPTIMIZERS[
            cfg_dict["model"]["train_params"]["optimizer_name"]
        ]  # without the () !
        scheduler = SCHEDULERS[cfg_dict["model"]["train_params"]["scheduler_name"]]

        module = MODULES[cfg_dict["model"]["model_name"]](
            model=model,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            **cfg_dict["model"]["train_params"],
        )
        logger.info(module)

        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param("num_trainable_parameters", num_trainable)

        ###### Trainer #####
        trainer = TRAINERS[cfg_dict["model"]["train_params"]["trainer_name"]](
            **cfg_dict["model"]["train_params"],
        )

        if cfg_dict["model"]["preprocessor"] == "PyTorch":
            mlflow.pytorch.autolog()
            torch.cuda.empty_cache()
            torch.set_float32_matmul_precision("medium")

        trainer.fit(module, train_dataloader, val_dataloader)

        ########## Evaluation ##########

        logger.info("Starting evaluation...")

        zipped = zip([df_val, df_test], [val_dataloader, test_dataloader], ["val", "test"])

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
