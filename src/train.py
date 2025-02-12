import logging
import os
import sys

import hydra
import mlflow
import pandas as pd
import torch
from joblib import Memory
from omegaconf import DictConfig, OmegaConf

from framework_classes import (
    DATA_GETTER,
    DATASETS,
    LOSSES,
    MODELS,
    MODULES,
    OPTIMIZERS,
    PREPROCESSORS,
    SCHEDULERS,
    TOKENIZERS,
    TRAINERS,
)
from utils.data import get_df_naf, get_Y
from utils.mappings import mappings
from utils.mlflow import create_or_restore_experiment

memory = Memory(location="cache_dir", verbose=1)  # Set cache location

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)

print(os.cpu_count())


@memory.cache
def load_or_preprocess_data(cfg_dict_data, cfg_dict_model_preprocessor):
    """
    Load and preprocess data, using joblib caching to avoid redundant computation.
    """
    # Fetch data
    df_s3, df_s4 = DATA_GETTER[cfg_dict_data["sirene"]](**cfg_dict_data)
    Y = get_Y(revision=cfg_dict_data["revision"])
    df_naf = get_df_naf(revision=cfg_dict_data["revision"])

    # Preprocess data
    preprocessor = PREPROCESSORS[cfg_dict_model_preprocessor]()

    if df_s4 is not None:
        df_train_s4, df_val_s4, df_test = preprocessor.preprocess(
            df=df_s4,
            df_naf=df_naf,
            y=Y,
            text_feature=cfg_dict_data["text_feature"],
            textual_features=cfg_dict_data["textual_features"],
            categorical_features=cfg_dict_data["categorical_features"],
            test_size=0.1,
        )
    else:
        raise ValueError("Sirene 4 data should be provided.")

    if df_s3 is not None:
        df_train_s3, df_val_s3, df_test_s3 = preprocessor.preprocess(
            df=df_s3,
            df_naf=df_naf,
            y=Y,
            text_feature=cfg_dict_data["text_feature"],
            textual_features=cfg_dict_data["textual_features"],
            categorical_features=cfg_dict_data["categorical_features"],
            test_size=0.1,
            s3=True,
        )
        # Merge Sirene 3 into the training set
        df_s3_processed = pd.concat([df_train_s3, df_val_s3, df_test_s3])
        df_train = pd.concat([df_s3_processed, df_train_s4]).reset_index(drop=True)

        # Assert no data was lost
        assert len(df_s3) == len(df_s3_processed)
        assert len(df_train_s4) + len(df_s3) == len(df_train)

    else:
        df_train = df_train_s4

    df_val = df_val_s4
    return df_train, df_val, Y


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    mlflow.set_tracking_uri(cfg_dict["mlflow"]["remote_server_uri"])
    create_or_restore_experiment(cfg_dict["mlflow"]["experiment_name"])
    mlflow.set_experiment(cfg_dict["mlflow"]["experiment_name"])

    run_name = (
        cfg_dict["model"]["name"]
        + "_"
        + str(cfg_dict["model"]["model_params"]["embedding_dim"])
        + "_"
        + str(cfg_dict["tokenizer"]["num_tokens"])
    )

    logger.info("Run name: " + run_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("mlflow.runName", run_name)
        # Log config
        mlflow.log_params(cfg_dict)

        ##### Data #########

        df_train, df_val, Y = load_or_preprocess_data(
            cfg_dict["data"], cfg_dict["model"]["preprocessor"]
        )

        mlflow.log_param("number_of_training_observations", df_train.shape[0])

        train_text, train_categorical_variables = (
            df_train[cfg_dict["data"]["text_feature"]].values,
            df_train[cfg_dict["data"]["categorical_features"]].values,
        )
        val_text, val_categorical_variables = (
            df_val[cfg_dict["data"]["text_feature"]].values,
            df_val[cfg_dict["data"]["categorical_features"]].values,
        )

        ###### Tokenizer ######

        tokenizer = TOKENIZERS[cfg_dict["tokenizer"]["name"]](
            **cfg_dict["tokenizer"], training_text=train_text
        )
        logger.info(tokenizer)

        ###### Dataset ######

        if cfg_dict["model"]["dataset"] is not None:
            dataset_class = DATASETS[cfg_dict["model"]["dataset"]]

            train_dataset = dataset_class(
                texts=train_text,
                categorical_variables=train_categorical_variables,
                tokenizer=tokenizer,
                outputs=df_train[Y].values,
            )
            val_dataset = dataset_class(
                texts=val_text,
                categorical_variables=val_categorical_variables,
                tokenizer=tokenizer,
                outputs=df_val[Y].values,
            )

            if cfg_dict["model"]["dataset"] == "FastTextModelDataset":
                train_dataloader = train_dataset.create_dataloader(
                    **cfg_dict["model"]["training_params"], num_workers=os.cpu_count() - 1
                )
                val_dataloader = val_dataset.create_dataloader(
                    **cfg_dict["model"]["training_params"]
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

        if cfg_dict["model"]["name"] == "torchFastText":
            # for torchFastText only, we add the number of words in the vocabulary
            # In general, tokenizer.num_tokens == num_orws is a invariant
            num_rows = tokenizer.num_tokens + tokenizer.get_nwords() + 1
            padding_idx = num_rows - 1

        # PyTorch model
        model = MODELS[cfg_dict["model"]["name"]](
            **cfg_dict["model"]["model_params"],
            tokenizer=tokenizer,
            num_rows=num_rows,
            num_classes=num_classes,
            categorical_vocabulary_sizes=categorical_vocab_sizes,
            padding_idx=padding_idx,
        )
        logger.info(model)

        # Lightning
        loss = LOSSES[cfg_dict["model"]["training_params"]["loss_name"]]()
        optimizer = OPTIMIZERS[
            cfg_dict["model"]["training_params"]["optimizer_name"]
        ]  # without the () !
        scheduler = SCHEDULERS[cfg_dict["model"]["training_params"]["scheduler_name"]]

        module = MODULES[cfg_dict["model"]["name"]](
            model=model,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            **cfg_dict["model"]["training_params"],
        )
        logger.info(module)

        ###### Trainer #####
        trainer = TRAINERS[cfg_dict["model"]["training_params"]["trainer_name"]](
            **cfg_dict["model"]["training_params"]
        )

        if cfg_dict["model"]["preprocessor"] == "PyTorch":
            mlflow.pytorch.autolog()
            torch.cuda.empty_cache()
            torch.set_float32_matmul_precision("medium")

        trainer.fit(module, train_dataloader, val_dataloader)

        # Save model

        best_model = type(module).load_from_checkpoint(
            checkpoint_path=trainer.checkpoint_callback.best_model_path,
            model=module.model,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            **cfg_dict["model"]["training_params"],
        )
        mlflow.pytorch.log_model(
            pytorch_model=best_model,
            artifact_path=run_name,
            input_example=None,
        )

        ########## Evaluation ##########

        # To do


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
