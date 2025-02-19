import logging
import sys
import uuid

import hydra
import mlflow
import pandas as pd
import torch
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
from utils.data import get_df_naf, get_processed_data, get_Y
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
    return df_train, df_val, df_test, Y


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    create_or_restore_experiment(cfg_dict["mlflow"]["experiment_name"])
    mlflow.set_experiment(cfg_dict["mlflow"]["experiment_name"])

    with mlflow.start_run():

        # Log config
        log_dict(cfg_dict)

        ##### Data #########
        Y = get_Y(revision=cfg_dict["data"]["revision"])
        df_train, df_val, df_test = get_processed_data()

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

        tokenizer = TOKENIZERS[cfg_dict["tokenizer"]["tokenizer_name"]](
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
                    **cfg_dict["model"]["training_params"]
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
            # In general, tokenizer.num_tokens == num_rows is a invariant
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

        model = model.to(device)
        logger.info(model)

        # Lightning
        loss = LOSSES[cfg_dict["model"]["training_params"]["loss_name"]]().to(device)
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

        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param("num_trainable_parameters", num_trainable)

        ###### Trainer #####
        trainer = TRAINERS[cfg_dict["model"]["training_params"]["trainer_name"]](
            **cfg_dict["model"]["training_params"],
            experiment_name=cfg_dict["mlflow"]["experiment_name"]
        )

        if cfg_dict["model"]["preprocessor"] == "PyTorch":
            mlflow.pytorch.autolog()
            torch.cuda.empty_cache()
            torch.set_float32_matmul_precision("medium")

        trainer.fit(module, train_dataloader, val_dataloader)

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
