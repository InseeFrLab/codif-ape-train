import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from constants import (
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


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    ##### Data #########

    # Fetch data
    df_s3, df_s4 = DATA_GETTER[cfg_dict["data"]["sirene"]](**cfg_dict["data"])
    Y = get_Y(revision=cfg_dict["data"]["revision"])
    df_naf = get_df_naf(revision=cfg_dict["data"]["revision"])

    # Preprocess data
    preprocessor = PREPROCESSORS[cfg_dict["model"]["preprocessor"]]()

    # Debugging purposes only
    df_s4 = df_s4.sample(frac=0.0001, random_state=1)
    df_s3 = df_s3.sample(frac=0.0001, random_state=1)
    ##########

    if df_s4 is not None:
        df_train_s4, df_val_s4, df_test = preprocessor.preprocess(
            df=df_s4,
            df_naf=df_naf,
            y=Y,
            text_feature=cfg_dict["data"]["text_feature"],
            textual_features=cfg_dict["data"]["textual_features"],
            categorical_features=cfg_dict["data"]["categorical_features"],
            test_size=0.1,
        )
    else:
        raise ValueError("Sirene 4 data should be provided.")

    if df_s3 is not None:
        df_train_s3, df_val_s3, df_test_s3 = preprocessor.preprocess(
            df=df_s3,
            df_naf=df_naf,
            y=Y,
            text_feature=cfg_dict["data"]["text_feature"],
            textual_features=cfg_dict["data"]["textual_features"],
            categorical_features=cfg_dict["data"]["categorical_features"],
            test_size=0.1,
            s3=True,
        )
        # all sirene 3 data used as train set, we eval/test only on sirene 4 data
        df_s3_processed = pd.concat([df_train_s3, df_val_s3, df_test_s3])
        df_train = pd.concat([df_s3_processed, df_train_s4]).reset_index(drop=True)

        # Assert we have not lost data in the process
        assert len(df_s3) == len(df_s3_processed)
        assert len(df_train_s4) + len(df_s3) == len(df_train)

    else:
        df_train = df_train_s4

    df_val = df_val_s4

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
    print(tokenizer)

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
            val_dataloader = val_dataset.create_dataloader(**cfg_dict["model"]["training_params"])

    ###### Model #####

    num_classes = int(np.max(df_train[Y].values) + 1)
    categorical_vocab_sizes = np.max(train_categorical_variables, axis=0) + 1
    categorical_vocab_sizes = categorical_vocab_sizes.astype(int).tolist()
    print("num classes:", num_classes)
    print("categorical_vocab_sizes ", categorical_vocab_sizes)

    if cfg_dict["model"]["name"] == "torchFastText":
        # for torchFastText only, we add the number of words in the vocabulary
        # In general, tokenizer.num_tokens == num_orws is a invariant
        num_rows = tokenizer.num_tokens + tokenizer.get_nwords() + 1
        padding_idx = num_rows - 1

    # PyTorch model
    model = MODELS[cfg_dict["model"]["name"]](
        **cfg_dict["model"]["model_params"],
        tokenizer=tokenizer,
        num_tokens=num_rows,
        num_classes=num_classes,
        categorical_vocabulary_sizes=categorical_vocab_sizes,
        padding_idx=padding_idx,
    )
    print(model)

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
    print(module)

    ###### Trainer #####
    trainer = TRAINERS[cfg_dict["model"]["training_params"]["trainer_name"]](
        **cfg_dict["model"]["training_params"]
    )

    trainer.fit(module, train_dataloader, val_dataloader)


if __name__ == "__main__":
    train()
