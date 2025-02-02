import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from constants import DATA_GETTER, PREPROCESSORS, TOKENIZERS
from utils.data import get_df_naf, get_Y


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    ##### DATA #########
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Fetch data
    df_s3, df_s4 = DATA_GETTER[cfg_dict["data"]["sirene"]](**cfg_dict["data"])
    Y = get_Y(revision=cfg_dict["data"]["revision"])
    df_naf = get_df_naf(revision=cfg_dict["data"]["revision"])

    # Preprocess data
    preprocessor = PREPROCESSORS[cfg_dict["model"]["preprocessor"]]()

    # Debugging purposes only
    df_s4 = df_s4.sample(frac=0.001, random_state=42)
    df_s3 = df_s3.sample(frac=0.001, random_state=42)
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

    ###### Tokenizer ####
    tokenizer = TOKENIZERS[cfg_dict["tokenizer"]["name"]](
        **cfg_dict["tokenizer"], training_text=df_train[cfg_dict["data"]["text_feature"]].values
    )
    print(tokenizer)

    ###### Dataset #####

    ###### MODEL #####


if __name__ == "__main__":
    train()
