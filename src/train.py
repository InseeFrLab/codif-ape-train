import hydra
from omegaconf import DictConfig, OmegaConf

from constants import DATA_GETTER, PREPROCESSORS
from utils.data import get_df_naf, get_Y


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
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
    ############################

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
    print(df_train_s4.head())


if __name__ == "__main__":
    train()
