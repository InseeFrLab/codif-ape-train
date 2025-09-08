import os

import mlflow
import yaml
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf

from api_wrapper.mlflow_wrapper import MLFlowPyTorchWrapper
from pre_tokenizers import PreTokenizer

from .data import CATEGORICAL_FEATURES, COL_RENAMING, TEXT_FEATURE, TEXTUAL_FEATURES, get_df_naf
from .evaluation import get_inv_mapping


def create_or_restore_experiment(experiment_name):
    client = MlflowClient()

    try:
        # Check if the experiment exists (either active or deleted)
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment:
            if experiment.lifecycle_stage == "deleted":
                # Restore the experiment if it's deleted
                print(
                    f"Restoring deleted experiment: '{experiment_name}' (ID: {experiment.experiment_id})"
                )
                client.restore_experiment(experiment.experiment_id)
            else:
                print(
                    f"Experiment '{experiment_name}' already exists and is active (ID: {experiment.experiment_id})."
                )
        else:
            # Create the experiment if it doesn't exist
            print(f"Creating a new experiment: '{experiment_name}'")
            experiment_id = client.create_experiment(experiment_name)
            print(f"Created experiment '{experiment_name}' with ID: {experiment_id}")

    except RestException as e:
        print(f"An error occurred while handling the experiment '{experiment_name}': {e}")


def mlflow_log_model(logging_type, model, artifact_path):
    if logging_type == "model":
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
        )


def log_dict(cfg_dict):
    for key, value in cfg_dict.items():
        if isinstance(value, dict):
            log_dict(value)
        else:
            if key == "_target_":
                continue
            else:
                mlflow.log_param(key, value)
    return


def log_hydra_config(cfg, filename="hydra_config.yaml", save_dir=None):
    """
    Save and log the Hydra config to MLflow as an artifact.

    Args:
        cfg (omegaconf.DictConfig): Hydra config object.
        filename (str): Name of the YAML file to save.
        save_dir (str or None): Directory to save the config file in.
                                If None, uses current working directory.
    """
    if save_dir is None:
        save_dir = os.getcwd()

    os.makedirs(save_dir, exist_ok=True)
    config_save_path = os.path.join(save_dir, filename)

    OmegaConf.save(config=cfg, f=config_save_path)
    mlflow.log_artifact(config_save_path)


def load_module_and_config(run_id):
    logged_model = f"runs:/{run_id}/model"
    module = mlflow.pytorch.load_model(logged_model)
    # Download the artifact directory (e.g., to a temp dir)
    local_artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id)

    # Load the YAML config
    with open(f"{local_artifacts_path}/hydra_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    return module, config


def init_and_log_wrapper(cfg, logged_pth_path, pre_tokenizer: PreTokenizer):
    df_naf = get_df_naf(revision=cfg.revision)
    ape_to_lib = dict(df_naf[["APE_NIV5", "LIB_NIV5"]].drop_duplicates().values)
    inv_mapping = get_inv_mapping(cfg.revision)

    mlflow_wrapper = MLFlowPyTorchWrapper(
        libs=ape_to_lib,
        inv_mapping=inv_mapping,
        text_feature=TEXT_FEATURE,
        categorical_features=CATEGORICAL_FEATURES,
        textual_features=TEXTUAL_FEATURES,
        col_renaming=COL_RENAMING,
        pre_tokenizer=pre_tokenizer,
    )

    input_example = mlflow_wrapper._get_input_data_example()

    mlflow.pyfunc.log_model(
        artifact_path="pyfunc_model",
        python_model=mlflow_wrapper,
        input_example=input_example,
        artifacts={"torch_model_path": logged_pth_path, "nltk_data": "nltk_data"},
        code_paths=["src/api_wrapper/", "src/pre_tokenizers/", "src/models/"],
    )

    return
