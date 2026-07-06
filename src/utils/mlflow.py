import os
import shutil
from typing import List

import mlflow
import mlflow.data
import yaml
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf

from src.api_wrapper.mlflow_wrapper import MLFlowPyTorchWrapper

from .data import (
    CATEGORICAL_FEATURES,
    COL_RENAMING,
    SURFACE_COLS,
    TEXT_FEATURE,
    TEXTUAL_FEATURES,
    get_df_naf,
    get_split_path,
)


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


def log_dataset_inputs(data_module, revision):
    """
    Log train/val/test splits as MLflow dataset inputs on the active run.

    This gives each run a versioned reference (schema + content digest + S3
    source path) to the exact data it was trained/evaluated on, visible in
    the MLflow UI's "Datasets" tab and searchable via `dataset.digest`.
    """
    split_path = get_split_path(revision)
    splits = {
        "df_train.parquet": (data_module.df_train, "train"),
        "df_val.parquet": (data_module.df_val, "eval"),
        "df_test.parquet": (data_module.df_test, "eval"),
    }

    for filename, (df, context) in splits.items():
        dataset = mlflow.data.from_pandas(
            df,
            source=split_path + filename,
            name=f"{revision}-{filename.removesuffix('.parquet')}",
            targets=data_module.Y,
        )
        mlflow.log_input(dataset, context=context)


def load_module_and_config(run_id):
    logged_model = f"runs:/{run_id}/model"
    module = mlflow.pytorch.load_model(logged_model)
    # Download the artifact directory (e.g., to a temp dir)
    local_artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id)

    # Load the YAML config
    with open(f"{local_artifacts_path}/hydra_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    return module, config


def mlflow_code_packaging(subfolders: List[str] = None) -> str:
    """
    Prepares a 'src/' directory structure in 'mlflow_staging' for MLflow 3 artifact logging.
    Fixes code_paths import flattening issues by preserving the parent 'src' folder.
    The goal is to log only the minimum necessary and not all the elements from src/
    """
    if subfolders is None:
        subfolders = ["api_wrapper"]

    staging_dir = "mlflow_staging"
    target_src_dir = os.path.join(staging_dir, "src")

    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)

    os.makedirs(target_src_dir, exist_ok=True)

    for folder in subfolders:
        source_path = os.path.join("src", folder)
        destination_path = os.path.join(target_src_dir, folder)

        if os.path.exists(source_path):
            shutil.copytree(source_path, destination_path)
        else:
            print(f"⚠️ Warning: Source folder '{source_path}' not found.")

    return target_src_dir


def init_and_log_wrapper(cfg):
    df_naf = get_df_naf(revision=cfg.revision)
    ape_to_lib = dict(df_naf[["APE_NIV5", "LIB_NIV5"]].drop_duplicates().values)

    code_paths_mlflow_artifacts = mlflow_code_packaging(["api_wrapper"])

    mlflow_wrapper = MLFlowPyTorchWrapper(
        libs=ape_to_lib,
        text_feature=TEXT_FEATURE,
        categorical_features=CATEGORICAL_FEATURES,
        textual_features=TEXTUAL_FEATURES,
        surface_cols=SURFACE_COLS,
        col_renaming=COL_RENAMING,
    )

    input_example = mlflow_wrapper._get_input_data_example()

    run_id = mlflow.active_run().info.run_id
    hydra_config_path = f"runs:/{run_id}/hydra_config.yaml"
    ttc_model_path = f"runs:/{run_id}/ttc_model"

    mlflow.pyfunc.log_model(
        name="pyfunc_model",
        python_model=mlflow_wrapper,
        input_example=input_example,
        artifacts={
            "ttc_model_path": ttc_model_path,
            "hydra_config": hydra_config_path,
        },
        code_paths=[code_paths_mlflow_artifacts],
    )

    return
