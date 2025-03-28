import mlflow
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient


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
            mlflow.log_param(key, value)
    return