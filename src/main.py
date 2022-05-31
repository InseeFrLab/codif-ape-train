"""
Main script.
"""
import sys

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from fasttext_classifier.fasttext_evaluator import FastTextEvaluator
from fasttext_classifier.fasttext_preprocessor import FastTextPreprocessor
from fasttext_classifier.fasttext_trainer import FastTextTrainer
from fasttext_classifier.fasttext_wrapper import FastTextWrapper

Preprocessor = FastTextPreprocessor()
Trainer = FastTextTrainer()
Evaluator = FastTextEvaluator()


def main(
    remote_server_uri, experiment_name, run_name, data_url, dim, epoch, wordNgrams
):
    """
    Main method.
    """
    client = MlflowClient(tracking_uri=remote_server_uri)
    experiments = {
        experiment.name: experiment.experiment_id
        for experiment in client.list_experiments()
    }

    experiment_id = ""
    for existing_name, existing_id in experiments.items():
        if experiment_name == existing_name:
            experiment_id = existing_id
            break
    if experiment_id == "":
        experiment_id = client.create_experiment(experiment_name)
    run = client.create_run(experiment_id)
    run_id = run.info.run_id

    with mlflow.start_run(run_id):
        # load data, assumed to be stored in a .parquet file
        df = pd.read_parquet(data_url, engine="pyarrow")

        # Preprocess data
        df_train, df_test = Preprocessor.preprocess_for_model(
            df, ["APE_NIV5"], ["LIB_SICORE"]
        )

        # Run training of the model
        model = Trainer.train(df_train, ["APE_NIV5"], dim, epoch, wordNgrams)

        fasttext_model_path = run_name + ".bin"
        model.save_model(fasttext_model_path)

        artifacts = {"fasttext_model_path": fasttext_model_path}
        mlflow_pyfunc_model_path = run_name

        mlflow.pyfunc.log_model(
            artifact_path=mlflow_pyfunc_model_path,
            python_model=FastTextWrapper(),
            artifacts=artifacts,
        )

        # Run training of the model
        df_train, df_test = Evaluator.evaluate(df_train, df_test, model)

        # calculate accuracy on test data
        accuracy_test = sum(df_test["GoodPREDICTION"]) / df_test.shape[0] * 100
        # calculate accuracy on train data
        accuracy_train = sum(df_train["GoodPREDICTION"]) / df_train.shape[0] * 100

        # log parameters
        mlflow.log_param("dim", dim)
        mlflow.log_param("epoch", epoch)
        mlflow.log_param("wordNgrams", wordNgrams)

        # log metrics
        mlflow.log_metric("model_accuracy_test", accuracy_test)
        mlflow.log_metric("model_accuracy_train", accuracy_train)


if __name__ == "__main__":

    remote_server_uri = (
        str(sys.argv[1]) if len(sys.argv) > 1 else "https://mlflow.lab.sspcloud.fr/"
    )
    experiment_name = str(sys.argv[2]) if len(sys.argv) > 2 else "test"
    run_name = str(sys.argv[3]) if len(sys.argv) > 3 else "default"
    data_url = (
        str(sys.argv[4])
        if len(sys.argv) > 4
        else "../data/extraction_sirene_sample.parquet"
    )

    dim = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    epoch = int(sys.argv[6]) if len(sys.argv) > 6 else 5
    wordNgrams = int(sys.argv[7]) if len(sys.argv) > 7 else 3

    main(remote_server_uri, experiment_name, run_name, data_url, dim, epoch, wordNgrams)
