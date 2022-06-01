"""
Main script.
"""
import sys

import mlflow
import pandas as pd
import yaml

from fasttext_classifier.fasttext_evaluator import FastTextEvaluator
from fasttext_classifier.fasttext_preprocessor import FastTextPreprocessor
from fasttext_classifier.fasttext_trainer import FastTextTrainer
from fasttext_classifier.fasttext_wrapper import FastTextWrapper
from utils import get_root_path


def main(remote_server_uri, experiment_name, run_name, data_path):
    """
    Main method.
    """
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        preprocessor = FastTextPreprocessor()
        trainer = FastTextTrainer()

        # Load data, assumed to be stored in a .parquet file
        df = pd.read_parquet(data_path, engine="pyarrow")

        # Preprocess data
        df_train, df_test = preprocessor.preprocess(
            df=df, y="APE_NIV5", features=["LIB_SICORE"]
        )

        with open(get_root_path() / "config/config_fasttext.yaml", "r") as stream:
            config = yaml.safe_load(stream)
        params = config["params"]

        # Run training of the model
        model = trainer.train(df_train, "APE_NIV5", params)

        fasttext_model_path = run_name + ".bin"
        model.save_model(fasttext_model_path)

        artifacts = {"fasttext_model_path": fasttext_model_path}
        mlflow_pyfunc_model_path = run_name

        mlflow.pyfunc.log_model(
            artifact_path=mlflow_pyfunc_model_path,
            python_model=FastTextWrapper(),
            artifacts=artifacts,
        )

        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # Evaluation
        evaluator = FastTextEvaluator(model)
        accuracies, cmatrix = evaluator.evaluate(df_test)

        # Log metrics
        for metric, value in accuracies.items():
            mlflow.log_metric(metric, value)

        # On training set
        train_accuracies, train_cmatrix = evaluator.evaluate(df_train)
        for metric, value in train_accuracies.items():
            mlflow.log_metric(metric + "_train", value)

        # log confusion matrix
        mlflow.log_figure(cmatrix, "confusion_matrix.png")


if __name__ == "__main__":
    main(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]))
