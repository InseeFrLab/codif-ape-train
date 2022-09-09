"""
Main script.
"""
import sys
import time

import mlflow
import pandas as pd
import yaml

from constants import TEXT_FEATURE
from fasttext_classifier.fasttext_evaluator import FastTextEvaluator
from fasttext_classifier.fasttext_preprocessor import FastTextPreprocessor
from fasttext_classifier.fasttext_trainer import FastTextTrainer
from fasttext_classifier.fasttext_wrapper import FastTextWrapper
from utils import get_root_path


def main(remote_server_uri, experiment_name, run_name, data_path, config_path):
    """
    Main method.
    """
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        preprocessor = FastTextPreprocessor()
        trainer = FastTextTrainer()

        print("\n\n*** 1- Preprocessing the database...\n")
        t = time.time()
        # Load data, assumed to be stored in a .parquet file
        df = pd.read_parquet(data_path, engine="pyarrow")
        df = df.sample(frac=0.001)

        with open(get_root_path() / config_path, "r") as stream:
            config = yaml.safe_load(stream)
        params = config["params"]
        categorical_features = config["categorical_features"]
        Y = config["Y"][0]
        oversampling = config["oversampling"]

        # Preprocess data
        df_train, df_test, df_gu = preprocessor.preprocess(
            df=df,
            y=Y,
            text_feature=TEXT_FEATURE,
            categorical_features=categorical_features,
            oversampling=oversampling,
        )
        print(
            f"*** Done! Preprocessing lasted {round((time.time() - t)/60,1)} minutes.\n"
        )

        # Run training of the model
        print("*** 2- Training the model...\n")
        t = time.time()
        model = trainer.train(df_train, Y, TEXT_FEATURE, categorical_features, params)
        print(f"*** Done! Training lasted {round((time.time() - t)/60,1)} minutes.\n")

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
        mlflow.log_param("features", categorical_features)
        mlflow.log_param("Y", Y)

        # Evaluation
        print("*** 3- Evaluating the model...\n")
        t = time.time()
        evaluator = FastTextEvaluator(model)
        accuracies = evaluator.evaluate(
            df_test, Y, TEXT_FEATURE, categorical_features, 5
        )

        # Log metrics
        for metric, value in accuracies.items():
            mlflow.log_metric(metric, value)

        # On training set
        train_accuracies = evaluator.evaluate(
            df_train, Y, TEXT_FEATURE, categorical_features, 5
        )
        for metric, value in train_accuracies.items():
            mlflow.log_metric(metric + "_train", value)

        # On guichet unique set
        gu_accuracies = evaluator.evaluate(
            df_gu, Y, TEXT_FEATURE, categorical_features, 5
        )
        for metric, value in gu_accuracies.items():
            mlflow.log_metric(metric + "_gu", value)

        print(f"*** Done! Evaluation lasted {round((time.time() - t)/60,1)} minutes.\n")


if __name__ == "__main__":
    main(
        str(sys.argv[1]),
        str(sys.argv[2]),
        str(sys.argv[3]),
        str(sys.argv[4]),
        str(sys.argv[5]),
    )
