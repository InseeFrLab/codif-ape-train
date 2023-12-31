"""
Main script.
"""
import sys
import time

import mlflow
import pandas as pd
import yaml

from constants import FRAMEWORK_CLASSES, TEXT_FEATURE
from fasttext_classifier.fasttext_wrapper import FastTextWrapper
from tests.test_main import run_test
from utils import get_root_path


def main(remote_server_uri, experiment_name, run_name, data_path, config_path):
    """
    Main method.
    """
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        with open(get_root_path() / config_path, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        model_type = config["model_type"]
        framework_classes = FRAMEWORK_CLASSES[model_type]

        preprocessor = framework_classes["preprocessor"]()
        trainer = framework_classes["trainer"]()

        print("\n\n*** 1- Preprocessing the database...\n")
        t = time.time()
        # Load data, assumed to be stored in a .parquet file
        df = pd.read_parquet(data_path, engine="pyarrow")

        params = config["params"]
        categorical_features = config["categorical_features"]
        Y = config["Y"][0]
        oversampling = config["oversampling"]

        # Preprocess data
        df_train, df_test = preprocessor.preprocess(
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

        if model_type == "fasttext":
            fasttext_model_path = run_name + ".bin"
            model.save_model(fasttext_model_path)

            artifacts = {
                "fasttext_model_path": fasttext_model_path,
                "config_path": config_path,
                "train_data": "data/train_text.txt",
            }

            mlflow.pyfunc.log_model(
                artifact_path=run_name,
                code_path=["src/fasttext_classifier/", "src/base/"],
                python_model=FastTextWrapper(),
                artifacts=artifacts,
            )
        elif model_type == "pytorch":
            mlflow.pytorch.log_model(pytorch_model=model, artifact_path=run_name)
        else:
            raise KeyError("Model type is not valid.")

        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param("features", categorical_features)
        mlflow.log_param("Y", Y)

        # Evaluation
        print("*** 3- Evaluating the model...\n")
        t = time.time()
        if model_type == "fasttext":
            evaluator = framework_classes["evaluator"](model)
        elif model_type == "pytorch":
            evaluator = framework_classes["evaluator"](model, trainer.tokenizer)
        else:
            raise KeyError("Model type is not valid.")

        # Split Test and Guichet Unique
        df_gu = df_test[df_test.index.str.startswith("J")]
        df_test = df_test[~df_test.index.str.startswith("J")]

        accuracies = evaluator.evaluate(
            df_test, Y, TEXT_FEATURE, categorical_features, 5
        )

        # Log metrics
        for metric, value in accuracies.items():
            mlflow.log_metric(metric, value)

        # On guichet unique set
        gu_accuracies = evaluator.evaluate(
            df_gu, Y, TEXT_FEATURE, categorical_features, 5
        )
        for metric, value in gu_accuracies.items():
            mlflow.log_metric(metric + "_gu", value)

        print(f"*** Done! Evaluation lasted {round((time.time() - t)/60,1)} minutes.\n")

        # Tests
        print("*** 4- Performing standard tests...\n")
        t = time.time()
        with open(
            get_root_path() / "src/tests/tests.yaml", "r", encoding="utf-8"
        ) as stream:
            tests = yaml.safe_load(stream)
        for case in tests.keys():
            run_test(tests[case], preprocessor, evaluator)

        print(f"*** Done! Tests lasted {round((time.time() - t)/60,1)} minutes.\n")


if __name__ == "__main__":
    main(
        str(sys.argv[1]),
        str(sys.argv[2]),
        str(sys.argv[3]),
        str(sys.argv[4]),
        str(sys.argv[5]),
    )
