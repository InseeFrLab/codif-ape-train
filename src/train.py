"""
Main script.
"""
import os
import sys
import time

import mlflow
import pyarrow.parquet as pq

from constants import FRAMEWORK_CLASSES, TEXT_FEATURE
from fasttext_classifier.fasttext_wrapper import FastTextWrapper

# from tests.test_main import run_test
from utils.data import get_file_system


def main(
    remote_server_uri: str,
    experiment_name: str,
    run_name: str,
    Y: str,
    dim: str,
    lr: str,
    epoch: str,
    wordNgrams: str,
    minn: str,
    maxn: str,
    minCount: str,
    bucket: str,
    loss: str,
    label_prefix: str,
    categorical_features_1: str,
    categorical_features_2: str,
    categorical_features_3: str,
    categorical_features_4: str,
    model_type: str = "fasttext",
):
    """
    Main method.
    """
    params = {
        key: value
        for key, value in locals().items()
        if (key not in ["remote_server_uri", "experiment_name", "run_name", "Y", "model_type"])
        and not key.startswith("categorical_features")
    }
    params["thread"] = os.cpu_count()
    categorical_features = [
        value for key, value in locals().items() if key.startswith("categorical_features")
    ]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    fs = get_file_system()

    with mlflow.start_run(run_name=run_name):
        framework_classes = FRAMEWORK_CLASSES[model_type]

        preprocessor = framework_classes["preprocessor"]()
        trainer = framework_classes["trainer"]()

        print("\n\n*** 1- Preprocessing the database...\n")
        t = time.time()
        # Load data
        df = pq.read_table(
            "projet-ape/extractions/20240124_sirene4.parquet", filesystem=fs
        ).to_pandas()

        # Preprocess data
        df_train, df_test = preprocessor.preprocess(
            df=df,
            y=Y,
            text_feature=TEXT_FEATURE,
            categorical_features=categorical_features,
        )
        print(f"*** Done! Preprocessing lasted {round((time.time() - t)/60,1)} minutes.\n")

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
                "train_data": "train_text.txt",
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

        # # Evaluation
        # print("*** 3- Evaluating the model...\n")
        # t = time.time()
        # if model_type == "fasttext":
        #     evaluator = framework_classes["evaluator"](model)
        # elif model_type == "pytorch":
        #     evaluator = framework_classes["evaluator"](model, trainer.tokenizer)
        # else:
        #     raise KeyError("Model type is not valid.")

        # # Split Test and Guichet Unique
        # df_gu = df_test[df_test.index.str.startswith("J")]
        # df_test = df_test[~df_test.index.str.startswith("J")]

        # accuracies = evaluator.evaluate(df_test, Y, TEXT_FEATURE, categorical_features, 5)

        # # Log metrics
        # for metric, value in accuracies.items():
        #     mlflow.log_metric(metric, value)

        # # On guichet unique set
        # gu_accuracies = evaluator.evaluate(df_gu, Y, TEXT_FEATURE, categorical_features, 5)
        # for metric, value in gu_accuracies.items():
        #     mlflow.log_metric(metric + "_gu", value)

        # print(f"*** Done! Evaluation lasted {round((time.time() - t)/60,1)} minutes.\n")

        # # Tests
        # print("*** 4- Performing standard tests...\n")
        # t = time.time()
        # with open(get_root_path() / "src/tests/tests.yaml", "r", encoding="utf-8") as stream:
        #     tests = yaml.safe_load(stream)
        # for case in tests.keys():
        #     run_test(tests[case], preprocessor, evaluator)

        # print(f"*** Done! Tests lasted {round((time.time() - t)/60,1)} minutes.\n")


if __name__ == "__main__":
    main(
        str(sys.argv[1]),
        str(sys.argv[2]),
        str(sys.argv[3]),
        str(sys.argv[4]),
        str(sys.argv[5]),
        str(sys.argv[6]),
        str(sys.argv[7]),
        str(sys.argv[8]),
        str(sys.argv[9]),
        str(sys.argv[10]),
        str(sys.argv[10]),
        str(sys.argv[11]),
        str(sys.argv[12]),
        str(sys.argv[13]),
        str(sys.argv[14]),
        str(sys.argv[15]),
        str(sys.argv[16]),
        str(sys.argv[17]),
        str(sys.argv[18]),
        str(sys.argv[19]),
    )
