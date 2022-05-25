"""
Main script.
"""
import sys

import mlflow
import pandas as pd

from fasttext_classifier.fasttext_evaluator import FastTextEvaluator
from fasttext_classifier.fasttext_preprocessor import FastTextPreprocessor
from fasttext_classifier.fasttext_trainer import FastTextTrainer
from fasttext_classifier.fasttext_wrapper import FastTextWrapper


def main(
    remote_server_uri, experiment_name, run_name, data_path, dim, epoch, word_ngrams
):
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

        # Run training of the model
        model = trainer.train(df_train, "APE_NIV5", dim, epoch, word_ngrams)

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
        mlflow.log_param("dim", dim)
        mlflow.log_param("epoch", epoch)
        mlflow.log_param("word_ngrams", word_ngrams)

        # Evaluation
        evaluator = FastTextEvaluator(model)
        accuracies = evaluator.evaluate(df_test)

        # Log metrics
        for metric, value in accuracies.items():
            mlflow.log_metric(metric, value)

        # On training set
        train_accuracies = evaluator.evaluate(df_train)
        for metric, value in train_accuracies.items():
            mlflow.log_metric(metric + "_train", value)


if __name__ == "__main__":
    main(
        str(sys.argv[1]),
        str(sys.argv[2]),
        str(sys.argv[3]),
        str(sys.argv[4]),
        int(sys.argv[5]),
        int(sys.argv[6]),
        int(sys.argv[7]),
    )
