"""
Main script.
"""
import sys

import fasttext
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from fast_text_wrapper import FastTextWrapper
from preprocess import clean_lib


def get_pred(lib: str, mod: fasttext.FastText):
    """
    Returns the prediction of model `mod` on text `lib`
    along with the output probability.

    Args:
        lib: Text description.
        mod: Model.

    Returns:
        List: List with the prediction and probability for the
            given text.
    """
    out = mod.predict(lib)
    pred = out[0][0].replace("__label__", "")
    prob = out[1][0]
    return [pred, prob]


def main(remote_server_uri, experiment_name, run_name, data_url, dim, epoch):
    """
    Main method.
    """
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        # load data, assumed to be stored in a .parquet file
        df = pd.read_parquet(data_url)
        df["LIB_CLEAN"] = df["LIB_SICORE"].apply(lambda x: clean_lib(x))

        # train/test split
        train, test = train_test_split(df, test_size=0.2)

        # train the model with training_data
        with open("../data/train_text.txt", "w") as f:
            for item in train.iterrows():
                formatted_item = "__label__{} {}".format(
                    item[1]["APE_SICORE"], item[1]["LIB_CLEAN"]
                )
                f.write("%s\n" % formatted_item)

        model = fasttext.train_supervised(
            "../data/train_text.txt", dim=dim, epoch=epoch
        )

        fasttext_model_path = run_name + ".bin"
        model.save_model(fasttext_model_path)

        artifacts = {"fasttext_model_path": fasttext_model_path}
        mlflow_pyfunc_model_path = run_name

        mlflow.pyfunc.log_model(
            artifact_path=mlflow_pyfunc_model_path,
            python_model=FastTextWrapper(),
            artifacts=artifacts,
        )

        # predict testing data
        test[["PREDICTION", "PROBA"]] = (
            test["LIB_CLEAN"].apply(lambda x: get_pred(x, model)).to_list()
        )

        # calculate accuracy
        accuracy = sum(test["APE_SICORE"] == test["PREDICTION"]) / test.shape[0] * 100

        # log parameters
        mlflow.log_param("dim", dim)
        mlflow.log_param("epoch", epoch)

        # log metrics
        mlflow.log_metric("model_accuracy", accuracy)


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

    main(remote_server_uri, experiment_name, run_name, data_url, dim, epoch)
