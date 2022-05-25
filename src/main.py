"""
Main script.
"""
import sys

import dask.dataframe as dd
import fasttext
import mlflow
import numpy as np
import pandas as pd
from dask_ml.model_selection import train_test_split

from fasttext_classifier.fasttext_wrapper import FastTextWrapper
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


def main(
    remote_server_uri, experiment_name, run_name, data_url, dim, epoch, wordNgrams
):
    """
    Main method.
    """
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        # load data, assumed to be stored in a .parquet file
        ddf = dd.read_parquet(data_url, engine="pyarrow")

        # Preprocess data
        ddf_train, ddf_test = run_preprocessing(ddf)

        # Run training of the model
        model = run_training(ddf_train, dim, epoch, wordNgrams)

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
        df_test, df_train = run_prediction(ddf_test, ddf_train, model)

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


def run_preprocessing(ddf):
    """
    Preprocess the input data
    """
    ddf = ddf.rename(columns={"APE_SICORE": "APE_NIV5"})
    # On se restreint à nos deux variables d'intérêt
    ddf = ddf[["APE_NIV5", "LIB_SICORE"]]

    # On définit les valeurs manquantes comme des NaN
    ddf = ddf.fillna(value=np.nan)

    # On supprime les valeurs manquantes
    ddf = ddf.dropna()

    ddf["LIB_CLEAN"] = ddf["LIB_SICORE"].apply(
        lambda x: clean_lib(x), meta=pd.Series(dtype="str", name="LIB_CLEAN")
    )

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        ddf["LIB_CLEAN"], ddf["APE_NIV5"], test_size=0.2, random_state=0
    )
    ddf_train = dd.concat([X_train, y_train], axis=1, ignore_unknown_divisions=True)
    ddf_test = dd.concat([X_test, y_test], axis=1, ignore_unknown_divisions=True)

    return ddf_train, ddf_test


def run_training(ddf_train, dim, epoch, wordNgrams):

    # train the model with training_data
    with open("data/train_text.txt", "w") as f:
        for item in ddf_train.iterrows():
            formatted_item = "__label__{} {}".format(
                item[1]["APE_NIV5"], item[1]["LIB_CLEAN"]
            )
            f.write("%s\n" % formatted_item)

    model = fasttext.train_supervised(
        "data/train_text.txt", dim=dim, epoch=epoch, wordNgrams=wordNgrams
    )
    return model


def run_prediction(ddf_test, ddf_train, mod):
    # go back to pandas because dask doesnt support double assignment
    df_test = ddf_test.compute()
    df_train = ddf_train.compute()
    # predict testing data
    df_test[["PREDICTION_NIV5", "PROBA"]] = (
        df_test["LIB_CLEAN"].apply(lambda x: get_pred(x, mod)).to_list()
    )
    df_test["GoodPREDICTION"] = df_test["APE_NIV5"] == df_test["PREDICTION_NIV5"]
    for i in range(2, 5):
        df_test["PREDICTION_NIV" + str(i)] = df_test["PREDICTION_NIV5"].str[:i]

    # predict training data
    df_train[["PREDICTION_NIV5", "PROBA"]] = (
        df_train["LIB_CLEAN"].apply(lambda x: get_pred(x, mod)).to_list()
    )
    df_train["GoodPREDICTION"] = df_train["APE_NIV5"] == df_train["PREDICTION_NIV5"]
    for i in range(2, 5):
        df_train["PREDICTION_NIV" + str(i)] = df_train["PREDICTION_NIV5"].str[:i]

    return df_test, df_train


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
