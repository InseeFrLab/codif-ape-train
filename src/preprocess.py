import string

import numpy as np
import pandas as pd
from dask_ml.model_selection import train_test_split
from nltk.corpus import stopwords

stopwords_ = set(stopwords.words("french") + ["a"])


def clean_lib(lib):
    # On supprime toutes les ponctuations
    lib = lib.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )
    # On supprime tous les chiffres
    lib = lib.translate(str.maketrans(string.digits, " " * len(string.digits)))

    # On supprime les stopwords et on renvoie les mots en majuscule
    return " ".join([x.lower() for x in lib.split() if x.lower() not in stopwords_])


def run_preprocessing(ddf):
    """
    Preprocess the input data
    """
    # On se restreint à nos deux variables d'intérêt
    ddf = ddf[["APE_SICORE", "LIB_SICORE"]]

    # On définit les valeurs manquantes comme des NaN
    ddf = ddf.fillna(value=np.nan)

    # On supprime les valeurs manquantes
    ddf = ddf.dropna()

    ddf["LIB_CLEAN"] = ddf["LIB_SICORE"].apply(
        lambda x: clean_lib(x), meta=pd.Series(dtype="str", name="LIB_CLEAN")
    )

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        ddf["LIB_CLEAN"], ddf["APE_SICORE"], test_size=0.2, random_state=0
    )
    return X_train, X_test, y_train, y_test
