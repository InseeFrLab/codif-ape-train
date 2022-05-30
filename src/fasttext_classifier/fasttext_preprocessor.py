"""
"""
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask_ml.model_selection import train_test_split

from base.preprocessor import Preprocessor
from preprocess import clean_lib


class FastTextPreprocessor(Preprocessor):
    """ """

    def __init__(self):
        """ """
        super().__init__()

    def preprocess_for_model(self, ddf, y_name, X_names):
        """
        Preprocess the input data and return train and test datasets
        """

        ddf = ddf.rename(columns={"APE_SICORE": "APE_NIV5"})
        # On se restreint à nos deux variables d'intérêt
        ddf = ddf[y_name + X_names]

        # On définit les valeurs manquantes comme des NaN
        ddf = ddf.fillna(value=np.nan)

        # On supprime les valeurs manquantes
        ddf = ddf.dropna()

        ddf["LIB_CLEAN"] = ddf["LIB_SICORE"].apply(
            lambda x: clean_lib(x), meta=pd.Series(dtype="str", name="LIB_CLEAN")
        )

        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            ddf["LIB_CLEAN"],
            ddf["APE_NIV5"],
            test_size=0.2,
            random_state=0,
            shuffle=True,
        )
        ddf_train = dd.concat([X_train, y_train], axis=1, ignore_unknown_divisions=True)
        ddf_test = dd.concat([X_test, y_test], axis=1, ignore_unknown_divisions=True)

        return ddf_train, ddf_test
