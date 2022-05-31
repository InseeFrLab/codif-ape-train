"""
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from base.preprocessor import Preprocessor
from preprocess import clean_lib


class FastTextPreprocessor(Preprocessor):
    """ """

    def __init__(self):
        """ """
        super().__init__()

    def preprocess_for_model(self, df, y_name, X_names):
        """
        Preprocess the input data and return train and test datasets
        """

        df = df.rename(columns={"APE_SICORE": "APE_NIV5"})
        # On se restreint à nos deux variables d'intérêt
        df = df[y_name + X_names]

        # On définit les valeurs manquantes comme des NaN
        df = df.fillna(value=np.nan)

        # On supprime les valeurs manquantes
        df = df.dropna(subset=y_name + [X_names[0]])

        df["LIB_CLEAN"] = [clean_lib(idx, df, X_names) for idx in df.index]

        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            df["LIB_CLEAN"],
            df["APE_NIV5"],
            test_size=0.2,
            random_state=0,
            shuffle=True,
        )
        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)

        return df_train, df_test
