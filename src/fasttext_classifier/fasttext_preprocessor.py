"""
FastTextPreprocessor class.
"""
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from base.preprocessor import Preprocessor

pd.options.mode.chained_assignment = None


class FastTextPreprocessor(Preprocessor):
    """
    FastTextPreprocessor class.
    """

    def preprocess_for_model(
        self,
        df: pd.DataFrame,
        y: str,
        text_feature: str,
        categorical_features: Optional[List[str]] = None,
        oversampling: Optional[Dict[str, int]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses data to feed to a classifier of the
        fasttext library for training and evaluation.

        Args:
            df (pd.DataFrame): Text descriptions to classify.
            y (str): Name of the variable to predict.
            text_feature (str): Name of the text feature.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.
            oversampling (Optional[List[str]]): Parameters for oversampling
        Returns:
            pd.DataFrame: Preprocessed DataFrames for training,
            evaluation and "guichet unique"
        """
        df = self.clean_lib(df, text_feature)

        # Guichet unique split
        df_gu = df[df.index.str.startswith("J")]
        df = df[~df.index.str.startswith("J")]
        # Train/test split
        features = [text_feature]
        if categorical_features is not None:
            features += categorical_features

        df_train, df_test = self.train_test_split_by_class(
            df, y, features, test_size=0.2, random_state=0, shuffle=True
        )

        if oversampling is not None:
            print("\t*** Oversampling the train database...\n")
            t = time.time()
            df_train = self.oversample_df(df_train, oversampling["threshold"], y)
            print(
                f"\t*** Done! Oversampling lasted {round(time.time() - t,1)} seconds.\n"
            )

        return df_train, df_test, df_gu

    def clean_lib(self, df: pd.DataFrame, text_feature: str) -> pd.DataFrame:
        """
        Cleans a text feature for pd.DataFrame `df` at index idx.

        Args:
            df (pd.DataFrame): DataFrame.
            text_feature (str): Name of the text feature.

        Returns:
            df (pd.DataFrame): DataFrame.
        """
        # On définit 2 Regex de mots à supprimer du jeu de données
        LongWord2remove = r"\bconforme au kbis\b|\bsans changement\b|\bsans acitivite\b|\bactivite inchangee\b|\bactivites inchangees\b|\bsiege social\b|\ba definir\b|\ba preciser\b|\bci dessus\b|\bci desus\b|\bvoir activit principale\b|\bvoir activite principale\b|\bvoir objet social\b|\bidem extrait kbis\b|\bidem cadre precedent\b|\bn a plus a etre mentionne sur l extrait decret\b|\bcf statuts\b|\bactivite principale case\b|\bactivites principales case\b|\bactivite principale\b|\bactivites principales\b|\bidem case\b|\bvoir case\b|\baucun changement\b|\bsans modification\b|\bactivite non modifiee\b"
        Word2remove = r"\bcode\b|\bcadre\b|\bape\b|\bape[a-z]{1}\b|\bnaf\b|\binchangee\b|\binchnagee\b|\bkbis\b|\bk bis\b|\binchangees\b|\bnp\b|\binchange\b|\bnc\b|\bidem\b|\bxx\b|\bxxx\b"

        # On passe tout en minuscule
        df[text_feature] = df[text_feature].map(str.lower)

        # On supprime toutes les ponctuations
        df[text_feature] = df[text_feature].replace(
            to_replace=r"[^\w\s]", value=" ", regex=True
        )

        # On supprime tous les chiffres
        df[text_feature] = df[text_feature].replace(
            to_replace=r"[\d+]", value=" ", regex=True
        )

        # On supprime les longs mots sans sens
        df[text_feature] = df[text_feature].replace(
            to_replace=LongWord2remove, value="", regex=True
        )

        # On supprime les mots courts sans sens
        df[text_feature] = df[text_feature].replace(
            to_replace=Word2remove, value="", regex=True
        )

        # On supprime les mots d'une seule lettre
        df[text_feature] = df[text_feature].replace(
            to_replace=r"\b[a-z]{1}\b", value="", regex=True
        )

        # On supprime les multiple space
        df[text_feature] = df[text_feature].replace(r"\s\s+", " ", regex=True)

        # On strip les libellés
        df[text_feature] = df[text_feature].str.strip()

        # On remplace les empty string par des NaN
        df[text_feature] = df[text_feature].replace(r"^\s*$", np.nan, regex=True)

        # On supprime les NaN
        df = df.dropna(subset=[text_feature])

        # On tokenize tous les libellés
        libs_token = [lib.split() for lib in df[text_feature].to_list()]

        # On supprime les mots duppliqué dans un même libellé
        libs_token = [
            sorted(set(libs_token[i]), key=libs_token[i].index)
            for i in range(len(libs_token))
        ]

        # Pour chaque libellé on supprime les stopword et on racinise les mots
        df[text_feature] = [
            " ".join(
                [
                    self.stemmer.stem(word)
                    for word in libs_token[i]
                    if word not in self.stopwords
                ]
            )
            for i in range(len(libs_token))
        ]

        return df

    def oversample_df(self, df: pd.DataFrame, threshold: int, Y: str):
        Code2Oversample = df.value_counts(Y)[
            df.value_counts(Y) < threshold
        ].index.to_list()
        df_oversampled = pd.DataFrame(columns=df.columns)

        for aCode in Code2Oversample:
            Nb2sample = threshold - df[df[Y] == aCode].shape[0]
            df_oversampled = pd.concat(
                [df_oversampled, df[df[Y] == aCode].sample(n=Nb2sample, replace=True)]
            )

        return pd.concat([df, df_oversampled])

    def train_test_split_by_class(
        self,
        df: pd.DataFrame,
        y: str,
        features: List,
        test_size: float,
        random_state: int,
        shuffle: bool,
    ):

        df_train = pd.DataFrame(
            columns=[y]
            + features
            + [f"APE_NIV{i}" for i in range(1, 6) if f"{i}" not in [y[-1]]]
        )
        df_test = pd.DataFrame(columns=df_train.columns)
        Code2Split = set(df[y])

        for aCode in Code2Split:
            df_chunk = df[df[y] == aCode]
            if df_chunk.shape[0] == 1:
                df_train_chunk = df_chunk[
                    [y]
                    + features
                    + [f"APE_NIV{i}" for i in range(1, 6) if f"{i}" not in [y[-1]]]
                ]
                df_test_chunk = pd.DataFrame(columns=df_train_chunk.columns)
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    df_chunk[
                        features
                        + [f"APE_NIV{i}" for i in range(1, 6) if str(i) not in [y[-1]]]
                    ],
                    df_chunk[y],
                    test_size=test_size,
                    random_state=random_state,
                    shuffle=shuffle,
                )
                df_train_chunk = pd.concat([X_train, y_train], axis=1)
                df_test_chunk = pd.concat([X_test, y_test], axis=1)
            df_train = pd.concat([df_train, df_train_chunk])
            df_test = pd.concat([df_test, df_test_chunk])

        return df_train, df_test
