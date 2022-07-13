"""
FastTextPreprocessor class.
"""
import string
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from base.preprocessor import Preprocessor


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
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses data to feed to a classifier of the
        fasttext library for training and evaluation.

        Args:
            df (pd.DataFrame): Text descriptions to classify.
            y (str): Name of the variable to predict.
            text_feature (str): Name of the text feature.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.

        Returns:
            pd.DataFrame: Preprocessed DataFrames for training,
            evaluation and "guichet unique"
        """
        df[text_feature] = [self.clean_lib(df, idx, text_feature) for idx in df.index]
        # Guichet unique split
        df_gu = df[df.index.str.startswith("J")]
        df = df[~df.index.str.startswith("J")]
        # Train/test split
        features = [text_feature]
        if categorical_features is not None:
            features += categorical_features
        X_train, X_test, y_train, y_test = train_test_split(
            df[features],
            df[y],
            test_size=0.2,
            random_state=0,
            shuffle=True,
        )
        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)

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

    def get_aggregated_APE(
        self,
        df: pd.DataFrame,
        y: str,
    ) -> pd.DataFrame:
        """
        Computes the underlying aggregated levels of the NAF classification
        for ground truth for pd.DataFrame `df`.

        Args:
            df (pd.DataFrame): DataFrame.
            y (str): Name of the variable to predict.

        Returns:
            DataFrame: Initial DataFrame including true values at each level
            of the NAF classification.
        """
        try:
            df_naf = pd.read_csv(r"./data/naf_extended.csv", dtype=str)
        except FileNotFoundError:
            df_naf = pd.read_csv(r"../data/naf_extended.csv", dtype=str)
        df_naf.set_index("NIV5", inplace=True, drop=False)

        df = df.rename(columns={"APE_SICORE": "APE_NIV5"})
        res = pd.DataFrame(
            {
                "APE_NIV" + str(level): df[y].str[:level].to_list()
                for level in range(2, 5)
            }
        )

        # Determine the most aggregated classification
        res["APE_NIV1"] = res["APE_NIV2"]
        for naf2 in pd.unique(df_naf["NIV2"]):
            res["APE_NIV1"][res["APE_NIV2"] == naf2] = df_naf["NIV1"][
                (df_naf["NIV2"] == naf2).argmax()
            ]

        res = res.set_axis(df.index)

        return df.join(res)
