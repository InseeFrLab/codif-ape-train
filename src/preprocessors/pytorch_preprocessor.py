"""
PytorchPreprocessor class.
"""

import string
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mappings import SURFACE_COLS, mappings

from .base import Preprocessor

try:
    import nltk

    nltk.data.path.append("nltk_data/")
    from nltk.corpus import stopwords as ntlk_stopwords
    from nltk.stem.snowball import SnowballStemmer

    HAS_NLTK = True

except ImportError:
    HAS_NLTK = False

try:
    import unidecode

    HAS_UNIDECODE = True
except ImportError:
    HAS_UNIDECODE = False


class PytorchPreprocessor(Preprocessor):
    """
    FastTextPreprocessor class.
    """

    @staticmethod
    def clean_textual_features(
        df: pd.DataFrame,
        textual_features: List[str],
    ) -> pd.DataFrame:
        """
        Cleans the other textual features for pd.DataFrame `df`.

        Args:
            df (pd.DataFrame): DataFrame.
            textual_features (List[str]): Names of the other textual features.
            method (str): The method when the function is used (training or
                evaluation).
            recase (bool): if True, try applying standard casing.

        Returns:
            df (pd.DataFrame): DataFrame.
        """
        for textual_feature in textual_features:
            df[textual_feature] = PytorchPreprocessor.clean_text_feature(
                df[textual_feature], remove_stop_words=True
            )
            df[textual_feature] = df[textual_feature].str.replace(
                "nan", ""
            )  # empty string instead of "nan" (nothing will be added to the libelle)
            df[textual_feature] = df[textual_feature].apply(
                lambda x: " " + x if x != "" else x
            )  # add a space before the text because it will be concatenated to the libelle

        return df

    @staticmethod
    def clean_categorical_features(
        df: pd.DataFrame, categorical_features: List[str], y: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Cleans the categorical features for pd.DataFrame `df`.

        Args:
            df (pd.DataFrame): DataFrame.
            categorical_features (List[str]): Names of the categorical features.
            y (str): Name of the variable to predict.

        Returns:
            df (pd.DataFrame): DataFrame.
        """
        for surface_col in SURFACE_COLS:
            if surface_col in categorical_features:
                df[surface_col] = df[surface_col].astype(float)
                df = PytorchPreprocessor.categorize_surface(df, surface_col)
        for variable in categorical_features:
            if variable not in SURFACE_COLS:  # Mapping already done for this variable
                if len(set(df[variable].unique()) - set(mappings[variable].keys())) > 0:
                    raise ValueError(
                        f"Missing values in mapping for {variable} ",
                        set(df[variable].unique()) - set(mappings[variable].keys()),
                    )
                df[variable] = df[variable].apply(mappings[variable].get)
        if y is not None:
            if len(set(df[y].unique()) - set(mappings[y].keys())) > 0:
                raise ValueError(
                    f"Missing values in mapping for {y}, ",
                    set(df[y].unique()) - set(mappings[y].keys()),
                )
            df[y] = df[y].apply(mappings[y].get)

        return df

    @staticmethod
    def clean_text_feature(text: list[str], remove_stop_words=True):
        """
        Cleans a text feature.

        Args:
            text (list[str]): List of text descriptions.
            remove_stop_words (bool): If True, remove stopwords.

        Returns:
            list[str]: List of cleaned text descriptions.

        """
        if not HAS_NLTK:
            raise ImportError(
                "nltk is not installed and is required for preprocessing. Run 'pip install torchFastText[preprocess]'."
            )
        if not HAS_UNIDECODE:
            raise ImportError(
                "unidecode is not installed and is required for preprocessing. Run 'pip install torchFastText[preprocess]'."
            )

        # Define stopwords and stemmer
        stopwords = tuple(ntlk_stopwords.words("french")) + tuple(string.ascii_lowercase)
        stemmer = SnowballStemmer(language="french")

        # Remove of accented characters
        text = np.vectorize(unidecode.unidecode)(np.array(text))

        # To lowercase
        text = np.char.lower(text)

        # Remove one letter words
        def mylambda(x):
            return " ".join([w for w in x.split() if len(w) > 1])

        text = np.vectorize(mylambda)(text)

        # Remove duplicate words and stopwords in texts
        # Stem words
        libs_token = [lib.split() for lib in text.tolist()]
        libs_token = [
            sorted(set(libs_token[i]), key=libs_token[i].index) for i in range(len(libs_token))
        ]
        if remove_stop_words:
            text = [
                " ".join([stemmer.stem(word) for word in libs_token[i] if word not in stopwords])
                for i in range(len(libs_token))
            ]
        else:
            text = [
                " ".join([stemmer.stem(word) for word in libs_token[i]])
                for i in range(len(libs_token))
            ]

        # Return clean DataFrame
        return text

    @staticmethod
    def categorize_surface(
        df: pd.DataFrame, surface_feature_name: str, like_sirene_3: bool = True
    ) -> pd.DataFrame:
        """
        Categorize the surface of the activity.

        Args:
            df (pd.DataFrame): DataFrame to categorize.
            surface_feature_name (str): Name of the surface feature.
            like_sirene_3 (bool): If True, categorize like Sirene 3.

        Returns:
            pd.DataFrame: DataFrame with a new column "surf_cat".
        """
        df_copy = df.copy()
        # Check surface feature exists
        if surface_feature_name not in df.columns:
            raise ValueError(f"Surface feature {surface_feature_name} not found in DataFrame.")
        # Check surface feature is a float variable
        if not (pd.api.types.is_float_dtype(df[surface_feature_name])):
            raise ValueError(f"Surface feature {surface_feature_name} must be a float variable.")

        if like_sirene_3:
            # Categorize the surface
            df_copy["surf_cat"] = pd.cut(
                df_copy[surface_feature_name],
                bins=[0, 120, 400, 2500, np.inf],
                labels=["1", "2", "3", "4"],
            ).astype(str)
        else:
            # Log transform the surface
            df_copy["surf_log"] = np.log(df[surface_feature_name])

            # Categorize the surface
            df_copy["surf_cat"] = pd.cut(
                df_copy.surf_log,
                bins=[0, 3, 4, 5, 12],
                labels=["1", "2", "3", "4"],
            ).astype(str)

        df_copy[surface_feature_name] = df_copy["surf_cat"].replace("nan", "0")
        df_copy[surface_feature_name] = df_copy[surface_feature_name].astype(int)
        df_copy = df_copy.drop(columns=["surf_log", "surf_cat"], errors="ignore")
        return df_copy

    def preprocess_for_model(
        self,
        df: pd.DataFrame,
        y: str,
        text_feature: str,
        textual_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        test_size: float = 0.2,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses data to feed to a Pytorch classifier.

        Args:
            df (pd.DataFrame): Raw text descriptions to classify and relevant variables.
            df_naf (pd.DataFrame): Dataframe that contains all codes and libs.
            y (str): Name of the variable to predict.
            text_feature (str): Name of the text feature.
            textual_features (List[str]): Names of the other textual features.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.
            oversampling (Optional[List[str]]): Parameters for oversampling.
            test_size (float): Size of the test set.
            recase (bool): if True, try applying standard casing.
            add_codes (bool): Whether to add missing APE codes. Unused.

        Returns:
            pd.DataFrame: Preprocessed DataFrames for training,
            evaluation and "guichet unique"
        """

        # Clean text + textual features, add textual features at the end of the text (libelle)
        df[text_feature] = self.clean_text_feature(df[text_feature], remove_stop_words=True)
        df = self.clean_textual_features(df, textual_features)
        df[text_feature] = df[text_feature] + df[textual_features].apply(
            lambda x: "".join(x), axis=1
        )

        # Clean categorical features
        df = self.clean_categorical_features(df, categorical_features=categorical_features, y=y)

        num_classes = df[
            y
        ].nunique()  # we are sure, after the "oversampling" (adding the true labels), that each label is present at least once

        # isolate the added "true" labels (code libelles): we will add them to the training set
        oversampled_labels = df.iloc[-num_classes:]
        df = df.iloc[:-num_classes]

        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(
                columns=[y, *textual_features]
            ),  # drop the textual additional var as they are already concatenated to the libelle
            df[y],
            test_size=test_size,
            random_state=0,
            shuffle=True,
        )

        df_test = pd.concat([X_test, y_test], axis=1)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=test_size,
            random_state=0,
            shuffle=True,
        )

        # Adding the true labels to the training set
        X_train = pd.concat(
            [X_train, oversampled_labels.drop(columns=[y, *textual_features])], axis=0
        )
        y_train = pd.concat([y_train, oversampled_labels[y]], axis=0)

        df_train = pd.concat([X_train, y_train], axis=1)
        df_val = pd.concat([X_val, y_val], axis=1)

        return df_train, df_val, df_test
