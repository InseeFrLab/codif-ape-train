"""
FastTextPreprocessor class.
"""

import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import unidecode
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
        df_naf: pd.DataFrame,
        y: str,
        text_feature: str,
        textual_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        oversampling: Optional[Dict[str, int]] = None,
        test_size: float = 0.2,
        recase: bool = False,
        add_codes: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses data to feed to a classifier of the
        fasttext library for training and evaluation.

        Args:
            df (pd.DataFrame): Text descriptions to classify.
            df_naf (pd.DataFrame): Dataframe that contains all codes and libs.
            y (str): Name of the variable to predict.
            text_feature (str): Name of the text feature.
            textual_features (Optional[List[str]]): Names of the
                other textual features.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.
            oversampling (Optional[List[str]]): Parameters for oversampling.
            test_size (float): Size of the test set.
            recase (bool): if True, try applying standard casing.

        Returns:
            pd.DataFrame: Preprocessed DataFrames for training and
                evaluation.
        """

        textual_features_cleaned = [
            self.clean_lib(df[text].tolist()) for text in [text_feature] + textual_features
        ]
        df.loc[:, [text_feature] + textual_features] = list(
            zip(*textual_features_cleaned)
        )  # Transpose the list of lists

        df[text_feature] = df[[text_feature] + textual_features].apply(
            lambda row: " ".join(f"[{col}] {val}" for col, val in row.items() if val != ""), axis=1
        )

        # Train/test split
        features = [text_feature]
        if textual_features is not None:
            features += textual_features
        if categorical_features is not None:
            features += categorical_features

        X_train, X_test, y_train, y_test = train_test_split(
            df[features + [f"APE_NIV{i}" for i in range(1, 6)]],
            df[y],
            test_size=test_size,
            random_state=0,
            shuffle=True,
        )

        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)

        # Adding missing APE codes in the train database by adding the official label as
        # text feature
        if add_codes:
            df_train = self.add_missing_codes(
                df_train, df_naf, y, text_feature, textual_features, categorical_features
            )

        if oversampling is not None:
            print("\t*** Oversampling the train database...\n")
            t = time.time()
            df_train = self.oversample_df(df_train, oversampling["threshold"], y)
            print(f"\t*** Done! Oversampling lasted {round((time.time() - t)/60,1)} minutes.\n")

        return df_train, df_test

    @staticmethod
    def clean_categorical_features(
        df: pd.DataFrame, y: str, categorical_features: List[str]
    ) -> pd.DataFrame:
        """
        Cleans the categorical features for pd.DataFrame `df`.

        Args:
            df (pd.DataFrame): DataFrame.
            y (str): Name of the variable to predict.
            categorical_features (List[str]): Names of the categorical features.

        Returns:
            df (pd.DataFrame): DataFrame.
        """
        return df

    def clean_lib(self, libs: list) -> list:
        """
        Cleans a list of text labels efficiently.

        Args:
            libs (list): List of text labels to clean.

        Returns:
            list: Cleaned text labels.
        """
        # Comprehensive replacement patterns
        replacement_patterns = [
            # Libellé vide de sens fournit par Christine (DOIT ETRE FAIT EN AMONT DU MODELE EN JAVA)
            (
                r"\bidem\b|\bvoir ci dessous\b|\[vide\]|\bundefined\b|\bpas d objet\b|\(voir ci dessus\)|\(voir extrait siege social\\/etablissement principal\)|\bcf activite principale\b|\bcf activite principale et objet\b|\bcf activites de l entreprise\b|\bcf activites principales de l entreprise\b|\bcf actvites principales\b|\bcf k bis\b|\bcf le principales activites de l  entreprise\b|\bcf le sprincipale activites de l  entreprise\b|\bcf le sprincipales activites de l  entreprise\b|\bcf les activites principales de l  entreprise\b|\bcf les ppales activites de l  entreprise\b|\bcf les ppales activites de la ste\b|\bcf les principale activites de l  entreprise\b|\bcf les principales activites\b|\bcf les principales activites de l  entreprise\b|\bcf les principales activites de l  entreprises\b|\bcf les principales activites ppales de l  entreprise\b|\bcf les principales activtes de l  entreprise\b|\bcf les principales acttivites de l  entreprise\b|\bcf les prinipales activites de l  entreprise\b|\bcf lesprincipales activites de l  entreprise\b|\bcf objet\b|\bcf obs\b|\bcf principales activite de l  entreprise\b|\bcf principales activites de l  entreprise\b|cf rubrique \"principales activites de l entreprise\" idem|cf rubrique n2 ci dessus \(743b\)|\bcf supra\b|\bcf ci  dessus\b|\bcommerce de detail, idem case 2\b|\bextension a: voir ci dessus\b|\bid\b|\bid principales activites\b|\bid principales activites de l  entreprise\b|\bidem ci dessus\b|idem \( voir principales activites\)|\bidem  dessus\b|\bidem 1ere page\b|\bidem a principales activites de l  entreprise\b|\bidem activiet eprincipale\b|\bidem activite\b|\bidem activite 1ere page\b|\bidem activite ci  dessus\b|\bidem activite de l  entreprise\b|\bidem activite enoncee ci  dessus\b|\bidem activite entreprise\b|\bidem activite generales\b|\bidem activite premiere page\b|\bidem activite principale\b|\bidem activite princippale\b|\bidem activite prinicpale\b|\bidem activite sur 1ere page\b|\bidem activites ci dessus\b|\bidem activites declarees au siege et principal\b|\bidem activites enoncees ci dessus\b|\bidem activites entreprise\b|\bidem activites principales\b|\bidem activites principales de l entreprise\b|\bidem activites siege\b|\bidem activte principale\b|\bidem activtie 1ere page\b|\bidem au siege\b|\bidem au siege social\b|\bidem aux principales actiivtes\b|\bidem aux principales activites\b|\bidem case 13\b|\bidem ci dessous\b|\bidem ci dessus enoncee\b|\bidem cidessus\b|\bidem objet\b|\bidem premiere page\b|\bidem pricincipales activites de l entreprise\b|\bidem pricipales activites\b|\bidem principale activite\b|\bidem principales activite de l entreprise\b|\bidem principales activite de l entreprises\b|\bidem principales activite l entreprise\b|\bidem principales activites\b|\bidem principales activites citees ci dessus\b|\bidem principales activites de l entreprises\b|idem principales activites de l entreprise\(objet\)|\bidem principales activites et objet social\b|\bidem principales activitse de l entreprise\b|\bidem que celle decrite plus haut\b|\bidem que ci dessus\b|\bidem que l activite decrite plus haut\b|\bidem que les activites principales\b|\bidem que les activites principales ci dessus\b|\bidem que les activitges principales\b|\bidem que les principales activites\b|\bidem que les principales activites de l entreprise\b|\bidem que pour le siege\b|\bidem rubrique principales activites de l entreprise\b|\bidem siege\b|idem siege \+ voir observation|\bidem siege et ets principal\b|\bidem siege social\b|idem siege, \(\+ articles americains\)|\bidem societe\b|\bidem voir activite principale\b|\bidem voir ci dessus\b|\bidentique a l objet social indique en case 2 de l imprime m2\b|\bidm ci dessus\b|\bnon indiquee\b|\bnon precise\b|\bnon precisee\b|\bnon precisees\b|\bvoir 1ere page\b|\bvoir activite ci dessus\b|\bvoir activite principale\b|\bvoir activite principale ci dessus\b|\bvoir activites principales\b|\bvoir cidessus\b|\bvoir idem ci dessus\b|\bvoir objet social\b|\bvoir page 1\b|\bvoir page precedente\b|\bvoir plus haut\b|\bvoir princiale activite\b|\bvoir princiales activites\b|\bvoir princiapales activites\b|\bvoir princiaples activites\b|\bvoir principale activite\b|\bvoir principales activites\b|\bvoir principales activites de l entreprise\b|\bvoir principales actvites\b|\bvoir principalesactivites\b|\bvoir principles activites\b|\bvoir rubrique principales activites de l entreprise\b|\bvoir sur la 1ere page\b|\bvoir dessus\b|voir: \"principales activite de l entreprise\"|voir: \"principales activites de l entreprises\"|voir: \"principales activites de l entrprise\"|voir: \"principales activites en entreprise\"|\bconforme au kbis\b|\bsans changement\b|\bsans activite\b|\bsans acitivite\b|\bactivite inchangee\b|\bactivites inchangees\b|\bsiege social\b|\ba definir\b|\ba preciser\b|\bci dessus\b|\bci desus\b|\bci desssus\b|\bvoir activit principale\b|\bidem extrait kbis\b|\bn a plus a etre mentionne sur l extrait decret\b|\bcf statuts\b|\bactivite principale case\b|\bactivites principales case\b|\bactivite principale\b|\bactivites principales\b|\bvoir case\b|\baucun changement\b|\bsans modification\b|\bactivite non modifiee\b|\bactivite identique\b|\bpas de changement\b|\bcode\b|\bape\b|\bnaf\b|\binchangee\b|\binchnagee\b|\bkbis\b|\bk bis\b|\binchangees\b|\bnp\b|\binchange\b|\bnc\b|\bxx\b|\bxxx\b|\binconnue\b|\binconnu\b|\bvoir\b|\bannexe\b|\bmo\b|\biem\b|\binchanges\b|\bactivite demeure\b|\bactivite inchangée\b|\bcase precedente\b|\bidem cadre precedent\b|\bactivite demeure\b|\bactivite inchangée\b|\bnon renseignee\b|\bneant\b|\bnon renseigne\b",
                "",
            ),
            # supprime hyphen pour les mots comme e-commerce
            (r"e-", "e"),
            # accole le e pour les mots comme e-commerce
            (r"\be\s", " e"),
            # Remove punctuation
            (r"[^\w\s]", " "),
            # Remove specific words without meaning
            (
                r"\bcode\b|\bcadre\b|\bape\b|\bape[a-z]{1}\b|\bnaf\b|\binchangee\b|\binchnagee\b|\bkbis\b|\bk bis\b|\binchangees\b|\bnp\b|\binchange\b|\bnc\b|\bidem\b|\bxx\b|\bxxx\b|\baa\b|\baaa\b|\bidem cadre precedent\b|\bidem case\b|\binchanges\b|\bmo\b|\biem\b|\bci dessus\b|\bet\b",
                "",
            ),
            # Remove digits
            (r"[\d+]", " "),
        ]

        def clean_single_label(label):
            # Normalize encoding and convert to lowercase
            label = unidecode.unidecode(label.lower())

            # Apply replacement patterns
            for pattern, replacement in replacement_patterns:
                label = re.sub(pattern, replacement, label, flags=re.IGNORECASE)

            # Remove single-letter words
            label = " ".join([word for word in label.split() if len(word) > 1])

            # Remove stopwords and stem
            label_words = [
                self.stemmer.stem(word) for word in label.split() if word not in self.stopwords
            ]

            return " ".join(label_words)

        # Clean and remove duplicated words within each label
        cleaned_libs = [clean_single_label(label) for label in libs]

        return cleaned_libs

    def oversample_df(self, df: pd.DataFrame, threshold: int, y: str):
        """
        Oversamples the minority classes in a pandas DataFrame to achieve a more balanced dataset.

        Args:
            df (pd.DataFrame): The input DataFrame to be oversampled.
            threshold (int): The minimum number of samples for each class. Classes with fewer
                samples than the threshold will be oversampled.
            Y (str): The name of the column containing the class labels.

        Returns:
            pd.DataFrame: The oversampled DataFrame with a balanced distribution of classes.
        """
        code_to_oversample = df.value_counts(y)[df.value_counts(y) < threshold].index.to_list()
        df_oversampled = pd.DataFrame(columns=df.columns)

        for code in code_to_oversample:
            nb_to_sample = threshold - df[df[y] == code].shape[0]
            df_oversampled = pd.concat(
                [df_oversampled, df[df[y] == code].sample(n=nb_to_sample, replace=True)]
            )

        return pd.concat([df, df_oversampled])

    def add_missing_codes(
        self,
        df: pd.DataFrame,
        df_naf: pd.DataFrame,
        y: str,
        text_feature: str,
        textual_features: list,
        categorical_features: list,
        add_all: bool = False,
    ):
        """
        Adds missing APE codes in the train database by adding the official label as text feature.

        Args:

            df (pd.DataFrame): The input DataFrame to be oversampled.
            df_naf (pd.DataFrame): The DataFrame containing all APE codes and labels.
            y (str): The name of the column containing the class labels.
            text_feature (str): The name of the text feature.
            textual_features (list): The list of textual features.
            categorical_features (list): The list of categorical features.


        Returns:
            pd.DataFrame: The oversampled DataFrame with a balanced distribution of classes.
        """
        if not add_all:
            missing_codes = set(df_naf["APE_NIV5"]) - set(df[y])
        else:
            missing_codes = set(df_naf["APE_NIV5"])

        fake_obs = df_naf[df_naf["APE_NIV5"].isin(missing_codes)]
        fake_obs.loc[:, text_feature] = self.clean_lib(fake_obs.LIB_NIV5.to_list())
        fake_obs.index = [f"FAKE_TRAIN_{i}" for i in range(fake_obs.shape[0])]
        fake_obs[y] = fake_obs["APE_NIV5"]
        fake_obs[text_feature] = fake_obs[[text_feature]].apply(
            lambda row: " ".join(f"[{col}] {val}" for col, val in row.items() if val != ""), axis=1
        )
        df = pd.concat([df, fake_obs[[col for col in fake_obs.columns if col in df.columns]]])

        if textual_features is not None:
            for feature in textual_features:
                df[feature] = df[feature].fillna(value="")
        if categorical_features is not None:
            for feature in categorical_features:
                df[feature] = df[feature].fillna(value="NaN")

        print(f"\t*** {len(missing_codes)} missing codes have been added in the database...\n")
        return df
