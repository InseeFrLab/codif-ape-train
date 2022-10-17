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
        df_naf: pd.DataFrame,
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
            df_naf (pd.DataFrame): Dataframe that contains all codes and libs.
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

        # Adding missing APE codes in the database
        MissingCodes = set(df_naf[y]) - set(df[y])
        Fake_obs = df_naf[df_naf[y].isin(MissingCodes)]
        Fake_obs.loc[:, text_feature] = Fake_obs.LIB_NIV5
        Fake_obs.index = [f"FAKE_{i}" for i in range(Fake_obs.shape[0])]
        Fake_obs = self.clean_lib(Fake_obs, text_feature)
        df = pd.concat([df, Fake_obs])
        print(
            f"\t*** {len(MissingCodes)} missing codes have been added in the database.\n"
        )

        # Guichet unique split
        df_gu = df[df.index.str.startswith("J")]
        df = df[~df.index.str.startswith("J")]
        # Train/test split
        features = [text_feature]
        if categorical_features is not None:
            features += categorical_features

        X_train, X_test, y_train, y_test = train_test_split(
            df[
                features + [f"APE_NIV{i}" for i in range(1, 6) if str(i) not in [y[-1]]]
            ],
            df[y],
            test_size=0.2,
            random_state=0,
            shuffle=True,
        )

        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)

        # Adding missing APE codes in the train database
        MissingCodes = set(df_naf[y]) - set(df_train[y])
        Fake_obs = df_naf[df_naf[y].isin(MissingCodes)]
        Fake_obs.loc[:, text_feature] = Fake_obs.LIB_NIV5
        Fake_obs.index = [f"FAKE_TRAIN_{i}" for i in range(Fake_obs.shape[0])]
        Fake_obs = self.clean_lib(Fake_obs, text_feature)
        df_train = pd.concat([df_train, Fake_obs])
        print(
            f"\t*** {len(MissingCodes)} missing codes have been added in the train database...\n"
        )

        if oversampling is not None:
            print("\t*** Oversampling the train database...\n")
            t = time.time()
            df_train = self.oversample_df(df_train, oversampling["threshold"], y)
            print(
                f"\t*** Done! Oversampling lasted {round((time.time() - t)/60,1)} minutes.\n"
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
        # Libellé vide de sens fournit par Christine
        LibVideSens = r"\bidem\b|\bvoir ci dessous\b|\[vide\]|\bundefined\b|\bpas d objet\b|\(voir ci dessus\)|\(voir extrait siege social\/etablissement principal\)|\bcf activite principale\b|\bcf activite principale et objet\b|\bcf activites de l entreprise\b|\bcf activites principales de l entreprise\b|\bcf actvites principales\b|\bcf k bis\b|\bcf le principales activites de l  entreprise\b|\bcf le sprincipale activites de l  entreprise\b|\bcf le sprincipales activites de l  entreprise\b|\bcf les activites principales de l  entreprise\b|\bcf les ppales activites de l  entreprise\b|\bcf les ppales activites de la ste\b|\bcf les principale activites de l  entreprise\b|\bcf les principales activites\b|\bcf les principales activites de l  entreprise\b|\bcf les principales activites de l  entreprises\b|\bcf les principales activites ppales de l  entreprise\b|\bcf les principales activtes de l  entreprise\b|\bcf les principales acttivites de l  entreprise\b|\bcf les prinipales activites de l  entreprise\b|\bcf lesprincipales activites de l  entreprise\b|\bcf objet\b|\bcf obs\b|\bcf principales activite de l  entreprise\b|\bcf principales activites de l  entreprise\b|cf rubrique \"principales activites de l entreprise\" idem|cf rubrique n2 ci dessus \(743b\)|\bcf supra\b|\bcf ci  dessus\b|\bcommerce de detail, idem case 2\b|\bextension a: voir ci dessus\b|\bid\b|\bid principales activites\b|\bid principales activites de l  entreprise\b|\bidem ci dessus\b|idem \( voir principales activites\)|\bidem  dessus\b|\bidem 1ere page\b|\bidem a principales activites de l  entreprise\b|\bidem activiet eprincipale\b|\bidem activite\b|\bidem activite 1ere page\b|\bidem activite ci  dessus\b|\bidem activite de l  entreprise\b|\bidem activite enoncee ci  dessus\b|\bidem activite entreprise\b|\bidem activite generales\b|\bidem activite premiere page\b|\bidem activite principale\b|\bidem activite princippale\b|\bidem activite prinicpale\b|\bidem activite sur 1ere page\b|\bidem activites ci dessus\b|\bidem activites declarees au siege et principal\b|\bidem activites enoncees ci dessus\b|\bidem activites entreprise\b|\bidem activites principales\b|\bidem activites principales de l entreprise\b|\bidem activites siege\b|\bidem activte principale\b|\bidem activtie 1ere page\b|\bidem au siege\b|\bidem au siege social\b|\bidem aux principales actiivtes\b|\bidem aux principales activites\b|\bidem case 13\b|\bidem ci dessous\b|\bidem ci dessus enoncee\b|\bidem cidessus\b|\bidem objet\b|\bidem premiere page\b|\bidem pricincipales activites de l entreprise\b|\bidem pricipales activites\b|\bidem principale activite\b|\bidem principales activite de l entreprise\b|\bidem principales activite de l entreprises\b|\bidem principales activite l entreprise\b|\bidem principales activites\b|\bidem principales activites citees ci dessus\b|\bidem principales activites de l entreprises\b|idem principales activites de l entreprise\(objet\)|\bidem principales activites et objet social\b|\bidem principales activitse de l entreprise\b|\bidem que celle decrite plus haut\b|\bidem que ci dessus\b|\bidem que l activite decrite plus haut\b|\bidem que les activites principales\b|\bidem que les activites principales ci dessus\b|\bidem que les activitges principales\b|\bidem que les principales activites\b|\bidem que les principales activites de l entreprise\b|\bidem que pour le siege\b|\bidem rubrique principales activites de l entreprise\b|\bidem siege\b|idem siege \+ voir observation|\bidem siege et ets principal\b|\bidem siege social\b|idem siege, \(\+ articles americains\)|\bidem societe\b|\bidem voir activite principale\b|\bidem voir ci dessus\b|\bidentique a l objet social indique en case 2 de l imprime m2\b|\bidm ci dessus\b|\bnon indiquee\b|\bnon precise\b|\bnon precisee\b|\bnon precisees\b|\bvoir 1ere page\b|\bvoir activite ci dessus\b|\bvoir activite principale\b|\bvoir activite principale ci dessus\b|\bvoir activites principales\b|\bvoir cidessus\b|\bvoir idem ci dessus\b|\bvoir objet social\b|\bvoir page 1\b|\bvoir page precedente\b|\bvoir plus haut\b|\bvoir princiale activite\b|\bvoir princiales activites\b|\bvoir princiapales activites\b|\bvoir princiaples activites\b|\bvoir principale activite\b|\bvoir principales activites\b|\bvoir principales activites de l entreprise\b|\bvoir principales actvites\b|\bvoir principalesactivites\b|\bvoir principles activites\b|\bvoir rubrique principales activites de l entreprise\b|\bvoir sur la 1ere page\b|\bvoir dessus\b|voir: \"principales activite de l entreprise\"|voir: \"principales activites de l entreprises\"|voir: \"principales activites de l entrprise\"|voir: \"principales activites en entreprise\"|\bconforme au kbis\b|\bsans changement\b|\bsans activite\b|\bsans acitivite\b|\bactivite inchangee\b|\bactivites inchangees\b|\bsiege social\b|\ba definir\b|\ba preciser\b|\bci dessus\b|\bci desus\b|\bci desssus\b|\bvoir activit principale\b|\bidem extrait kbis\b|\bn a plus a etre mentionne sur l extrait decret\b|\bcf statuts\b|\bactivite principale case\b|\bactivites principales case\b|\bactivite principale\b|\bactivites principales\b|\bvoir case\b|\baucun changement\b|\bsans modification\b|\bactivite non modifiee\b|\bactivite identique\b|\bpas de changement\b|\bcode\b|\bape\b|\bnaf\b|\binchangee\b|\binchnagee\b|\bkbis\b|\bk bis\b|\binchangees\b|\bnp\b|\binchange\b|\bnc\b|\bxx\b|\bxxx\b|\binconnue\b|\binconnu\b|\bvoir\b|\bannexe\b|\bmo\b|\biem\b|\binchanges\b|\bactivite demeure\b|\bactivite inchangée\b|\bcase precedente\b|\bidem cadre precedent\b|\bactivite demeure\b|\bactivite inchangée\b|\bnon renseignee\b|\bneant\b|\bnon renseigne\b"

        # On définit une regex de mots à supprimer du jeu de données
        Word2remove = r"\bcode\b|\bcadre\b|\bape\b|\bape[a-z]{1}\b|\bnaf\b|\binchangee\b|\binchnagee\b|\bkbis\b|\bk bis\b|\binchangees\b|\bnp\b|\binchange\b|\bnc\b|\bidem\b|\bxx\b|\bxxx\b|\baa\b|\baaa\b|\bidem cadre precedent\b|\bidem case\b|\binchanges\b|\bmo\b|\biem\b|\bci dessus\b|\bet\b"

        # On passe tout en minuscule
        df[text_feature] = df[text_feature].map(str.lower)

        # On supprime les libellés vide de sens (DOIT ETRE FAIT EN AMONT DU MODELE EN JAVA)
        df[text_feature] = df[text_feature].replace(
            to_replace=LibVideSens, value="", regex=True
        )

        # supprime hyphen pour les mots comme e-commerce
        df[text_feature] = df[text_feature].replace(
            to_replace=r"e-", value="e", regex=True
        )

        # accole le e pour les mots comme e-commerce
        df[text_feature] = df[text_feature].replace(
            to_replace=r"\be\s", value=" e", regex=True
        )

        # On supprime toutes les ponctuations
        df[text_feature] = df[text_feature].replace(
            to_replace=r"[^\w\s]+", value=" ", regex=True
        )

        # On supprime certains mots sans sens (DOIT ETRE FAIT DANS LE PREPROCESSING EN JAVA)
        df[text_feature] = df[text_feature].replace(
            to_replace=Word2remove, value="", regex=True
        )

        # On supprime les mots d'une seule lettre
        df[text_feature] = df[text_feature].apply(
            lambda x: ' '.join([w for w in x.split() if len(w) > 1])
        )

        # On supprime tous les chiffres
        df[text_feature] = df[text_feature].replace(
            to_replace=r"[\d+]", value=" ", regex=True
        )

        # On supprime les mots d'une seule lettre
        df[text_feature] = df[text_feature].apply(
            lambda x: ' '.join([w for w in x.split() if len(w) > 1])
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
