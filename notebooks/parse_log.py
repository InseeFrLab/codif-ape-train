import sys
import pandas as pd
import unidecode
import re
import fasttext
import numpy as np
from nltk.corpus import stopwords as ntlk_stopwords
from nltk.stem.snowball import SnowballStemmer
from enum import Enum

sys.path.append("../")

HEADER_LEN = 57
EVENT_LEN = 41


class EventType(Enum):
    """

    """
    ID = "s.i.AbstractBatchCodificationServiceImpl"
    INFO = "r.b.j.c.s.i.BatchCodificationServiceImpl"
    RAW_INPUT = "stractLiasse1ToLiasseVarInteretProcessor"


def extract_log_info(f):
    """_summary_

    Args:
        f (_type_): _description_

    Returns:
        _type_: _description_
    """
    event_types = []
    descriptions = []
    for line in f:
        line_without_timestamp = line[HEADER_LEN:]
        if not line_without_timestamp:
            continue
        event_types.append(line_without_timestamp[:EVENT_LEN].rstrip())
        descriptions.append(line_without_timestamp[EVENT_LEN + 2:])
    return pd.DataFrame(
        list(zip(event_types, descriptions)),
        columns=['event_type', 'description']
    )


def parse_raw_input(raw_input, fields, regexes):
    """_summary_

    Args:
        raw_input (_type_): _description_
        fields (_type_): _description_
        regexes (_type_): _description_

    Returns:
        _type_: _description_
    """
    raw_input_dict = {}
    for field, regex in zip(fields, regexes):
        matches = regex.search(raw_input)
        if matches.group(1) is not None:
            raw_input_dict[field] = matches.group(1).strip('"')
        else:
            raw_input_dict[field] = matches.group(2)
    return raw_input_dict


def extract_first_pred(predictions):
    """_summary_

    Args:
        predictions (_type_): _description_

    Returns:
        _type_: _description_
    """
    regex = re.compile(r'proposé = ([^;]*) ;.*associée = ([^;]*)\].*proposé = ([^;]*) ;.*associée = ([^;]*)\]')
    matches = regex.search(predictions)
    first_code = matches.group(1)
    second_code = matches.group(3)
    first_proba = matches.group(2)
    second_proba = matches.group(4)
    return (first_code, second_code, float(first_proba), float(second_proba))


def clean_lib(df, text_feature):
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

        # On supprime les mots d'une seule lettre
        df[text_feature] = df[text_feature].replace(
            to_replace=r"(?:\s|^)[a-z]{1}(?:\s|$)", value="", regex=True
        )

        # On supprime toutes les ponctuations
        df[text_feature] = df[text_feature].replace(
            to_replace=r"[^\w\s]", value=" ", regex=True
        )

        # On supprime tous les chiffres
        df[text_feature] = df[text_feature].replace(
            to_replace=r"[\d+]", value=" ", regex=True
        )

        # On supprime les longs mots sans sens
        #df[text_feature] = df[text_feature].replace(
        #    to_replace=LongWord2remove, value="", regex=True
        #)

        # On supprime les mots courts sans sens
        #df[text_feature] = df[text_feature].replace(
        #    to_replace=Word2remove, value="", regex=True
        #)

        # On supprime les multiple space
        df[text_feature] = df[text_feature].replace(r"\s\s+", " ", regex=True)

        # On strip les libellés
        df[text_feature] = df[text_feature].str.strip()

        # On remplace les empty string par des NaN
        df[text_feature] = df[text_feature].replace(r"^\s*$", np.nan, regex=True)

        # On supprime les NaN
        # df = df.dropna(subset=[text_feature])
        df[text_feature] =df[text_feature].fillna(value="NaN")

        # On tokenize tous les libellés
        libs_token = [lib.split() for lib in df[text_feature].to_list()]

        # Pour chaque libellé on supprime les stopword et on racinise les mots
        libs_token = [
            [
                stemmer.stem(word)
                for word in libs_token[i]
                if word not in stopwords
            ]
            for i in range(len(libs_token))
        ]

        # On supprime les mots duppliqué dans un même libellé
        libs_token = [
            sorted(set(libs_token[i]), key=libs_token[i].index)
            for i in range(len(libs_token))
        ]

        df[text_feature] = [
            " ".join(libs_token[i]) for i in range(len(libs_token))
        ]

        return df


RAW_INPUT_FIELDS = [
    "norme",
    "siren",
    "nic",
    "liasseType",
    "categorieJuridique",
    "domas",
    "ssdom",
    "domaineAssoc",
    "ssDomaineAssoc",
    "libelleActivitePrincipaleEtablissement",
    "sedentarite",
    "natureActivites",
    "surface",
    "lieuExercice",
    "presenceSalaries"
]
RAW_INPUT_REGEXES = [
    re.compile(r'{}'.format(field + '=([^,]*)[,\]]'))
    for field in RAW_INPUT_FIELDS
]
INFO_FIELDS = [
    "natureActivites",
    "liasseType",
    "evenementType",
    "surface",
    "libelleNettoye",
    "predictions",
    "bilan"
]
INFO_REGEXES = [
    re.compile(r'{}'.format(field + '=([^,]*),'))
    for field in INFO_FIELDS
]
INFO_FIELDS.append("fasttextVersion")
INFO_REGEXES.append(re.compile(r'fasttextVersion=([^,]*)\]'))
INFO_FIELDS.append("libelleActivite")
INFO_REGEXES.append(re.compile(r'libelleActivite=(\"[^\"]*\")?([^,]*),'))


if __name__ == "__main__":
    with open("../data/api_log.log") as f:
        df = extract_log_info(f)

    df_ids = df[df.event_type == EventType.ID.value]
    df_info = df[df.event_type == EventType.INFO.value]
    df_raw_input = df[df.event_type == EventType.RAW_INPUT.value]
    df_raw_input = df_raw_input[
        df_raw_input.description.str.startswith("LiasseVarInteretCodification")
    ]

    df = pd.DataFrame(
        [
            parse_raw_input(info_input, INFO_FIELDS, INFO_REGEXES)
            for info_input in df_info.description
            if not info_input.__contains__('""')
        ]
    )

    predictions = [
        extract_first_pred(predictions)
        for predictions in df["predictions"]
    ]
    df["first_pred"] = [prediction[0] for prediction in predictions]
    df["second_pred"] = [prediction[1] for prediction in predictions]
    df["first_proba"] = [prediction[2] for prediction in predictions]
    df["second_proba"] = [prediction[3] for prediction in predictions]

    stemmer = SnowballStemmer(language="french")
    stopwords = tuple(ntlk_stopwords.words("french"))

    df["libelleActivite"] = [unidecode.unidecode(lib) for lib in df["libelleActivite"]]

    dff = df.copy()
    dff["libelleActivite"] = df["libelleActivite"].fillna(value="NaN")
    df_prepro = clean_lib(dff, "libelleActivite")
    lib_raw = df.libelleActivite.to_list()
    lib_clean_PY = df_prepro.libelleActivite.to_list()
    lib_clean_JAVA = df.libelleNettoye.apply(lambda x : x.split(" AUTO")[0]).to_list()
    compare_libs = pd.DataFrame({"RAW" : lib_raw, "PYTHON" : lib_clean_PY, "JAVA" : lib_clean_JAVA})
    compare_libs["CHECK"] = compare_libs.PYTHON ==  compare_libs.JAVA
    compare_libs.to_csv("comparison.csv")
