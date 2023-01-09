import sys
import pandas as pd
import unidecode
import re
import string
import numpy as np
from nltk.corpus import stopwords as ntlk_stopwords
from nltk.stem.snowball import SnowballStemmer
from enum import Enum

sys.path.append("../")

HEADER_LEN = 55
EVENT_LEN = 77


class EventType(Enum):
    """

    """
    INFO_169 = "fr.insee.sirene4.repertoire.api.codification.rest.CodificationController:169"
    INFO_122 = "fr.insee.sirene4.repertoire.api.codification.rest.CodificationController:122"


def extract_log_info(f):
    """_summary_

    Args:
        f (_type_): _description_

    Returns:
        _type_: _description_
    """
    timestamp = []
    event_types = []
    descriptions = []
    for line in f:
        line_without_timestamp = line[HEADER_LEN:]
        if not line_without_timestamp:
            continue
        timestamp.append(line[:23])
        event_types.append(line_without_timestamp[:EVENT_LEN].rstrip())
        descriptions.append(line_without_timestamp[EVENT_LEN + 2:])
    return pd.DataFrame(
        list(zip(timestamp, event_types, descriptions)),
        columns=['timestamp', 'event_type', 'description']
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
    regex = re.compile(r"\[prediction n° \d+ : code naf proposé = ([A-Z0-9]+) ; proba associée = ([\d.E-]+)\]")
    matches = re.findall(regex, predictions)
    first_code = matches[0][0]
    second_code = matches[1][0]
    first_proba = matches[0][1]
    second_proba = matches[1][1]
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
#        Libellé vide de sens fournit par Christine
#        LibVideSens = r"\bidem\b|\bvoir ci dessous\b|\[vide\]|\bundefined\b|\bpas d objet\b|\(voir ci dessus\)|\(voir extrait siege social\/etablissement principal\)|\bcf activite principale\b|\bcf activite principale et objet\b|\bcf activites de l entreprise\b|\bcf activites principales de l entreprise\b|\bcf actvites principales\b|\bcf k bis\b|\bcf le principales activites de l  entreprise\b|\bcf le sprincipale activites de l  entreprise\b|\bcf le sprincipales activites de l  entreprise\b|\bcf les activites principales de l  entreprise\b|\bcf les ppales activites de l  entreprise\b|\bcf les ppales activites de la ste\b|\bcf les principale activites de l  entreprise\b|\bcf les principales activites\b|\bcf les principales activites de l  entreprise\b|\bcf les principales activites de l  entreprises\b|\bcf les principales activites ppales de l  entreprise\b|\bcf les principales activtes de l  entreprise\b|\bcf les principales acttivites de l  entreprise\b|\bcf les prinipales activites de l  entreprise\b|\bcf lesprincipales activites de l  entreprise\b|\bcf objet\b|\bcf obs\b|\bcf principales activite de l  entreprise\b|\bcf principales activites de l  entreprise\b|cf rubrique \"principales activites de l entreprise\" idem|cf rubrique n2 ci dessus \(743b\)|\bcf supra\b|\bcf ci  dessus\b|\bcommerce de detail, idem case 2\b|\bextension a: voir ci dessus\b|\bid\b|\bid principales activites\b|\bid principales activites de l  entreprise\b|\bidem ci dessus\b|idem \( voir principales activites\)|\bidem  dessus\b|\bidem 1ere page\b|\bidem a principales activites de l  entreprise\b|\bidem activiet eprincipale\b|\bidem activite\b|\bidem activite 1ere page\b|\bidem activite ci  dessus\b|\bidem activite de l  entreprise\b|\bidem activite enoncee ci  dessus\b|\bidem activite entreprise\b|\bidem activite generales\b|\bidem activite premiere page\b|\bidem activite principale\b|\bidem activite princippale\b|\bidem activite prinicpale\b|\bidem activite sur 1ere page\b|\bidem activites ci dessus\b|\bidem activites declarees au siege et principal\b|\bidem activites enoncees ci dessus\b|\bidem activites entreprise\b|\bidem activites principales\b|\bidem activites principales de l entreprise\b|\bidem activites siege\b|\bidem activte principale\b|\bidem activtie 1ere page\b|\bidem au siege\b|\bidem au siege social\b|\bidem aux principales actiivtes\b|\bidem aux principales activites\b|\bidem case 13\b|\bidem ci dessous\b|\bidem ci dessus enoncee\b|\bidem cidessus\b|\bidem objet\b|\bidem premiere page\b|\bidem pricincipales activites de l entreprise\b|\bidem pricipales activites\b|\bidem principale activite\b|\bidem principales activite de l entreprise\b|\bidem principales activite de l entreprises\b|\bidem principales activite l entreprise\b|\bidem principales activites\b|\bidem principales activites citees ci dessus\b|\bidem principales activites de l entreprises\b|idem principales activites de l entreprise\(objet\)|\bidem principales activites et objet social\b|\bidem principales activitse de l entreprise\b|\bidem que celle decrite plus haut\b|\bidem que ci dessus\b|\bidem que l activite decrite plus haut\b|\bidem que les activites principales\b|\bidem que les activites principales ci dessus\b|\bidem que les activitges principales\b|\bidem que les principales activites\b|\bidem que les principales activites de l entreprise\b|\bidem que pour le siege\b|\bidem rubrique principales activites de l entreprise\b|\bidem siege\b|idem siege \+ voir observation|\bidem siege et ets principal\b|\bidem siege social\b|idem siege, \(\+ articles americains\)|\bidem societe\b|\bidem voir activite principale\b|\bidem voir ci dessus\b|\bidentique a l objet social indique en case 2 de l imprime m2\b|\bidm ci dessus\b|\bnon indiquee\b|\bnon precise\b|\bnon precisee\b|\bnon precisees\b|\bvoir 1ere page\b|\bvoir activite ci dessus\b|\bvoir activite principale\b|\bvoir activite principale ci dessus\b|\bvoir activites principales\b|\bvoir cidessus\b|\bvoir idem ci dessus\b|\bvoir objet social\b|\bvoir page 1\b|\bvoir page precedente\b|\bvoir plus haut\b|\bvoir princiale activite\b|\bvoir princiales activites\b|\bvoir princiapales activites\b|\bvoir princiaples activites\b|\bvoir principale activite\b|\bvoir principales activites\b|\bvoir principales activites de l entreprise\b|\bvoir principales actvites\b|\bvoir principalesactivites\b|\bvoir principles activites\b|\bvoir rubrique principales activites de l entreprise\b|\bvoir sur la 1ere page\b|\bvoir dessus\b|voir: \"principales activite de l entreprise\"|voir: \"principales activites de l entreprises\"|voir: \"principales activites de l entrprise\"|voir: \"principales activites en entreprise\"|\bconforme au kbis\b|\bsans changement\b|\bsans activite\b|\bsans acitivite\b|\bactivite inchangee\b|\bactivites inchangees\b|\bsiege social\b|\ba definir\b|\ba preciser\b|\bci dessus\b|\bci desus\b|\bci desssus\b|\bvoir activit principale\b|\bidem extrait kbis\b|\bn a plus a etre mentionne sur l extrait decret\b|\bcf statuts\b|\bactivite principale case\b|\bactivites principales case\b|\bactivite principale\b|\bactivites principales\b|\bvoir case\b|\baucun changement\b|\bsans modification\b|\bactivite non modifiee\b|\bactivite identique\b|\bpas de changement\b|\bcode\b|\bape\b|\bnaf\b|\binchangee\b|\binchnagee\b|\bkbis\b|\bk bis\b|\binchangees\b|\bnp\b|\binchange\b|\bnc\b|\bxx\b|\bxxx\b|\binconnue\b|\binconnu\b|\bvoir\b|\bannexe\b|\bmo\b|\biem\b|\binchanges\b|\bactivite demeure\b|\bactivite inchangée\b|\bcase precedente\b|\bidem cadre precedent\b|\bactivite demeure\b|\bactivite inchangée\b|\bnon renseignee\b|\bneant\b|\bnon renseigne\b"

        # On définit une regex de mots à supprimer du jeu de données
        Word2remove = r"\bcode\b|\bcadre\b|\bape\b|\bape[a-z]{1}\b|\bnaf\b|\binchangee\b|\binchnagee\b|\bkbis\b|\bk bis\b|\binchangees\b|\bnp\b|\binchange\b|\bnc\b|\bidem\b|\bxx\b|\bxxx\b|\baa\b|\baaa\b|\bidem cadre precedent\b|\bidem case\b|\binchanges\b|\bmo\b|\biem\b|\bci dessus\b|\bet\b"

        # On harmonise l'encodage (principalement suppression accents)
        df[text_feature] = df[text_feature].map(unidecode.unidecode)

        # On passe tout en minuscule
        df[text_feature] = df[text_feature].map(str.lower)

        # On supprime les libellés vide de sens (DOIT ETRE FAIT EN AMONT DU MODELE EN JAVA)
        #df[text_feature] = df[text_feature].replace(
        #    to_replace=LibVideSens, value="", regex=True
        #)

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

        # On supprime les mots duppliqués dans un même libellé
        libs_token = [
            sorted(set(libs_token[i]), key=libs_token[i].index)
            for i in range(len(libs_token))
        ]

        df[text_feature] = [
            " ".join(libs_token[i]) for i in range(len(libs_token))
        ]

        return df


INFO_FIELDS = [
    "sourceAppel",
    "libelleActivite",
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

    df_info_122 = df[df.event_type == EventType.INFO_122.value]
    df_info_169 = df[df.event_type == EventType.INFO_169.value]
    df_info = pd.concat([df_info_122, df_info_169])
    
    df = pd.DataFrame(
        [
            parse_raw_input(info_input, INFO_FIELDS, INFO_REGEXES)
            for info_input in df_info.description
            if not info_input.__contains__('""')
        ]
    )

    df["timestamp"] = [
                    df_info.timestamp[i]
                    for i in df_info.index
                    if not df_info.description[i].__contains__('""')
                ]

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df[df.libelleNettoye != "null"]

    predictions = [
        extract_first_pred(predictions)
        for predictions in df["predictions"]
    ]
    df["first_pred"] = [prediction[0] for prediction in predictions]
    df["second_pred"] = [prediction[1] for prediction in predictions]
    df["first_proba"] = [prediction[2] for prediction in predictions]
    df["second_proba"] = [prediction[3] for prediction in predictions]
    df["score"] = df["first_proba"] - df["second_proba"]
    df["DEC15"] = df["timestamp"] > "2022-12-15 14:12"
    df.to_csv("logs_analysis.csv")

    stemmer = SnowballStemmer(language="french")
    stopwords = tuple(ntlk_stopwords.words("french")) + tuple(string.ascii_lowercase)

    dff = df.copy()
    dff["libelleActivite"] = df["libelleActivite"].fillna(value="NaN")
    df_prepro = clean_lib(dff, "libelleActivite")
    lib_raw = df.libelleActivite.to_list()
    lib_clean_PY = df_prepro.libelleActivite.to_list()
    lib_clean_JAVA = df.libelleNettoye.apply(lambda x : x.split(" AUTO")[0]).to_list()
    compare_libs = pd.DataFrame({"RAW" : lib_raw, "PYTHON" : lib_clean_PY, "JAVA" : lib_clean_JAVA})
    compare_libs["CHECK"] = compare_libs.PYTHON ==  compare_libs.JAVA
    compare_libs.to_csv("comparison.csv")
