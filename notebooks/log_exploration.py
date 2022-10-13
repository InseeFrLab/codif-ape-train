import sys
sys.path.append("../")

header_length = len("2022-10-05 11:56:15.580  INFO 8982 --- [           main] ")
header_length

event_length = len("f.i.sirene4.repertoire.BatchApplication  ")
event_length

first_line = "2022-10-05 11:56:15.580  INFO 8982 --- [           main] f.i.sirene4.repertoire.BatchApplication  : Starting BatchApplication v2.3.3 using Java 11.0.16.1 on qfbatrelst01.ad.insee.intra with PID 8982 (/opt/insee/sirene4/qf3/lib/repertoire-batch-2.3.3.jar started by www-data in /opt/insee/sirene4/qf3/tmp)"

line_without_timestamp = first_line[header_length:]
event_type = line_without_timestamp[:event_length]
event_type.rstrip()

description = line_without_timestamp[event_length + 2:]
description

import pandas as pd

def extract_log_info(f):
    event_types = []
    descriptions = []
    for line in f:
        line_without_timestamp = line[header_length:]
        if not line_without_timestamp:
            continue
        event_types.append(line_without_timestamp[:event_length].rstrip())
        descriptions.append(line_without_timestamp[event_length + 2:])
    return pd.DataFrame(list(zip(event_types, descriptions)), columns =['event_type', 'description'])

with open("../data/api_log.log") as f:
    df = extract_log_info(f)

df

identifier = "s.i.AbstractBatchCodificationServiceImpl"
log_info = "r.b.j.c.s.i.BatchCodificationServiceImpl"
raw_input = "stractLiasse1ToLiasseVarInteretProcessor"

df_ids = df[df.event_type == identifier]
df_ids.head()

df_info = df[df.event_type == log_info]
df_info.head()

df_raw_input = df[df.event_type == raw_input]
df_raw_input.head()

df_raw_input = df_raw_input[df_raw_input.description.str.startswith("LiasseVarInteretCodification")]
df_raw_input.head()

test_str = df_raw_input.iloc[0, 1]
test_str

import re

regex = re.compile(r'norme=([^,]*),')
matches = regex.search(test_str)
matches.group(1)

regex = re.compile(r'liasseType=([^,]*),')
matches = regex.search(test_str)
matches.group(1)

raw_fields = [
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
raw_regexes = [re.compile(r'{}'.format(field + '=([^,]*)[,\]]')) for field in raw_fields]
raw_regexes

def parse_raw_input(raw_input, fields, regexes):
    raw_input_dict = {}
    for field, regex in zip(fields, regexes):
        matches = regex.search(raw_input)
        raw_input_dict[field] = matches.group(1)
    return raw_input_dict

parse_raw_input(test_str, raw_fields, raw_regexes)

dict_series = [parse_raw_input(raw_input, raw_fields, raw_regexes) for raw_input in df_raw_input.description]

pd.DataFrame(list(dict_series))

test_str = df_info.iloc[0, 1]
test_str

info_fields = [
    "libelleActivite",
    "natureActivites",
    "liasseType",
    "evenementType",
    "surface",
    "libelleNettoye",
    "predictions",
    "bilan"
]
info_regexes = [re.compile(r'{}'.format(field + '=([^,]*),')) for field in info_fields]

info_fields.append("fasttextVersion")
info_regexes.append(re.compile(r'fasttextVersion=([^,]*)\]'))

info_regexes

parse_raw_input(test_str, info_fields, info_regexes)

def extract_first_pred(predictions):
    regex = re.compile(r'proposé = ([^;]*) ;.*associée = ([^;]*)\].*proposé = ([^;]*) ;.*associée = ([^;]*)\]')
    matches = regex.search(predictions)
    first_code = matches.group(1)
    second_code = matches.group(3)
    first_proba = matches.group(2)
    second_proba = matches.group(4)
    return (first_code, second_code, float(first_proba), float(second_proba))

extract_first_pred(parse_raw_input(test_str, info_fields, info_regexes)["predictions"])

dict_series = [parse_raw_input(info_input, info_fields, info_regexes) for info_input in df_info.description]

df = pd.DataFrame(list(dict_series))

predictions = [extract_first_pred(predictions) for predictions in df["predictions"]]
df["first_pred"] = [prediction[0] for prediction in predictions]
df["second_pred"] = [prediction[1] for prediction in predictions]
df["first_proba"] = [prediction[2] for prediction in predictions]
df["second_proba"] = [prediction[3] for prediction in predictions]
df

# mc cp minio/projet-ape/mlflow-artifacts/1/5490ebb3b62a43e494517f819cf20322/artifacts/default/artifacts/default.bin models/model.bin

import fasttext
model = fasttext.load_model("codification-ape/models/model.bin")

tmp = model.predict(df.iloc[:]["libelleNettoye"].to_list(), k=2)

dict_results = {f"pred_{pred+1}" : [tmp[0][lib][pred][-5:] for lib in range(len(tmp[0]))]
  for pred in range(2)
} | {
f"prob_{pred+1}" : [tmp[1][lib][pred] for lib in range(len(tmp[0]))]
  for pred in range(2)
}

df_results = pd.DataFrame(dict_results)
df_results

sum(df.first_pred == df_results.pred_1)/df.shape[0]

import numpy as np
from nltk.corpus import stopwords as ntlk_stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language="french")
stopwords = tuple(ntlk_stopwords.words("french"))

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
    # df = df.dropna(subset=[text_feature])
    df[text_feature] =df[text_feature].fillna(value="NaN")

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
                stemmer.stem(word)
                for word in libs_token[i]
                if word not in stopwords
            ]
        )
        for i in range(len(libs_token))
    ]

    return df
dff = df
dff["libelleActivite"] = df["libelleActivite"].fillna(value="NaN")
df_prepro = clean_lib(dff, "libelleActivite")

lib_raw = df.libelleActivite.to_list()
lib_clean_PY = df_prepro.libelleActivite.to_list()
lib_clean_JAVA = df.libelleNettoye.apply(lambda x : x.split(" AUTO")[0]).to_list()
compare_libs = pd.DataFrame({"RAW" : lib_raw, "PYTHON" : lib_clean_PY, "JAVA" : lib_clean_JAVA})
compare_libs["CHECK"] = compare_libs.PYTHON ==  compare_libs.JAVA
compare_libs