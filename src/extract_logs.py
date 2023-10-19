from tqdm import tqdm
from datetime import datetime

import pandas as pd
import numpy as np
import os
import re

# Fonction pour extraire les informations des prédictions
def extract_predictions(predictions_str):
    regex_pattern = r'\[prediction n° (\d+) : code naf proposé = (\w+) ; proba associée = ([\w.]+(?:E-?\d+)?)\]'
    matches = re.findall(regex_pattern, predictions_str)
    return matches

# Fonction pour créer un DataFrame à partir de la ligne
def extract_data_by_line(line):

    FIELDS = [
        "sourceAppel",
        "libelleActivite",
        "natureActivites",
        "liasseType",
        "evenementType",
        "surface",
        "libelleNettoye",
        "bilan",
        "fasttextVersion"
    ]

    DATA = {
        field : re.search(r'{}'.format(field + '=([^,]+)'), line).group(1) if field != "fasttextVersion" else re.search(r'{}'.format(field + '=([^,]+)'), line).group(1)[:-1]
        for field in FIELDS
    }

    DATA["timestamp"] = datetime.strptime(re.search(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)', line).group(1), "%Y-%m-%d %H:%M:%S.%f")
    
    try:
        predictions_str = re.search(r'predictions=(\[.*?\],)', line).group(1)
        predictions = extract_predictions(predictions_str)
        
    except AttributeError:
        predictions = [[np.nan] * 3] * 2

    DATA["prediction_1"] = predictions[0][1]
    DATA["proba_1"] = float(predictions[0][2])
    DATA["prediction_2"] = predictions[1][1]
    DATA["proba_2"] = float(predictions[1][2])

    return DATA

# Fonction pour extraire les logs d'un fichier
def extract_log_info(f):
    """_summary_

    Args:
        f (_type_): _description_

    Returns:
        _type_: _description_
    """
    PATTERN = r'fr.insee.sirene4.repertoire.api.codification.rest.CodificationController:\d{3}'


    results =  {
        "timestamp" : [],
        "sourceAppel" : [],
        "libelleActivite" : [],
        "natureActivites" : [],
        "liasseType" : [],
        "evenementType" : [],
        "surface" : [],
        "libelleNettoye" : [],
        "prediction_1" : [],
        "proba_1" : [],
        "prediction_2" : [],
        "proba_2" : [],
        "bilan" : [],
        "fasttextVersion" : [],
    }

    for line in f:
        idx = line.find("CodificationBilan")
        is_empty = bool(re.match(r'^\s*$', line))
        is_valid_line = bool(re.search(PATTERN, line))

        if (is_empty or not is_valid_line) and idx == -1:
            continue

        data = extract_data_by_line(line)

        for key in results.keys():
            results[key].append(data[key])
    return pd.DataFrame(results)


logs_by_file = []

for file in tqdm(os.listdir("log_files/log_zip/")):
    with open(f"log_files/log_zip/{file}") as f:
        logs_by_file.append(extract_log_info(f))

all_logs = pd.concat(logs_by_file)
all_logs.to_parquet(f"logs_{datetime.now().strftime('%Y-%m-%d')}.parquet")
