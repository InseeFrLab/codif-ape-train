import sys
import pandas as pd
from enum import Enum
import re
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
