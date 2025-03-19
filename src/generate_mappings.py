import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

from preprocessors.pytorch_preprocessor import PytorchPreprocessor
from utils.data import get_all_data, get_df_naf, get_Y

""" Run this file using the following command:
> python src/generate_mappings.py

from the root directory of the project (codif-ape-train).

It generates the json files in data/mappings.

"""

text_feature = "libelle"
textual_features_1 = "NAT_LIB"
textual_features_2 = "AGRI"
categorical_features_1 = "TYP"
categorical_features_2 = "NAT"
categorical_features_3 = "SRF"
categorical_features_4 = "CJ"
categorical_features_5 = "CRT"


def generate_mappings(start_month=1, start_year=2018, revision="NAF2008"):
    Y = get_Y(revision=revision)

    textual_features = [
        value for key, value in globals().items() if key.startswith("textual_features_")
    ]
    categorical_features = [
        value for key, value in globals().items() if key.startswith("categorical_features_")
    ]
    # Load data

    # Sirene 4

    if revision == "NAF2008":
        df_s3, df_s4 = get_all_data(
            revision=revision, start_month=start_month, start_year=start_year
        )

    # Detailed NAF
    df_naf = get_df_naf(revision=revision)

    directory = "data/mappings/"
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists

    # Y mapping from df_naf
    sorted_by_count_unique_values = df_naf["APE_NIV5"].value_counts()
    mapping = {k: v for v, k in enumerate(sorted_by_count_unique_values.index)}

    with open(os.path.join(directory, f"{Y}_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=3)

    # for naf2025, only generate Y mapping and exit
    if revision == "NAF2025":
        return

    preprocessor = PytorchPreprocessor()

    df_s4 = preprocessor.preprocess(
        df=df_s4,
        df_naf=df_naf,
        y=Y,
        text_feature=text_feature,
        textual_features=textual_features,
        categorical_features=categorical_features,
        test_size=0.1,
        mapping=True,  # We do not want full processing, just replance NaN by suitable string...
    )

    df_s3 = preprocessor.preprocess(
        df_s3,
        df_naf,
        Y,
        text_feature,
        textual_features,
        categorical_features,
        recase=True,
        s3=True,
        mapping=True,
    )

    df = pd.concat([df_s3, df_s4], axis=0).reset_index(drop=True)

    for variable in tqdm(categorical_features):
        if variable != "SRF":
            sorted_by_count_unique_values = df[variable].value_counts()
            mapping = {k: v for v, k in enumerate(sorted_by_count_unique_values.index)}

            # save in json
            with open(os.path.join(directory, f"{variable}_mapping.json"), "w") as f:
                json.dump(mapping, f, indent=3)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--start_month", type=int, default=1)
    parser.add_argument("--start_year", type=int, default=2018)
    parser.add_argument("--revision", type=str, default="NAF2008")

    args = parser.parse_args()
    generate_mappings(
        start_month=args.start_month, start_year=args.start_year, revision=args.revision
    )
