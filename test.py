from src.utils.data import (
    CATEGORICAL_FEATURES,
    SURFACE_COLS,
    TEXT_FEATURE,
    get_raw_data,
    get_Y,
    mappings,
    get_df_naf
)
import numpy as np


from torchTextClassifiers.value_encoder import ValueEncoder, DictEncoder

revision = "NAF2025"
df_train, df_val, df_test = get_raw_data(revision=revision)
df_naf = get_df_naf(revision=revision)
Y = get_Y(revision=revision)

label_columns = [f"APE_NIV{i}" for i in range(1,5)] + [Y]
label_encoders = []
for key in [f"APE_NIV{i}" for i in range(1,6)]:
    mapping = mappings[Y][key]
    label_encoder = DictEncoder(mapping)
    label_encoders.append(label_encoder)


cat_encoders = {}
for col in ["CJ", "NAT", "TYP", "CRT"]:
    cat_encoders[col] = DictEncoder(mappings[col])

cat_encoders["SRF"] = DictEncoder({0: 0, 1: 1, 2: 2, 3: 3, 4: 4})
value_encoder = ValueEncoder(
    categorical_encoders=cat_encoders,
    label_encoder=label_encoders,)


value_encoder.transform(df_train[["CJ", "NAT", "TYP", "CRT"]].values)
value_encoder.transform_labels(df_train[label_columns].values)