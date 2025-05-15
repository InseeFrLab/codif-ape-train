import pandas as pd

from preprocessors import PytorchPreprocessor

from ..models.forms import SingleForm


def preprocess_inputs(
    inputs: list[SingleForm], text_feature, textual_features, categorical_features
) -> dict:
    """
    Preprocess both single and batch inputs using shared logic.
    """

    df = pd.DataFrame([form.model_dump() for form in inputs])

    df = df.rename(
        {
            "description_activity": text_feature,
            "other_nature_activity": textual_features[0],
            "precision_act_sec_agricole": textual_features[1],
            "type_form": categorical_features[0],
            "nature": categorical_features[1],
            "surface": categorical_features[2],
            "cj": categorical_features[3],
            "activity_permanence_status": categorical_features[4],
        },
        axis=1,
    )

    for feature in textual_features:
        df[feature] = df[feature].fillna(value="")
    for feature in categorical_features:
        df[feature] = df[feature].fillna(value="NaN")

    df[text_feature] = PytorchPreprocessor.clean_text_feature(
        df[text_feature], remove_stop_words=True
    )
    df = PytorchPreprocessor.clean_textual_features(df, textual_features)
    df[text_feature] = df[text_feature] + df[textual_features].apply(lambda x: "".join(x), axis=1)

    # Clean categorical features
    df = PytorchPreprocessor.clean_categorical_features(
        df, categorical_features=categorical_features
    )
    df = df.drop(columns=textual_features)

    return df
