import string

from nltk.corpus import stopwords

stopwords_ = set(stopwords.words("french") + ["a"])


def get_features(idx, data, features):
    dic_features = {
        feature: data.at[idx, feature]
        if isinstance(data.at[idx, feature], str)
        else "NaN"
        for feature in features
    }
    return " ".join([feat + "_" + mod for feat, mod in dic_features.items()])


def clean_lib(idx, data, X_names):
    # On supprime toutes les ponctuations
    lib = data.at[idx, X_names[0]].translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )
    # On supprime tous les chiffres
    lib = lib.translate(str.maketrans(string.digits, " " * len(string.digits)))

    # On supprime les stopwords et on renvoie les mots en minuscule
    lib_clean = " ".join(
        [x.lower() for x in lib.split() if x.lower() not in stopwords_]
    )

    if len(X_names) == 1:
        return lib_clean
    else:
        return lib_clean + " " + get_features(idx, data, X_names[1:])
