from nltk.corpus import stopwords
import string

stopwords_ = set(stopwords.words('french') + ['a'])


def clean_lib(lib):
    # On supprime toutes les ponctuations
    lib = lib.translate(
        str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    )
    # On supprime tous les chiffres
    lib = lib.translate(
        str.maketrans(string.digits, ' ' * len(string.digits))
    )

    # On supprime les stopwords et on renvoie les mots en majuscule
    return " ".join([x.lower() for x in lib.split() if x.lower() not in stopwords_])
