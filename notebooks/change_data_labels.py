# pylint: disable=C0301,C0114


import re

import numpy as np
import pandas as pd

df = pd.read_parquet(
    "../data/extraction_sirene_20220712_harmonized_20221014.parquet", engine="pyarrow"
)
df = df[
    ["DATE", "APE_SICORE", "LIB_SICORE", "AUTO", "NAT_SICORE", "EVT_SICORE", "SURF"]
]

# define replacement patterns
replacements = {
    r"\bidem\b|\bvoir ci dessous\b|\[vide\]|\bundefined\b|\bpas d objet\b|\(voir ci dessus\)|\(voir extrait siege social\\/etablissement principal\)|\bcf activite principale\b|\bcf activite principale et objet\b|\bcf activites de l entreprise\b|\bcf activites principales de l entreprise\b|\bcf actvites principales\b|\bcf k bis\b|\bcf le principales activites de l  entreprise\b|\bcf le sprincipale activites de l  entreprise\b|\bcf le sprincipales activites de l  entreprise\b|\bcf les activites principales de l  entreprise\b|\bcf les ppales activites de l  entreprise\b|\bcf les ppales activites de la ste\b|\bcf les principale activites de l  entreprise\b|\bcf les principales activites\b|\bcf les principales activites de l  entreprise\b|\bcf les principales activites de l  entreprises\b|\bcf les principales activites ppales de l  entreprise\b|\bcf les principales activtes de l  entreprise\b|\bcf les principales acttivites de l  entreprise\b|\bcf les prinipales activites de l  entreprise\b|\bcf lesprincipales activites de l  entreprise\b|\bcf objet\b|\bcf obs\b|\bcf principales activite de l  entreprise\b|\bcf principales activites de l  entreprise\b|cf rubrique \"principales activites de l entreprise\" idem|cf rubrique n2 ci dessus \(743b\)|\bcf supra\b|\bcf ci  dessus\b|\bcommerce de detail, idem case 2\b|\bextension a: voir ci dessus\b|\bid\b|\bid principales activites\b|\bid principales activites de l  entreprise\b|\bidem ci dessus\b|idem \( voir principales activites\)|\bidem  dessus\b|\bidem 1ere page\b|\bidem a principales activites de l  entreprise\b|\bidem activiet eprincipale\b|\bidem activite\b|\bidem activite 1ere page\b|\bidem activite ci  dessus\b|\bidem activite de l  entreprise\b|\bidem activite enoncee ci  dessus\b|\bidem activite entreprise\b|\bidem activite generales\b|\bidem activite premiere page\b|\bidem activite principale\b|\bidem activite princippale\b|\bidem activite prinicpale\b|\bidem activite sur 1ere page\b|\bidem activites ci dessus\b|\bidem activites declarees au siege et principal\b|\bidem activites enoncees ci dessus\b|\bidem activites entreprise\b|\bidem activites principales\b|\bidem activites principales de l entreprise\b|\bidem activites siege\b|\bidem activte principale\b|\bidem activtie 1ere page\b|\bidem au siege\b|\bidem au siege social\b|\bidem aux principales actiivtes\b|\bidem aux principales activites\b|\bidem case 13\b|\bidem ci dessous\b|\bidem ci dessus enoncee\b|\bidem cidessus\b|\bidem objet\b|\bidem premiere page\b|\bidem pricincipales activites de l entreprise\b|\bidem pricipales activites\b|\bidem principale activite\b|\bidem principales activite de l entreprise\b|\bidem principales activite de l entreprises\b|\bidem principales activite l entreprise\b|\bidem principales activites\b|\bidem principales activites citees ci dessus\b|\bidem principales activites de l entreprises\b|idem principales activites de l entreprise\(objet\)|\bidem principales activites et objet social\b|\bidem principales activitse de l entreprise\b|\bidem que celle decrite plus haut\b|\bidem que ci dessus\b|\bidem que l activite decrite plus haut\b|\bidem que les activites principales\b|\bidem que les activites principales ci dessus\b|\bidem que les activitges principales\b|\bidem que les principales activites\b|\bidem que les principales activites de l entreprise\b|\bidem que pour le siege\b|\bidem rubrique principales activites de l entreprise\b|\bidem siege\b|idem siege \+ voir observation|\bidem siege et ets principal\b|\bidem siege social\b|idem siege, \(\+ articles americains\)|\bidem societe\b|\bidem voir activite principale\b|\bidem voir ci dessus\b|\bidentique a l objet social indique en case 2 de l imprime m2\b|\bidm ci dessus\b|\bnon indiquee\b|\bnon precise\b|\bnon precisee\b|\bnon precisees\b|\bvoir 1ere page\b|\bvoir activite ci dessus\b|\bvoir activite principale\b|\bvoir activite principale ci dessus\b|\bvoir activites principales\b|\bvoir cidessus\b|\bvoir idem ci dessus\b|\bvoir objet social\b|\bvoir page 1\b|\bvoir page precedente\b|\bvoir plus haut\b|\bvoir princiale activite\b|\bvoir princiales activites\b|\bvoir princiapales activites\b|\bvoir princiaples activites\b|\bvoir principale activite\b|\bvoir principales activites\b|\bvoir principales activites de l entreprise\b|\bvoir principales actvites\b|\bvoir principalesactivites\b|\bvoir principles activites\b|\bvoir rubrique principales activites de l entreprise\b|\bvoir sur la 1ere page\b|\bvoir dessus\b|voir: \"principales activite de l entreprise\"|voir: \"principales activites de l entreprises\"|voir: \"principales activites de l entrprise\"|voir: \"principales activites en entreprise\"|\bconforme au kbis\b|\bsans changement\b|\bsans activite\b|\bsans acitivite\b|\bactivite inchangee\b|\bactivites inchangees\b|\bsiege social\b|\ba definir\b|\ba preciser\b|\bci dessus\b|\bci desus\b|\bci desssus\b|\bvoir activit principale\b|\bidem extrait kbis\b|\bn a plus a etre mentionne sur l extrait decret\b|\bcf statuts\b|\bactivite principale case\b|\bactivites principales case\b|\bactivite principale\b|\bactivites principales\b|\bvoir case\b|\baucun changement\b|\bsans modification\b|\bactivite non modifiee\b|\bactivite identique\b|\bpas de changement\b|\bcode\b|\bape\b|\bnaf\b|\binchangee\b|\binchnagee\b|\bkbis\b|\bk bis\b|\binchangees\b|\bnp\b|\binchange\b|\bnc\b|\bxx\b|\bxxx\b|\binconnue\b|\binconnu\b|\bvoir\b|\bannexe\b|\bmo\b|\biem\b|\binchanges\b|\bactivite demeure\b|\bactivite inchangée\b|\bcase precedente\b|\bidem cadre precedent\b|\bactivite demeure\b|\bactivite inchangée\b|\bnon renseignee\b|\bneant\b|\bnon renseigne\b": " ",
    r"e-": "e",
    r"\be\s": " e",
    r"[^\w\s]": " ",
    r"\bcode\b|\bcadre\b|\bape\b|\bape[a-z]{1}\b|\bnaf\b|\binchangee\b|\binchnagee\b|\bkbis\b|\bk bis\b|\binchangees\b|\bnp\b|\binchange\b|\bnc\b|\bidem\b|\bxx\b|\bxxx\b|\baa\b|\baaa\b|\bidem cadre precedent\b|\bidem case\b|\binchanges\b|\bmo\b|\biem\b|\bci dessus\b|\bet\b": "",
    r"[\d+]": " ",
    r"^\s+|\s+$": "",
}

df["LIB_CLEAN"] = df["LIB_SICORE"].str.lower()

# apply replacements to LIB_CLEAN column
for pattern, replacement in replacements.items():
    df["LIB_CLEAN"] = df["LIB_CLEAN"].str.replace(pattern, replacement, regex=True)

df["LIB_CLEAN"] = df["LIB_CLEAN"].apply(lambda x: re.sub(r"\b\w\b", "", x))

# define replacement patterns
replacements = {
    # On supprime les espaces multiples
    r"\s\s+": " ",
    # On strip les libellés
    r"^\s+|\s+$": "",
    # On remplace les empty string par des NaN
    r"^\s*$": np.nan,
}

# apply replacements to LIB_CLEAN column
for pattern, replacement in replacements.items():
    df["LIB_CLEAN"] = df["LIB_CLEAN"].str.replace(pattern, replacement, regex=True)

df.dropna(subset=["APE_SICORE", "LIB_CLEAN"], inplace=True)

# We apply some change from equipe metier
idx = (
    df["LIB_CLEAN"].str.contains(
        "lmnp|loueur en meuble non professionnel|loueur bailleur non professionnel|location meublee non professionnelle|loueur meuble non professionnel|loueurs en meubles non professionnels|loueur en meubl non professionnel|loueur en meubles non professionnel",
        case=False,
    )
) & ((df["AUTO"].isin(["E", "L", "S", "X", "I"])) | (df["AUTO"].isnull()))

df["APE_SICORE"] = np.where(idx, "6820A", df["APE_SICORE"])
df.to_parquet("data_sirene3.parquet")
