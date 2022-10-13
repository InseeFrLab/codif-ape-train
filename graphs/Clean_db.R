rm(list = ls())
setwd("~/codification-ape/graphs")
library(aws.s3)
library(dplyr)
library(tidyr)
library(ggplot2)
library(cowplot)
library(arrow)
library(stringr)
library(stringi)
library(readr)

source("theme_custom.R")
aws.s3::get_bucket("projet-ape", region = "", prefix = "data/extraction_sirene_20220712.parquet")


##### Data importation #####
# Import the main database
df <- 
  aws.s3::s3read_using(
    FUN = arrow::read_parquet,
    # Mettre les options de FUN ici
    object = "/data/extraction_sirene_20220712.parquet",
    bucket = "projet-ape",
    opts = list("region" = "")
  )%>% 
  select(LIA_NUM, DATE, APE_SICORE, LIB_SICORE, AUTO, NAT_SICORE, EVT_SICORE, SURF)

# Import the table with all APE codes
naf <- read_csv("~/codification-ape/data/naf_extended.csv")
naf <- naf  %>% 
  mutate_all(~sub("\\.", "", .))%>%
  rename(APE_SICORE_NEW = NIV5)

Lib2remove <- arrow::read_csv_arrow("~/codification-ape/data/Lib2remove.csv")

##### Pré-traitement sur la base de données #####

Lib2remove <- Lib2remove%>%
  mutate(LIB_CLEAN = tolower(LIB))

dff <- df%>%
  select(LIA_NUM, DATE, LIB_SICORE, APE_SICORE)%>%
  mutate(LIB_CLEAN = tolower(LIB_SICORE), #On vire les trucs vide de sens qui doivent etre viré en amont du modèle
         LIB_CLEAN = str_replace_all(LIB_CLEAN, "\\bidem\\b|\\bvoir ci dessous\\b|\\[vide\\]|\\bundefined\\b|\\bpas d objet\\b|\\(voir ci dessus\\)|\\(voir extrait siege social\\/etablissement principal\\)|\\bcf activite principale\\b|\\bcf activite principale et objet\\b|\\bcf activites de l entreprise\\b|\\bcf activites principales de l entreprise\\b|\\bcf actvites principales\\b|\\bcf k bis\\b", " "),
         LIB_CLEAN = str_replace_all(LIB_CLEAN, "\\bcf le principales activites de l  entreprise\\b|\\bcf le sprincipale activites de l  entreprise\\b|\\bcf le sprincipales activites de l  entreprise\\b|\\bcf les activites principales de l  entreprise\\b|\\bcf les ppales activites de l  entreprise\\b|\\bcf les ppales activites de la ste\\b|\\bcf les principale activites de l  entreprise\\b|\\bcf les principales activites\\b|\\bcf les principales activites de l  entreprise\\b", " "),
         LIB_CLEAN = str_replace_all(LIB_CLEAN, "\\bcf les principales activites de l  entreprises\\b|\\bcf les principales activites ppales de l  entreprise\\b|\\bcf les principales activtes de l  entreprise\\b|\\bcf les principales acttivites de l  entreprise\\b|\\bcf les prinipales activites de l  entreprise\\b|\\bcf lesprincipales activites de l  entreprise\\b|\\bcf objet\\b|\\bcf obs\\b|\\bcf principales activite de l  entreprise\\b|\\bcf principales activites de l  entreprise\\b|cf rubrique \"principales activites de l entreprise\" idem|cf rubrique n2 ci dessus \\(743b\\)", " "),
         LIB_CLEAN = str_replace_all(LIB_CLEAN, "\\bcf supra\\b|\\bcf ci  dessus\\b|\\bcommerce de detail, idem case 2\\b|\\bextension a: voir ci dessus\\b|\\bid\\b|\\bid principales activites\\b|\\bid principales activites de l  entreprise\\b|\\bidem ci dessus\\b|idem \\( voir principales activites\\)|\\bidem  dessus\\b|\\bidem 1ere page\\b|\\bidem a principales activites de l  entreprise\\b|\\bidem activiet eprincipale\\b|\\bidem activite\\b|\\bidem activite 1ere page\\b", " "),
         LIB_CLEAN = str_replace_all(LIB_CLEAN, "\\bidem activite ci  dessus\\b|\\bidem activite de l  entreprise\\b|\\bidem activite enoncee ci  dessus\\b|\\bidem activite entreprise\\b|\\bidem activite generales\\b|\\bidem activite premiere page\\b|\\bidem activite principale\\b|\\bidem activite princippale\\b|\\bidem activite prinicpale\\b|\\bidem activite sur 1ere page\\b|\\bidem activites ci dessus\\b|\\bidem activites declarees au siege et principal\\b|\\bidem activites enoncees ci dessus\\b|\\bidem activites entreprise\\b|\\bidem activites principales\\b", " "),
         LIB_CLEAN = str_replace_all(LIB_CLEAN, "\\bidem activites principales de l entreprise\\b|\\bidem activites siege\\b|\\bidem activte principale\\b|\\bidem activtie 1ere page\\b|\\bidem au siege\\b|\\bidem au siege social\\b|\\bidem aux principales actiivtes\\b|\\bidem aux principales activites\\b|\\bidem case 13\\b|\\bidem ci dessous\\b|\\bidem ci dessus enoncee\\b|\\bidem cidessus\\b|\\bidem objet\\b|\\bidem premiere page\\b|\\bidem pricincipales activites de l entreprise\\b|\\bidem pricipales activites\\b|\\bidem principale activite\\b|\\bidem principales activite de l entreprise\\b", " "),
         LIB_CLEAN = str_replace_all(LIB_CLEAN, "\\bidem principales activite de l entreprises\\b|\\bidem principales activite l entreprise\\b|\\bidem principales activites\\b|\\bidem principales activites citees ci dessus\\b|\\bidem principales activites de l entreprises\\b|idem principales activites de l entreprise\\(objet\\)|\\bidem principales activites et objet social\\b|\\bidem principales activitse de l entreprise\\b|\\bidem que celle decrite plus haut\\b|\\bidem que ci dessus\\b|\\bidem que l activite decrite plus haut\\b", " "),
         LIB_CLEAN = str_replace_all(LIB_CLEAN, "\\bidem que les activites principales\\b|\\bidem que les activites principales ci dessus\\b|\\bidem que les activitges principales\\b|\\bidem que les principales activites\\b|\\bidem que les principales activites de l entreprise\\b|\\bidem que pour le siege\\b|\\bidem rubrique principales activites de l entreprise\\b|\\bidem siege\\b|idem siege \\+ voir observation|\\bidem siege et ets principal\\b|\\bidem siege social\\b|idem siege, \\(\\+ articles americains\\)|\\bidem societe\\b", " "),
         LIB_CLEAN = str_replace_all(LIB_CLEAN, "\\bidem voir activite principale\\b|\\bidem voir ci dessus\\b|\\bidentique a l objet social indique en case 2 de l imprime m2\\b|\\bidm ci dessus\\b|\\bnon indiquee\\b|\\bnon precise\\b|\\bnon precisee\\b|\\bnon precisees\\b|\\bvoir 1ere page\\b|\\bvoir activite ci dessus\\b|\\bvoir activite principale\\b|\\bvoir activite principale ci dessus\\b|\\bvoir activites principales\\b|\\bvoir cidessus\\b|\\bvoir idem ci dessus\\b|\\bvoir objet social\\b|\\bvoir page 1\\b|\\bvoir page precedente\\b|\\bvoir plus haut\\b|\\bvoir princiale activite\\b|\\bcase precedente\\b", " "),
         LIB_CLEAN = str_replace_all(LIB_CLEAN, "\\bvoir princiales activites\\b|\\bvoir princiapales activites\\b|\\bvoir princiaples activites\\b|\\bvoir principale activite\\b|\\bvoir principales activites\\b|\\bvoir principales activites de l entreprise\\b|\\bvoir principales actvites\\b|\\bvoir principalesactivites\\b|\\bvoir principles activites\\b|\\bvoir rubrique principales activites de l entreprise\\b|\\bvoir sur la 1ere page\\b|\\bvoir dessus\\b", " "),  
         LIB_CLEAN = str_replace_all(LIB_CLEAN, "voir: \"principales activite de l entreprise\"voir: \"principales activites de l entreprises\"|voir: \"principales activites de l entrprise\"|voir: \"principales activites en entreprise\"|\\bconforme au kbis\\b|\\bsans changement\\b|\\bsans activite\\b|\\bsans acitivite\\b|\\bactivite inchangee\\b|\\bactivites inchangees\\b|\\bsiege social\\b|\\ba definir\\b|\\ba preciser\\b|\\bci dessus\\b|\\bci desus\\b|\\bci desssus\\b|\\bvoir activit principale\\b", " "),  
         LIB_CLEAN = str_replace_all(LIB_CLEAN, "\\bidem extrait kbis\\b|\\bn a plus a etre mentionne sur l extrait decret\\b|\\bcf statuts\\b|\\bactivite principale case\\b|\\bactivites principales case\\b|\\bactivite principale\\b|\\bactivites principales\\b|\\bvoir case\\b|\\baucun changement\\b|\\bsans modification\\b|\\bactivite non modifiee\\b|\\bactivite identique\\b|\\bpas de changement\\b|\\bcode\\b|\\bape\\b|\\bnaf\\b|\\binchangee\\b|\\binchnagee\\b|\\bkbis\\b|\\bk bis\\b|\\binchangees\\b|\\bnp\\b|\\binchange\\b|\\bidem cadre precedent\\b", " "),  
         LIB_CLEAN = str_replace_all(LIB_CLEAN, "\\bnc\\b|\\bxx\\b|\\bxxx\\b|\\baa\\b|\\baaa\\b|\\binconnue\\b|\\binconnu\\b|\\bvoir\\b|\\bannexe\\b|\\bmo\\b|\\biem\\b|\\binchanges\\b|\\bactivite demeure\\b|\\bactivite inchangée\\b|\\bactivite demeure\\b|\\bactivite inchangée\\b|\\bnon renseignee\\b|\\bneant\\b|\\bnon renseigne\\b", " ")
  )%>%
  mutate(LIB_CLEAN = str_replace_all(LIB_CLEAN, "e-", "e"),
         LIB_CLEAN = str_replace_all(LIB_CLEAN, "\\be\\s", " e"),
  )%>%
  mutate(LIB_CLEAN = str_replace_all(LIB_CLEAN, "[:punct:]", " ")
  )%>%
  mutate(LIB_CLEAN = str_replace_all(LIB_CLEAN, "\\bcode\\b|\\bcadre\\b|\\bape\\b|\\bape[a-z]{1}\\b|\\bnaf\\b|\\binchangee\\b|\\binchnagee\\b|\\bkbis\\b|\\bk bis\\b|\\binchangees\\b|\\bnp\\b|\\binchange\\b|\\bnc\\b|\\bidem\\b|\\bxx\\b|\\bxxx\\b|\\bidem case\\b|\\binchanges\\b|\\bmo\\b|\biem\\b|\\bci dessus\\b", "")
  )%>%
  mutate(LIB_CLEAN = str_replace_all(LIB_CLEAN, "\\b[a-z]{1}\\b", " ")
  )%>%
  mutate(LIB_CLEAN = str_replace_all(LIB_CLEAN, "[\\d+]", " ")
  )%>%
  mutate(LIB_CLEAN = str_replace_all(LIB_CLEAN, "\\b[a-z]{1}\\b", " ")
  )%>%
  mutate(LIB_CLEAN = na_if(str_squish(LIB_CLEAN), "")
  )%>% 
  drop_na(APE_SICORE, LIB_CLEAN)

LibDuplicated <- dff%>%subset(duplicated(LIB_CLEAN))%>%pull(LIB_CLEAN)%>%unique()
df_LibDuplicated <- dff%>%subset(LIB_CLEAN %in% LibDuplicated)

# Get the most frequent code for a given lib and compute the frequency of this code
APE_libs_proportions <- df_LibDuplicated%>%
  group_by(LIB_CLEAN)%>%
  summarize(
    APE_SICORE_NEW = names(which.max(table(APE_SICORE))),
    NbDiff = length(unique(APE_SICORE)),
    N = n(),
    Prop = NbDiff/N,
  )%>%
  filter(NbDiff>1)
  
# On définit un seuil en dessous duquel on attribut le code APE le plus fréquent pour un lib unique
libs_to_harmonised <- APE_libs_proportions%>%
  filter(Prop < 0.48)

# On récupère les libs dont on ne connait pas assez bien le bon code APE pour les supprimer
libs_to_remove <- APE_libs_proportions%>%
  filter(Prop >= 0.48)%>%
  pull(LIB_CLEAN)

# On regarde combien d'observation cela représente
APE_libs_proportions%>%
  filter(Prop >= 0.48)%>%
  pull(N)%>%
  sum()

# reaffecter les bon codes
# reconstruire la base
zz <- dff%>% left_join(APE_libs_proportions%>%select(LIB_CLEAN, APE_SICORE_NEW))%>%
  mutate(APE_SICORE_NEW = case_when(is.na(APE_SICORE_NEW) ~ APE_SICORE,
                                    T ~ APE_SICORE_NEW)
  )%>%
  filter(!(LIB_CLEAN %in% libs_to_remove))
  
  
  select(LIA_NUM,DATE, APE_SICORE_NEW, LIB_SICORE, EVT_SICORE, AUTO, NAT_SICORE, SURF)%>%
  rename(APE_SICORE = APE_SICORE_NEW)


dd<- zz %>% filter(APE_SICORE != APE_SICORE_NEW)

aa<-df_LibDuplicated %>%filter(LIB_CLEAN == "livraison de repas domicile velo gestion de parc de trottinettes")


#"\\bidem\\b|\\bvoir ci dessous\\b|\[vide\]|\\bundefined\\b|\\bpas d objet\\b|\(voir ci dessus\)|\(voir extrait siege social\/etablissement principal\)|\\bcf activite principale\\b|\\bcf activite principale et objet\\b|\\bcf activites de l entreprise\\b|\\bcf activites principales de l entreprise\\b|\\bcf actvites principales\\b|\\bcf k bis\\b|\\bcf le principales activites de l  entreprise\\b|\\bcf le sprincipale activites de l  entreprise\\b|\\bcf le sprincipales activites de l  entreprise\\b|\\bcf les activites principales de l  entreprise\\b|\\bcf les ppales activites de l  entreprise\\b|\\bcf les ppales activites de la ste\\b|\\bcf les principale activites de l  entreprise\\b|\\bcf les principales activites\\b|\\bcf les principales activites de l  entreprise\\b|\\bcf les principales activites de l  entreprises\\b|\\bcf les principales activites ppales de l  entreprise\\b|\\bcf les principales activtes de l  entreprise\\b|\\bcf les principales acttivites de l  entreprise\\b|\\bcf les prinipales activites de l  entreprise\\b|\\bcf lesprincipales activites de l  entreprise\\b|\\bcf objet\\b|\\bcf obs\\b|\\bcf principales activite de l  entreprise\\b|\\bcf principales activites de l  entreprise\\b|cf rubrique \"principales activites de l entreprise\" idem|cf rubrique n2 ci dessus \(743b\)|\\bcf supra\\b|\\bcf ci  dessus\\b|\\bcommerce de detail, idem case 2\\b|\\bextension a: voir ci dessus\\b|\\bid\\b|\\bid principales activites\\b|\\bid principales activites de l  entreprise\\b|\\bidem ci dessus\\b|idem \( voir principales activites\)|\\bidem  dessus\\b|\\bidem 1ere page\\b|\\bidem a principales activites de l  entreprise\\b|\\bidem activiet eprincipale\\b|\\bidem activite\\b|\\bidem activite 1ere page\\b|\\bidem activite ci  dessus\\b|\\bidem activite de l  entreprise\\b|\\bidem activite enoncee ci  dessus\\b|\\bidem activite entreprise\\b|\\bidem activite generales\\b|\\bidem activite premiere page\\b|\\bidem activite principale\\b|\\bidem activite princippale\\b|\\bidem activite prinicpale\\b|\\bidem activite sur 1ere page\\b|\\bidem activites ci dessus\\b|\\bidem activites declarees au siege et principal\\b|\\bidem activites enoncees ci dessus\\b|\\bidem activites entreprise\\b|\\bidem activites principales\\b|\\bidem activites principales de l entreprise\\b|\\bidem activites siege\\b|\\bidem activte principale\\b|\\bidem activtie 1ere page\\b|\\bidem au siege\\b|\\bidem au siege social\\b|\\bidem aux principales actiivtes\\b|\\bidem aux principales activites\\b|\\bidem case 13\\b|\\bidem ci dessous\\b|\\bidem ci dessus enoncee\\b|\\bidem cidessus\\b|\\bidem objet\\b|\\bidem premiere page\\b|\\bidem pricincipales activites de l entreprise\\b|\\bidem pricipales activites\\b|\\bidem principale activite\\b|\\bidem principales activite de l entreprise\\b|\\bidem principales activite de l entreprises\\b|\\bidem principales activite l entreprise\\b|\\bidem principales activites\\b|\\bidem principales activites citees ci dessus\\b|\\bidem principales activites de l entreprises\\b|idem principales activites de l entreprise\(objet\)|\\bidem principales activites et objet social\\b|\\bidem principales activitse de l entreprise\\b|\\bidem que celle decrite plus haut\\b|\\bidem que ci dessus\\b|\\bidem que l activite decrite plus haut\\b|\\bidem que les activites principales\\b|\\bidem que les activites principales ci dessus\\b|\\bidem que les activitges principales\\b|\\bidem que les principales activites\\b|\\bidem que les principales activites de l entreprise\\b|\\bidem que pour le siege\\b|\\bidem rubrique principales activites de l entreprise\\b|\\bidem siege\\b|idem siege \+ voir observation|\\bidem siege et ets principal\\b|\\bidem siege social\\b|idem siege, \(\+ articles americains\)|\\bidem societe\\b|\\bidem voir activite principale\\b|\\bidem voir ci dessus\\b|\\bidentique a l objet social indique en case 2 de l imprime m2\\b|\\bidm ci dessus\\b|\\bnon indiquee\\b|\\bnon precise\\b|\\bnon precisee\\b|\\bnon precisees\\b|\\bvoir 1ere page\\b|\\bvoir activite ci dessus\\b|\\bvoir activite principale\\b|\\bvoir activite principale ci dessus\\b|\\bvoir activites principales\\b|\\bvoir cidessus\\b|\\bvoir idem ci dessus\\b|\\bvoir objet social\\b|\\bvoir page 1\\b|\\bvoir page precedente\\b|\\bvoir plus haut\\b|\\bvoir princiale activite\\b|\\bvoir princiales activites\\b|\\bvoir princiapales activites\\b|\\bvoir princiaples activites\\b|\\bvoir principale activite\\b|\\bvoir principales activites\\b|\\bvoir principales activites de l entreprise\\b|\\bvoir principales actvites\\b|\\bvoir principalesactivites\\b|\\bvoir principles activites\\b|\\bvoir rubrique principales activites de l entreprise\\b|\\bvoir sur la 1ere page\\b|\\bvoir dessus\\b|voir: \"principales activite de l entreprise\"|voir: \"principales activites de l entreprises\"|voir: \"principales activites de l entrprise\"|voir: \"principales activites en entreprise\"|\\bconforme au kbis\\b|\\bsans changement\\b|\\bsans activite\\b|\\bsans acitivite\\b|\\bactivite inchangee\\b|\\bactivites inchangees\\b|\\bsiege social\\b|\\ba definir\\b|\\ba preciser\\b|\\bci dessus\\b|\\bci desus\\b|\\bci desssus\\b|\\bvoir activit principale\\b|\\bidem extrait kbis\\b|\\bn a plus a etre mentionne sur l extrait decret\\b|\\bcf statuts\\b|\\bactivite principale case\\b|\\bactivites principales case\\b|\\bactivite principale\\b|\\bactivites principales\\b|\\bvoir case\\b|\\baucun changement\\b|\\bsans modification\\b|\\bactivite non modifiee\\b|\\bactivite identique\\b|\\bpas de changement\\b|\\bcode\\b|\\bape\\b|\\bnaf\\b|\\binchangee\\b|\\bbinchanges\\b|\\binchnagee\\b|\\bkbis\\b|\\bk bis\\b|\\binchangees\\b|\\bnp\\b|\\binchange\\b|\\bnc\\b|\\bxx\\b|\\bxxx\\b|\\binconnue\\b|\\binconnu\\b|\\bvoir\\b|\\bannexe\\b"

aws.s3::s3write_using(
  df_new,
  FUN = arrow::write_parquet,
  # Mettre les options de FUN ici
  multipart = TRUE,
  object = "/data/extraction_sirene_20220712_harmonised.parquet",
  bucket = "projet-ape",
  opts = list("region" = "")
)



