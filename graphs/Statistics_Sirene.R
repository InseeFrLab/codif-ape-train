rm(list = ls())
setwd("~/codification-ape/graphs")
library(aws.s3)
library(dplyr)
library(tidyr)
library(ggplot2)
library(cowplot)
library(arrow)
library(stringr)
library(readr)

source("theme_custom.R")
aws.s3::get_bucket("projet-ape", region = "", prefix = "data/extraction_sirene_20220712.parquet")


##### Data importation #####
df <- 
  aws.s3::s3read_using(
    FUN = arrow::read_parquet,
    # Mettre les options de FUN ici
    object = "/data/extraction_sirene_20220712.parquet",
    bucket = "projet-ape",
    opts = list("region" = "")
  )

naf <- read_csv("~/codification-ape/data/naf_extended.csv")

naf <- naf  %>% 
  mutate_all(~sub("\\.", "", .))%>%
  rename(APE_SICORE_NEW = NIV5)

# e ####
LongWord2remove = "\\bconforme au kbis\\b|\\bsans changement\\b|\\bsans activite\\b|\\bsans acitivite\\b|\\bactivite inchangee\\b|\\bactivites inchangees\\b|\\bsiege social\\b|\\ba definir\\b|\\ba preciser\\b|\\bci dessus\\b|\\bci desus\\b|\\bci desssus\\b|\\bvoir activit principale\\b|\\bvoir activite principale\\b|\\bvoir objet social\\b|\\bidem extrait kbis\\b|\\bn a plus a etre mentionne sur l extrait decret\\b|\\bcf statuts\\b|\\bactivite principale case\\b|\\bactivites principales case\\b|\\bactivite principale\\b|\\bactivites principales\\b|\\bidem case\\b|\\bvoir case\\b|\\baucun changement\\b|\\bsans modification\\b|\\bactivite non modifiee\\b|\\bactivite identique\\b|\\bpas de changement\\b|\\bpas de changement.{0,6}activite\\b"
Word2remove = "|\\bcode\\b|\\bape\\b|\\bape[a-z]{1}\\b|\\bnaf\\b|\\binchangee\\b|\\binchanges\\b|\\binchnagee\\b|\\bkbis\\b|\\bk bis\\b|\\binchangees\\b|\\bnp\\b|\\binchange\\b|\\bnc\\b|\\bidem\\b|\\bxx\\b|\\bxxx\\b|\\binconnue\\b|\\binconnu\\b|\\bvoir\\b|\\bannexe\\b"

#voir activite principale
df_ <-df%>%
  mutate(LIB_SICORE_NEW = tolower(LIB_SICORE),
         LIB_SICORE_NEW = str_replace_all(LIB_SICORE_NEW, "[:punct:]", " "), #Enleve ponctuation, strip le text, remplace "" par NAN
         LIB_SICORE_NEW = str_replace_all(LIB_SICORE_NEW, "\\d+", " "), # on supprime tous les digits, strip le text, remplace "" par NAN
         LIB_SICORE_NEW = str_remove_all(LIB_SICORE_NEW, paste0(LongWord2remove,Word2remove)), #Supprime une certaine liste de mot, strip le text, remplace "" par NAN
         LIB_SICORE_NEW = str_remove_all(LIB_SICORE_NEW, "\\b[a-z]{1}\\b"), # On supprime tous les "mots" d'une seule lettre
         LIB_SICORE_NEW = na_if(str_squish(LIB_SICORE_NEW), "")
         )%>%
  drop_na(APE_SICORE, LIB_SICORE_NEW)%>% #suppr les NAN
  select(LIA_NUM,DATE, APE_SICORE, LIB_SICORE, LIB_SICORE_NEW, LIB_LIASSE_ETAB_E71, EVT_SICORE, AUTO, NAT_SICORE, SURF)


LibDuplicated <- df_%>%subset(duplicated(LIB_SICORE_NEW))%>%pull(LIB_SICORE_NEW)%>%unique()

df_LibDuplicated <- df_%>%subset(LIB_SICORE_NEW %in% LibDuplicated)

# Pour chaque libellé on regarde le nombre de code APE différent
HarmonizedAPE <- df_LibDuplicated%>%
  group_by(LIB_SICORE_NEW)%>%
  summarize(
         APE_SICORE_NEW = names(which.max(table(APE_SICORE))),
         NbDiff = length(unique(APE_SICORE)),
         N = n(),
         Prop = NbDiff/N,
  )%>%
  filter((NbDiff>1) & (Prop<0.8))


www <- df_%>%left_join(HarmonizedAPE)%>%
  mutate(APE_SICORE_NEW = case_when(is.na(APE_SICORE_NEW) ~ APE_SICORE,
                                    T ~ APE_SICORE_NEW)
  )
  
ss<-www%>%filter(APE_SICORE_NEW != APE_SICORE)


# Cleaning the dataset for the preprocessing
# run model with 1 digit and 3 digit
# rewrite APE code for same libs
# faire des stats sur les classes
# faire oversampling seuil à 5000 ou 2500


www <- www%>%
  left_join(naf, by = "APE_SICORE_NEW")%>%
  select(LIA_NUM,DATE, APE_SICORE, LIB_SICORE, LIB_SICORE_NEW, LIB_LIASSE_ETAB_E71, EVT_SICORE, AUTO, NAT_SICORE, SURF)

qq <- www%>%count(APE_SICORE_NEW)
qq_NEW <- www%>%count(APE_SICORE)












