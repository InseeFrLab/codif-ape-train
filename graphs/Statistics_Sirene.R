rm(list = ls())
setwd("~/codification-ape/graphs")
library(aws.s3)
library(dplyr)
library(tidyr)
library(ggplot2)
library(cowplot)
library(arrow)
library(stringr)

source("theme_custom.R")
aws.s3::get_bucket("projet-ape", region = "", prefix = "data/extraction_sirene_20220628.parquet")


##### Data importation #####
df <- 
  aws.s3::s3read_using(
    FUN = arrow::read_parquet,
    # Mettre les options de FUN ici
    object = "/data/extraction_sirene_20220628.parquet",
    bucket = "projet-ape",
    opts = list("region" = "")
  )

#w <- df%>%slice_head(n = 100000)

LongWord2remove = "\\bconforme au kbis\\b|\\bsans changement\\b|\\bsans acitivite\\b|\\bactivite inchangee\\b|\\bactivites inchangees\\b|\\bsiege social\\b|\\ba definir\\b|\\ba preciser\\b|\\bci desus\\b|\\bvoir activit principale\\b|\\bvoir activite principale\\b|\\bvoir objet social\\b|\\bidem extrait kbis\\b|\\bn a plus a etre mentionne sur l extrait decret\\b|\\bcf statuts\\b|\\bactivite principale case\\b|\\bactivites principales case\\b|\\bactivite principale\\b|\\bactivites principales\\b|\\bidem case\\b|\\bvoir case\\b|\\baucun changement\\b|\\b	
sans modification\\b|\\bactivite non modifiee\\b"
Word2remove = "\\bcode\\b|\\bape\\b|\\bape[a-z]{1}\\b|\\bnaf\\b|\\binchangee\\b|\\binchnagee\\b|\\bkbis\\b|\\bk bis\\b|\\binchangees\\b|\\bnp\\b|\\binchange\\b|\\bnc\\b|\\bidem\\b|\\bci desus\\b|\\bxx\\b|\\bxxx\\b"

#voir activite principale
dff <-df%>%
  mutate(LIB_SICORE_NEW = tolower(LIB_SICORE),
         LIB_SICORE_NEW = str_replace_all(LIB_SICORE_NEW, "[:punct:]", " "), #Enleve ponctuation, strip le text, remplace "" par NAN
         LIB_SICORE_NEW = str_replace_all(LIB_SICORE_NEW, "\\d+", " "), # on supprime tous les digits, strip le text, remplace "" par NAN
         LIB_SICORE_NEW = str_remove_all(LIB_SICORE_NEW, paste0(LongWord2remove,Word2remove)), #Supprime une certaine liste de mot, strip le text, remplace "" par NAN
         LIB_SICORE_NEW = str_remove_all(LIB_SICORE_NEW, "\\b[a-z]{1}\\b"), # On supprime tous les "mots" d'une seule lettre
         LIB_SICORE_NEW = na_if(str_squish(LIB_SICORE_NEW), "")
         )%>%
  drop_na(APE_SICORE, LIB_SICORE_NEW)%>% #suppr les NAN
  select(LIA_NUM,DATE, APE_SICORE, LIB_SICORE, LIB_SICORE_NEW, LIB_LIASSE_ETAB_E71, EVT_SICORE, AUTO, NAT_SICORE, SURF,CFE)

#mutate(LIB_SICORE = str_remove_all(LIB_SICORE, "\\d{4}\\s.{1}|\\d{4}.{1}"), # Supprime les codes APE dans libellés
LibDuplicated <- dff%>%subset(duplicated(LIB_SICORE_NEW))%>%pull(LIB_SICORE_NEW)%>%unique()

df_LibDuplicated <- dff%>%subset(LIB_SICORE_NEW %in% LibDuplicated)

# Pour chaque libellé on regarde le nombre de code APE différent
x <- df_LibDuplicated%>%
  group_by(LIB_SICORE_NEW)%>%
  summarize(
         APE_SICORE_NEW = names(which.max(table(APE_SICORE))),
         NbDiff = length(unique(APE_SICORE)),
         N = n(),
         Prop = NbDiff/N,
  )%>%
  filter((NbDiff>1) & (Prop<0.8))


www <- dff%>%left_join(x)%>%
  mutate(APE_SICORE_NEW = case_when(is.na(APE_SICORE_NEW) ~ APE_SICORE,
                                    T ~ APE_SICORE_NEW)
  )
  
ss<-www%>%filter(APE_SICORE_NEW != APE_SICORE)


# Cleaning the dataset for the preprocessing
# run model with 1 digit and 3 digit
# rewrite APE code for same libs
# faire des stats sur les classes
# faire oversampling seuil à 5000 ou 2500

q<-www%>%count(APE_SICORE_NEW)
  
df%>%count(APE_SICORE)












