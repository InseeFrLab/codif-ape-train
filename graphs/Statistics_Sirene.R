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

w <- df%>%slice_head(n = 100000)

LongWord2remove = "\\bconforme au kbis\\b|\\bsans changement\\b|\\bsans acitivite\\b|\\bactivite inchangee\\b|\\bactivites inchangees\\b|\\bsiege social\\b|\\ba definir\\b|\\ba preciser\\b|\\bci desus\\b|\\bvoir activit principale\\b|\\bvoir activite principale\\b|\\bvoir objet social\\b|\\bidem extrait kbis\\b|\\bn a plus a etre mentionne sur l extrait decret\\b|\\bcf statuts\\b|\\bactivite principale case\\b|\\bactivites principales case\\b|\\bactivite principale\\b|\\bactivites principales\\b|\\bidem case\\b|\\bvoir case\\b"
Word2remove = "\\bcode\\b|\\bape\\b|\\bape[a-z]{1}\\b|\\bnaf\\b|\\binchangee\\b|\\binchnagee\\b|\\bkbis\\b|\\bk bis\\b|\\binchangees\\b|\\bnp\\b|\\binchange\\b|\\bnc\\b|\\bidem\\b|\\bci desus\\b"

#voir activite principale
dff <-df%>%
  mutate(LIB_SICORE2 = tolower(LIB_SICORE),
         LIB_SICORE2 = str_replace_all(LIB_SICORE2, "[:punct:]", " "), #Enleve ponctuation, strip le text, remplace "" par NAN
         LIB_SICORE2 = str_replace_all(LIB_SICORE2, "\\d+", " "), # on supprime tous les digits, strip le text, remplace "" par NAN
         LIB_SICORE2 = str_remove_all(LIB_SICORE2, Word2remove), #Supprime une certaine liste de mot, strip le text, remplace "" par NAN
         LIB_SICORE2 = str_remove_all(LIB_SICORE2, "\\b[a-z]{1}\\b"), # On supprime tous les "mots" d'une seule lettre
         LIB_SICORE2 = na_if(str_squish(LIB_SICORE2), "")
         )%>%
  drop_na(APE_SICORE, LIB_SICORE2)%>% #suppr les NAN
  select(LIA_NUM,DATE, APE_SICORE, LIB_SICORE, LIB_SICORE2, LIB_LIASSE_ETAB_E71, EVT_SICORE, AUTO, NAT_SICORE, SURF,CFE)

#mutate(LIB_SICORE = str_remove_all(LIB_SICORE, "\\d{4}\\s.{1}|\\d{4}.{1}"), # Supprime les codes APE dans libellés
LibDuplicated <- dff%>%subset(duplicated(LIB_SICORE2))%>%pull(LIB_SICORE2)%>%unique()

df_LibDuplicated <- dff%>%subset(LIB_SICORE2 %in% LibDuplicated)
#%>%
#  subset(LIB_SICORE == LIB_LIASSE_ETAB_E71)

# Pour chaque libellé on regarde le nombre de code APE différent
x <- df_LibDuplicated%>%
  group_by(LIB_SICORE2)%>%
  summarize(
         NbDiff = length(unique(APE_SICORE)),
         N = n(),
         Prop = NbDiff/N,
  )
  
  x<-x%>%
  filter((NbDiff>1) & (NbDiff<N))


APE_SICORE_NEW = names(which.max(table(APE_SICORE))),



www <- dfff%>%left_join(x)%>%
  mutate(APE_SICORE_NEW = case_when(is.na(APE_SICORE_NEW) ~ APE_SICORE,
                                    T ~ APE_SICORE_NEW)
  )
  
ss<-www%>%filter(APE_SICORE_NEW != APE_SICORE)




