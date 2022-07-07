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

df <- df%>%
  select(LIA_NUM,DATE,APE_SICORE, LIB_SICORE, LIB_LIASSE_ETAB_E70, LIB_LIASSE_ETAB_E71,LIB_LIASSE_UL_U21, EVT_SICORE, AUTO, NAT_SICORE, SURF, CFE)


w <- dff%>%slice_head(n = 10000)

Word2remove = "CODE\\s|APE\\s|NAF\\s|INCHANGEE\\s|KBIS\\s|SANS CHANGEMENT\\s|
INCHANGEES\\s|NP\\s|SANS ACTIVITE\\s|ACTIVITE INCHANGEE\\s|INCHANGE\\s|
ACTIVITES INCHANGEES\\s|NC\\s|SIEGE SOCIAL\\s|A DEFINIR|A PRECISER"

dff <-df%>%
  select(LIA_NUM,DATE,APE_SICORE, LIB_SICORE, LIB_LIASSE_ETAB_E71, EVT_SICORE, AUTO, NAT_SICORE, SURF, CFE)%>%
  drop_na(APE_SICORE, LIB_SICORE)%>%
  mutate(LIB_SICORE = na_if(str_squish(str_replace_all(LIB_SICORE, "[:punct:]", "")),""))%>% #Enleve ponctuation, strip le text, remplace "" par NAN
  drop_na(LIB_SICORE)%>% #suppr les NAN
  mutate(LIB_SICORE = str_remove_all(LIB_SICORE, "\\d{4}\\s.{1}|\\d{4}.{1}"), # Supprime les codes APE dans libellés
         LIB_SICORE = na_if(str_squish(str_remove_all(LIB_SICORE, Word2remove)),""))%>% #Supprime une certaine liste de mot, strip le text, remplace "" par NAN
  drop_na(LIB_SICORE) #suppr les NAN
  

LibDuplicated <- dff%>%subset(duplicated(LIB_SICORE))%>%pull(LIB_SICORE)%>%unique()

df_LibDuplicated <- dff%>%subset(LIB_SICORE %in% LibDuplicated)%>%
  subset(LIB_SICORE == LIB_LIASSE_ETAB_E71)

# Pour chaque libellé on regarde le nombre de code APE différent
x <- df_LibDuplicated%>%
  group_by(LIB_SICORE)%>%
  summarize(
    NbDiff = length(unique(APE_SICORE)),
    N = n(),
    Prop = NbDiff/N
  )%>%
  subset(NbDiff>1)


qq <- df_LibDuplicated%>%
  subset((LIB_SICORE == "AVOCAT") & (CFE == "U"))%>%
  group_by(APE_SICORE)%>%
  summarise(
    N = n()
  )


aa <- df%>%subset((LIB_SICORE == "A PRECISER") & (APE_SICORE != "6910Z") & (LIB_SICORE == LIB_LIASSE_ETAB_E71))
aa <- df%>%subset((LIB_SICORE == "A PRECISER"))

# checker les libellés identiques avec code APE





