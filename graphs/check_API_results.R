library(dplyr)
library(tidyr)
library(readr)

df_PY <- 
  aws.s3::s3read_using(
    FUN = read_csv,
    # Mettre les options de FUN ici
    object = "data/Preds_from_PY.csv",
    bucket = "projet-ape",
    opts = list("region" = "")
  )

df_API <- 
  aws.s3::s3read_using(
    FUN = read_csv,
    # Mettre les options de FUN ici
    object = "data/Preds_from_API.csv",
    bucket = "projet-ape",
    opts = list("region" = "")
  )


sum(df_PY$CODE_APE_1 == df_API$CODE_APE_1)/nrow(df_API)
sum(df_PY$CODE_APE_2 == df_API$CODE_APE_2)/nrow(df_API)
sum(df_PY$CODE_APE_3 == df_API$CODE_APE_3)/nrow(df_API)
sum(df_PY$CODE_APE_4 == df_API$CODE_APE_4)/nrow(df_API)
sum(df_PY$CODE_APE_5 == df_API$CODE_APE_5)/nrow(df_API)

sum(df_API$APE_SICORE == df_PY$CODE_APE_1)/nrow(df_API)
sum(df_API$APE_SICORE == df_API$CODE_APE_1)/nrow(df_API)
