rm(list = ls())
setwd("~/codification-ape/graphs")
library(aws.s3)
library(dplyr)
source("theme_custom.R")
aws.s3::get_bucket("projet-ape", region = "", prefix = "data/preds_test.csv")

df_test <- 
  aws.s3::s3read_using(
    FUN = readr::read_csv,
    # Mettre les options de FUN ici
    object = "/data/preds_test.csv",
    bucket = "projet-ape",
    opts = list("region" = "")
  )
df_test <- df_test %>%
  mutate(Type="Test")

df_gu <- 
  aws.s3::s3read_using(
    FUN = readr::read_csv,
    # Mettre les options de FUN ici
    object = "/data/preds_gu.csv",
    bucket = "projet-ape",
    opts = list("region" = "")
  )

df <- df_gu %>%
  mutate(Type="GU")%>%
  bind_rows(df_test)

latex_percent <- function(x){
  x <- plyr::round_any(x,scales:::precision(x)/100)
  stringr::str_c(x * 100, "\\%")
}

## Aggrégation par moyenne
sample <- df %>%
  mutate(Results = predictions_5 == ground_truth_5) %>%
  group_by(ground_truth_1, Type)%>%
  summarise(
    Accuracy = mean(Results),
    Share = n()/nrow(df)*100
  )

ggplot(data = sample, aes(x=ground_truth_1, y=Accuracy, fill=Type))+
  geom_bar(stat = "identity", position = position_dodge())+
  geom_text(aes(label=round(Accuracy,2)), position=position_dodge(width=0.9), vjust=-0.25)+
  scale_fill_manual(values=c(Palette_col))+
  theme_custom()

## On regarde qu'au niveau 1

sample <- df %>%
  mutate(Results = predictions_1 == ground_truth_1) %>%
  group_by(ground_truth_1, Type)%>%
  summarise(
    Accuracy = mean(Results)
  )

ggplot(data = sample, aes(x=ground_truth_1, y=Accuracy, fill=Type))+
  geom_bar(stat = "identity", position = position_dodge())+
  geom_text(aes(label=round(Accuracy,2)), position=position_dodge(width=0.9), vjust=-0.25)+
  scale_fill_manual(values=c(Palette_col))+
  theme_custom()


## Stats desc

sample <- df %>%
  group_by(Type, ground_truth_1)%>%
  summarise(
    N = n()
  )%>%
  mutate(Share = N/ifelse(Type =="GU", nrow(df_gu), nrow(df_test)))

sample
ggplot(data = sample, aes(x=ground_truth_1, y=Share, fill=Type))+
  geom_bar(stat = "identity", position = position_dodge())+
  geom_text(aes(label=round(Share*100,1)), position=position_dodge(width=0.9), vjust=-0.25)+
  scale_y_continuous(labels = latex_percent)+
  scale_fill_manual(values=c(Palette_col))+
  theme_custom()




