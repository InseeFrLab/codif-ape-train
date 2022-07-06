rm(list = ls())
setwd("~/codification-ape/graphs")
library(aws.s3)
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(cowplot)

source("theme_custom.R")
aws.s3::get_bucket("projet-ape", region = "", prefix = "data/preds_test.csv")


##### Data importation #####
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

##### Matrice de confusion niveau 1 ##### 
PlotConfusionMatrix <- function(data, type){
  TotalSize <- data%>%subset(Type %in% type)%>%nrow()
  
  Shares <- data%>%
    subset(Type %in% type)%>%
    group_by(ground_truth_1, Type)%>%
    summarise(
      N = n(),
    )%>%
    mutate(Share = N/TotalSize*100)
  
  sample <- data%>%
    subset(Type %in% type)%>%
    mutate(ground_truth_1 = factor(ground_truth_1, levels = Shares$ground_truth_1),
           predictions_1 = factor(predictions_1, levels = Shares$ground_truth_1))%>%
    select(ground_truth_1, predictions_1)
  
  cm <- confusionMatrix(sample$ground_truth_1, sample$predictions_1)
  
  sample <- cm$table %>%
    data.frame() %>% 
    mutate(Prediction = factor(Prediction, levels = Shares$ground_truth_1)) %>%
    group_by(Reference) %>% 
    mutate(
      total = sum(Freq),
      perc = Freq / total*100
    )
  
  
  NewTitle <- paste0(sample$Reference, "\n(", rep(round(Shares$Share, 1), each=length(Shares$ground_truth_1)), "%)")
  names(NewTitle) <- sample$Reference 
  
  sample <- sample %>%
    mutate(Reference = recode(Reference, !!!NewTitle)
    )
  
  plot <-  ggplot(sample, aes(x=Reference, y= Prediction)) +
    geom_tile(aes(fill = perc))+
    scale_fill_gradient(low = "white", high = Palette_col[1]) +
    geom_text(aes(label = paste0(round(perc,0), "%")), size = 3)+
    theme_custom()+
    guides(fill = "none")  
  
  return(plot)
}
PlotConfusionMatrix(df, c("GU"))
PlotConfusionMatrix(df, c("Test"))

##### F1 Score pour le niveau 1 ##### 
get_dataF1 <- function(data, level){
  Factors  <- data %>% 
    rename_at(vars(starts_with(paste0("ground_truth_", level))), ~ "TRUTH")%>%
    pull(TRUTH)%>%
    unique()%>%
    sort()
  
  sample <- data %>% 
    select(starts_with(paste0("ground_truth_", level)), (ends_with(paste0("predictions_", level))), Type)%>%
    rename_at(vars(starts_with(paste0("ground_truth_", level))), ~ "TRUTH")%>%
    rename_at(vars(ends_with(paste0("predictions_", level))), ~ "PRED")%>%
    mutate(TRUTH = factor(TRUTH, levels = Factors),
           PRED = factor(PRED, levels = Factors))
  
  sample_test <- sample %>% subset(Type == "Test")
  sample_gu <- sample %>% subset(Type == "GU")
  
  sample_test <- confusionMatrix(sample_test$TRUTH, sample_test$PRED)$byClass[,7]%>%
    as_tibble()%>%
    mutate(Class = Factors,
           Type = "Test")%>%
    rename(F1 = value)
  
  sample <- confusionMatrix(sample_gu$TRUTH, sample_gu$PRED)$byClass[,7]%>%
    as_tibble()%>%
    mutate(Class = Factors,
           Type = "GU")%>%
    rename(F1 = value)%>%
    bind_rows(sample_test)
  return(sample)
}
PlotF1 <- function(data, level, title){
  data <- get_dataF1(data, level)
  plot <-ggplot(data, aes(x=Class, y=F1, fill=Type))+
    ggtitle(title)+
    geom_bar(stat = "identity", position = position_dodge())+
    geom_text(aes(label=round(F1,2)), position=position_dodge(width=0.9), vjust=-0.25)+
    scale_fill_manual(values=c(Palette_col))+
    theme_custom()
  return(plot)
}
PlotF1(df, 1, "F1-score au niveau 1")

##### Aggrégation par moyenne ##### 
PlotAccuracies <- function(data, level, title){
  sample <- data %>%
    mutate(Results = .data[[paste0("predictions_", level)]] == .data[[paste0("ground_truth_", level)]]) %>%
    group_by(ground_truth_1, Type)%>%
    summarise(
      Accuracy = mean(Results)
    )
  
  plot <-ggplot(sample, aes(x=ground_truth_1, y=Accuracy, fill=Type))+
    ggtitle(title)+
    geom_bar(stat = "identity", position = position_dodge())+
    geom_text(aes(label=round(Accuracy,2)), position=position_dodge(width=0.9), vjust=-0.25)+
    scale_fill_manual(values=c(Palette_col))+
    theme_custom()
  return(plot)
}
PlotAccuracies(df, 1, "Accuracy au niveau 1")
PlotAccuracies(df, 5, "Accuracy au niveau 5 (aggrégation par moyenne)")

##### Aggrégation par moyenne avec proportion ##### 
PlotAccuracies_with_Shares <- function(data, type, level, title){
  
  TotalSize <- data%>%subset(Type %in% type)%>%nrow()
  sample <- data %>%
    subset(Type %in% type)%>%
    mutate(Results = .data[[paste0("predictions_", level)]] == .data[[paste0("ground_truth_", level)]]) %>%
    group_by(ground_truth_1, Type)%>%
    summarise(
      Accuracy = mean(Results),
      N = n(),
    )%>%
    mutate(Share = N/TotalSize*100,
           NewTitle = paste0(ground_truth_1, "\n(", round(Share, 1), "%)"))
  plot <-ggplot(sample, aes(x=NewTitle, y=Accuracy, fill=Type))+
    ggtitle(title)+
    geom_bar(stat = "identity", position = position_dodge())+
    geom_text(aes(label=round(Accuracy,2)), position=position_dodge(width=0.9), vjust=-0.25)+
    scale_fill_manual(values=c(Palette_col))+
    theme_custom()
  return(plot)
}
PlotAccuracies_with_Shares(df, "Test", 1, "Accuracy au niveau 1")
PlotAccuracies_with_Shares(df, "GU", 1, "Accuracy au niveau 1")
PlotAccuracies_with_Shares(df, "Test", 5, "Accuracy au niveau 5 (aggrégation par moyenne)")
PlotAccuracies_with_Shares(df, "GU", 5, "Accuracy au niveau 5 (aggrégation par moyenne)")

##### Construction de la base avec reprise des données ##### 
acc_w_rep <- function(data, q, type, level){
  
  accuracy <- data %>% 
    subset(Type %in% type)%>%
    select(probabilities, probabilities_k2, ends_with(paste0("_", level)))%>%
    rename_with(~ gsub(paste0("_", level,"$"), "", .x))%>%
    mutate(Score = probabilities - probabilities_k2,
           Results = ground_truth == predictions,
           Revisions = case_when(Score >= quantile(Score, q) ~ Results,
                                 TRUE ~ TRUE)
    )%>%
    pull(Revisions)%>%
    mean
  return(accuracy)
}

df_reprise <- tibble(Rate = character(), Level = character(), Type = character(), Accuracy = numeric())
for (type in c("GU", "Test")) {
  for (level in paste(c(1,3,5))) {
    for (rate in paste(seq(0,0.25,0.05))) {
      cat(type, level, rate, '\n')
      df_reprise <- df_reprise %>%
        add_row(Rate = rate, 
                Level = level,
                Type = type,
                Accuracy = acc_w_rep(df, 
                                     as.double(rate), 
                                     type, 
                                     as.double(level)
                )
        )
    }
  }
}

PlotReprise <- function(data, type, title){
  sample <- subset(data, Type %in% type)%>%
    mutate(Level = paste("Niveau", Level),
           Level = factor(Level, levels = paste("Niveau", paste(seq(5,1,-1)))),
           Rate = paste0(as.double(Rate) * 100, "%"),
           Rate = factor(Rate, levels = paste0(seq(0,25,5),"%"))
    )
  
  plot<- ggplot(sample , aes(x=Level, y=Accuracy, fill=Rate))+
    ggtitle(title)+
    geom_bar(stat = "identity", position = position_dodge())+
    geom_text(aes(label=round(Accuracy,2)), position=position_dodge(width=0.9), vjust=-0.25)+
    scale_fill_manual(values=c(Palette_col))+
    guides(fill=guide_legend(nrow=1, byrow=TRUE))+
    theme_custom()+
    theme(
      text = element_text(size = 16)
    )
  return(plot)
}
PlotReprise(df_reprise, "Test", "Accuracy en fonction du taux de reprise et du niveau d'aggrégation \n (base Test)")
PlotReprise(df_reprise, "GU", "Accuracy en fonction du taux de reprise et du niveau d'aggrégation \n (base GU)")

##### TOP K accuracy ##### 
top_k_acc <- function(data, topk, type, level){
  accuracy <- data %>% 
    subset(Type %in% type)%>%
    rename_at(vars(paste0("predictions_", level)), ~ paste0("predictions_", level, "_k1"))%>%
    select(starts_with(paste0("ground_truth_", level)), (starts_with(paste0("predictions_", level)) & ends_with(paste(1:topk))))%>%
    rename_at(vars(starts_with(paste0("ground_truth_", level))), ~ "TRUTH")%>%
    rename_at(vars((starts_with(paste0("predictions_", level)) & ends_with(paste(1:topk))) ), ~ paste0("K",1:topk))%>%
    rowwise() %>%
    mutate(Results = ifelse(topk == 1, any(TRUTH %in% K1),
                            ifelse(topk == 2, any(TRUTH %in% c(K1, K2)), 
                                   ifelse(topk == 3, any(TRUTH %in% c(K1, K2, K3)), 
                                          ifelse(topk == 4, any(TRUTH %in% c(K1, K2, K3, K4)),
                                                 ifelse(topk == 5, any(TRUTH %in% c(K1, K2, K3, K4, K5))))))))%>%
    pull(Results)%>%
    mean
  return(accuracy)
}

df_topk <- tibble(Topk = character(), Level = character(), Type = character(), Accuracy = numeric())
for (type in c("GU", "Test")) {
  for (level in paste(1:5)) {
    for (topk in paste(seq(1:5))) {
      cat(type, level, topk, '\n')
      df_topk <- df_topk %>%
        add_row(Topk = topk, 
                Level = level,
                Type = type,
                Accuracy = top_k_acc(df, 
                                     as.double(topk), 
                                     type, 
                                     as.double(level)
                )
        )
    }
  }
}

PlotTopk <- function(data, type, title){
  sample <- subset(data, Type %in% type & Level %in% c(1,3,5))%>%
    mutate(Level = paste("Niveau", Level),
           Level = factor(Level, levels = paste("Niveau", paste(seq(5,1,-1)))),
           Topk = paste("Top", Topk)
    )
  
  plot<- ggplot(sample , aes(x=Level, y=Accuracy, fill=Topk))+
    ggtitle(title)+
    geom_bar(stat = "identity", position = position_dodge())+
    geom_text(aes(label=round(Accuracy,2)), position=position_dodge(width=0.9), vjust=-0.25)+
    scale_fill_manual(values=c(Palette_col))+
    guides(fill=guide_legend(nrow=1, byrow=TRUE))+
    theme_custom()+
    theme(
      text = element_text(size = 16)
    )
  return(plot)
}
PlotTopk(df_topk, "Test", "Top-k accuracy pour chaque niveau d'aggrégation \n (base Test)")
PlotTopk(df_topk, "GU", "Top-k accuracy pour chaque niveau d'aggrégation \n (base GU)")


# Stats desc la ou y a le plus d'erreur

# analyse les metrics de la confusion matrix


##### Stats desc ##### 
sample <- df %>%
  group_by(Type, ground_truth_1)%>%
  summarise(
    N = n()
  )%>%
  mutate(Share = N/ifelse(Type =="GU", nrow(df_gu), nrow(df_test)))


ggplot(data = sample, aes(x=ground_truth_1, y=Share, fill=Type))+
  ggtitle('Proportion des différentes catégories')+
  geom_bar(stat = "identity", position = position_dodge())+
  geom_text(aes(label=round(Share*100,1)), position=position_dodge(width=0.9), vjust=-0.25)+
  scale_y_continuous(labels = scales::percent)+
  scale_fill_manual(values=c(Palette_col))+
  theme_custom()


##### Indice de confiance ##### 
type <- "Test"
data <- df%>%
  subset(Type == type)%>%
  mutate(Results = ground_truth_5 == predictions_5,
         Score = log(probabilities / probabilities_k2))%>%
  select(probabilities, probabilities_k2, Results, Type, Score)

thresholds <- df%>%
  subset(Type == type)%>%
  mutate(Results = ground_truth_5 == predictions_5,
         Score = log(probabilities / probabilities_k2))%>%
  select(probabilities, probabilities_k2, Results, Type, Score)%>%
  pull(Score)%>%
  quantile(seq(0.05, 0.25,0.05))

ggplot(data)+
  ggtitle("Distribution de l'indice de confiance en fonction du résultat de la prédiction")+
  geom_histogram(aes(x=Score, fill=Results, y=..density..),binwidth=.01, alpha=.5, position="identity") +
  scale_fill_manual(values=c(Palette_col))+
  geom_vline(xintercept = thresholds, color = rgb(83, 83, 83, maxColorValue = 255))+
  theme_custom()

##### Test matrice de confusion niveau 2 GU ##### 
sample <- df_gu

sample <- sample%>%
  mutate(ground_truth_2 =factor(ground_truth_2,levels = sort(unique(sample$ground_truth_2))),
         predictions_2 =factor(predictions_2, levels = sort(unique(sample$ground_truth_2))))

cm <- confusionMatrix(sample$ground_truth_2, df_gu$predictions_2)

cm$table %>%
  data.frame() %>% 
  mutate(Prediction = factor(Prediction, levels =sort(unique(sample$ground_truth_2)))) %>%
  group_by(Reference) %>% 
  mutate(
    total = sum(Freq),
    perc = Freq / total*100
  )%>%  
  ggplot(aes(Prediction, Reference)) +
  geom_tile(aes(fill = perc))+
  scale_fill_gradient(low = "white", high = Palette_col[1]) +
  # geom_text(aes(label = paste0(round(perc,0), "%")), size = 3)+
  theme_custom()+
  guides(fill = "none")  



##### Distribution du F1-score par classe ##### 
PlotDistribF1 <- function(data, level, type){
  Factors  <- data %>% 
    subset(Type == type) %>% 
    rename_at(vars(starts_with(paste0("ground_truth_", level))), ~ "TRUTH")%>%
    pull(TRUTH)%>%
    unique()%>%
    sort()
  
  sample <- data %>% 
    subset(Type == type) %>% 
    select(starts_with(paste0("ground_truth_", level)), (ends_with(paste0("predictions_", level))), Type)%>%
    rename_at(vars(starts_with(paste0("ground_truth_", level))), ~ "TRUTH")%>%
    rename_at(vars(ends_with(paste0("predictions_", level))), ~ "PRED")%>%
    mutate(TRUTH = factor(TRUTH, levels = Factors),
           PRED = factor(PRED, levels = Factors))
  
  prop <- sample %>%
    group_by(TRUTH)%>%
    summarise(
      N = n()
    )
  
  cm <- confusionMatrix(sample$TRUTH, sample$PRED)
  
  sample <- cm$byClass[,c(7)]%>%
    as_tibble()%>%
    mutate(Class = Factors,
           N = prop$N)
  
  
  sample <- sample %>%
    arrange(desc(N)) %>%
    mutate(N_cumulative = cumsum(N),
           cumulative_pct = N_cumulative / sum(N),
           Group = case_when(
             cumulative_pct < 0.50 ~ "Catégories très représentées",
             cumulative_pct < 0.75 ~ "Catégories assez représentées",
             cumulative_pct < 0.95 ~ "Catégories peu représentées",
             TRUE ~ "Catégories très peu représentées"
           ),
           Group = factor(Group, levels = c("Catégories très représentées",
                                            "Catégories assez représentées",
                                            "Catégories peu représentées",
                                            "Catégories très peu représentées"
                                            
           ))
    )
  ggplot(sample) + 
    ggtitle("Distribution du F1-score en fonction de la fréquence de chaque classe")+
    geom_histogram(aes(x=value, fill=Group),binwidth=.05, position="identity") +
    facet_wrap(. ~ Group,ncol = 1, strip.position="left") +
    theme_custom()+
    scale_fill_manual(values = Palette_col)+
    theme(strip.placement = "outside",
          strip.text.y = element_blank(),
          text = element_text(size = 16)
    )+
    guides(fill=guide_legend(nrow=2, byrow=TRUE))
  
}
PlotDistribF1(df, 5, "GU")

##### Nombre de classe selon un seuil de F1 ##### 
PlotF1Inf <- function(data, threshold){
  Factors  <- data %>% 
    rename_at(vars(starts_with(paste0("ground_truth_", 5))), ~ "TRUTH")%>%
    pull(TRUTH)%>%
    unique()%>%
    sort()
  
  sample <- data %>% 
    select(starts_with(paste0("ground_truth_", 5)), (ends_with(paste0("predictions_", 5))), Type)%>%
    rename_at(vars(starts_with(paste0("ground_truth_", 5))), ~ "TRUTH")%>%
    rename_at(vars(ends_with(paste0("predictions_", 5))), ~ "PRED")%>%
    mutate(TRUTH = factor(TRUTH, levels = Factors),
           PRED = factor(PRED, levels = Factors))
  
  prop <- sample %>%
    group_by(TRUTH)%>%
    summarise(
      N = n()
    )
  
  cm <- confusionMatrix(sample$TRUTH, sample$PRED)
  
  sample <- cm$byClass[,c(7)]%>%
    as_tibble()%>%
    mutate(Class = Factors,
           N = prop$N)
  
  plot <- data%>%
    select(ground_truth_1, ground_truth_5)%>%
    distinct(ground_truth_1,ground_truth_5)%>%
    rename(Class = ground_truth_5)%>%
    full_join(sample)%>%
    rename(Class_1 = ground_truth_1)%>%
    subset(value < threshold)%>%
    group_by(Class_1)%>%
    summarise(
      N = n()
    )%>%
    ggplot(aes(x = Class_1, y= N))+
    ggtitle(paste("Nombre de classe dont le F1 score est inférieur à", threshold))+
    geom_bar(stat = "identity", position = position_dodge(), fill=Palette_col[1])+
    geom_text(aes(label=N), position=position_dodge(width=0.9), vjust=-0.25)+
    theme_custom()+  
    theme(
      text = element_text(size = 16)
    )
  return(plot)
}
PlotF1Inf(df, 0.3)
##### Matrice pour l'indice de confiance ##### 
PlotMatConfidence <- function(data, q){
  
  plot <- data%>%
    mutate(Results = ground_truth_5 == predictions_5,
           Score = probabilities - probabilities_k2)%>%
    select(Results, Type, Score)%>%
    mutate(IsInf = Score <= quantile(Score, q))%>%
    group_by(Results, IsInf)%>%
    summarise(
      N = n()
    )%>%
    group_by(Results)%>%
    mutate(
      total = sum(N),
      perc = N / total*100,
      IsInf = factor(IsInf, level= c(T,F)),
      Results = factor(Results, level= c(F,T))
      
    )%>%
    ggplot(aes(x=IsInf, y= Results)) +
    ggtitle(paste(q*100, "% de reprise"))+
    geom_tile(aes(fill = N))+
    scale_fill_gradient(low = "white", high = Palette_col[1]) +
    geom_text(aes(label = paste0(round(perc,0), "%")), size = 3)+
    theme_custom()+
    guides(fill = "none")+
    theme(
      axis.title.x=element_text(color = rgb(83, 83, 83, maxColorValue = 255)),      
      axis.title.y=element_text(angle=90, color = rgb(83, 83, 83, maxColorValue = 255)),
      
    )+
    xlab("Below threshold")+
    ylab("Prediction")
  
  return(plot)
}
Plot05 <- PlotMatConfidence(df, 0.05)
Plot10 <- PlotMatConfidence(df, 0.10)
Plot15 <- PlotMatConfidence(df, 0.15)
Plot20 <- PlotMatConfidence(df, 0.20)
Plot25 <- PlotMatConfidence(df, 0.25)
plot_grid(Plot05 + theme(plot.margin = unit(c(0,0,0,0), "cm")) ,
          Plot10 + theme(plot.margin = unit(c(0,0,0,0), "cm")),
          Plot15 + theme(plot.margin = unit(c(0,0,0,0), "cm")),
          Plot20 + theme(plot.margin = unit(c(0,0,0,0), "cm")),
          Plot25 + theme(plot.margin = unit(c(0,0,0,0), "cm")),
          align = "h", ncol = 2, vjust = -0.8)


# Regarder les density pour chaque classe
df%>%
  mutate(Results = ground_truth_5 == predictions_5,
         Score = probabilities - probabilities_k2)%>%
  select(Results, Type, Score, predictions_1)%>%
  ggplot()+
  ggtitle("Distribution de l'indice de confiance en fonction du résultat de la prédiction")+
  geom_histogram(aes(x=Score, fill=Results), binwidth=.01, alpha=.75, position="identity") +
  scale_fill_manual(values=c(Palette_col))+
  facet_wrap(. ~ predictions_1, ncol = 2, strip.position="left", scales = "free") +
  theme_custom()+
  theme(strip.placement = "outside"
  )


niv1 <- "K"
df%>%
  subset(predictions_1 == niv1)%>%
  mutate(Results = ground_truth_5 == predictions_5,
         Score = probabilities - probabilities_k2)%>%
  select(Results, Type, Score, predictions_1)%>%
  ggplot()+
  ggtitle(niv1)+
  geom_histogram(aes(x=Score, fill=Results), binwidth=.01, alpha=.5, position="identity") +
  scale_fill_manual(values=c(Palette_col))+
  theme_custom()+
  theme(strip.placement = "outside"
  )

df%>%
  subset(predictions_1 == niv1)%>%
  mutate(Results = ground_truth_5 == predictions_5,
         Score = probabilities - probabilities_k2)%>%
  select(Results, Type, Score, predictions_1)


#### Algo ####
step <- 5000
data <- df
cat <- "K"
level <- 1
gain_recall(df, 5000, "A",1)
gain_accuracy(df, 5000, "C",1)

gain_recall <- function(data, step, cat, level){
  
  sample <- data%>%
    rename_at(vars(ends_with(paste0("predictions_", level))), ~ "PRED")%>%
    subset(PRED == cat)%>%
    mutate(Results = ground_truth_5 == predictions_5,
           Score = probabilities - probabilities_k2)%>%
    select(Results, Score)
  
  N <- nrow(data)
  n <- nrow(sample)
  
  old_recall <- sample%>%
    pull(Results)%>%
    mean()
  
  new_recall <- sample%>%
    arrange(desc(Score))%>%
    slice_head(n = ifelse(n - step>0, n - step, n))%>%
    pull(Results)%>%
    mean()
  
  return( n/N * (new_recall - old_recall) )
}
gain_accuracy <- function(data, step, cat, level){
  
  sample_full <- data%>%
    rename_at(vars(ends_with(paste0("predictions_", level))), ~ "PRED")%>%
    mutate(Results = ground_truth_5 == predictions_5,
           Score = probabilities - probabilities_k2)%>%
    select(Results, Score, liasseNb, PRED)
  
  sample_class <- sample_full%>%
    subset(PRED == cat)
  
  idx2Remove <- sample_class%>%
    select(liasseNb, Score)%>%
    arrange(desc(Score))%>%
    slice_tail(n = ifelse(step<n, step, n))%>%
    pull(liasseNb)
  
  n <- nrow(sample_class)
  
  old_accuracy <- sample_full%>%
    pull(Results)%>%
    mean()
  
  new_accuracy <- sample_full%>%
    mutate(Results = case_when(
      liasseNb %in% idx2Remove ~ T,
      TRUE ~ Results
    ))%>%
    pull(Results)%>%
    mean()
  
  return( new_accuracy - old_accuracy )
}

sample <- df%>%
  select(predictions_2, ground_truth_2, predictions_5, ground_truth_5, probabilities, probabilities_k2, liasseNb)
N <- nrow(df)
step = 2500
Nrevised = 0 
iter = 0
list_New <- c()
q <- 0.1
level <- 2
Target2Revise <- floor(N*q)

while (Nrevised < Target2Revise) {
  step <- ifelse(Nrevised + step > Target2Revise, Target2Revise - Nrevised, step)
  AllModalities <- sample[paste0("ground_truth_",level)]%>% pull()%>%unique()%>%sort()
  results <- sapply(AllModalities, gain_accuracy, data = sample, step = step, level = level, simplify = FALSE)
  class2revise <- names(results[order(unlist(results),decreasing=TRUE)])[1]
  
  n <- sample%>%
    rename_at(vars(ends_with(paste0("predictions_", level))), ~ "PRED")%>%
    subset(PRED == class2revise)%>%
    nrow()
  
  idx2Remove <- sample%>%
    rename_at(vars(ends_with(paste0("predictions_", level))), ~ "PRED")%>%
    subset(PRED == class2revise)%>%
    mutate(Results = ground_truth_5 == predictions_5,
           Score = probabilities - probabilities_k2)%>%
    select(liasseNb, Score)%>%
    arrange(desc(Score))%>%
    slice_tail(n = ifelse(step<n, step, n))%>%
    pull(liasseNb)
  
  sample <- sample %>%
    filter(!(liasseNb %in% idx2Remove))
  
  Nrevised <- Nrevised + ifelse(step<n, step, n)
  iter <- iter + 1
  list_New <- c(list_New, idx2Remove)
  cat("*** Iteration ", iter, " : ", ifelse(step<n, step, n), " revised in class : ", class2revise, "\n")
}

##### Calculer l'accuracy des supprimés #####
listOld <- df %>% 
  mutate(Score = probabilities - probabilities_k2)%>%
  subset(Score <= quantile(Score, 0.1))%>%
  pull(liasseNb)

removed_old <- df %>%
  filter(liasseNb %in% listOld)%>%
  select(ground_truth_5, predictions_5)%>%
  mutate(Results = ground_truth_5 == predictions_5)%>%
  pull(Results)%>%
  mean()

removed_new <- df %>%
  filter(liasseNb %in% list_New)%>%
  select(ground_truth_5, predictions_5)%>%
  mutate(Results = ground_truth_5 == predictions_5)%>%
  pull(Results)%>%
  mean()

new_accuracy <- df%>%
  mutate(
    Results = ground_truth_5 == predictions_5,
    Results = case_when(
      liasseNb %in% list_New ~ T,
      TRUE ~ Results
    ))%>%
  pull(Results)%>%
  mean()

old_accuracy <- df %>%
  mutate(
    Results = ground_truth_5 == predictions_5,
    Results = case_when(
      liasseNb %in% listOld ~ T,
      TRUE ~ Results
    ))%>%
  pull(Results)%>%
  mean()

#### Run algorithm ####
gain_recall <- function(data, step, cat){
  sample_class <- data%>%
    subset(PRED == cat)
  
  N <- nrow(data)
  n <- nrow(sample_class)
  
  idx2Remove <- sample_class%>%
    select(liasseNb, Score)%>%
    arrange(desc(Score))%>%
    slice_tail(n = ifelse(step<n, step, n))%>%
    pull(liasseNb)
  
  old_recall <- sample_class%>%
    pull(Results)%>%
    mean()

  new_recall <- sample_class%>%
    mutate(Results = case_when(
      liasseNb %in% idx2Remove ~ T,
      TRUE ~ Results
    ))%>%
    pull(Results)%>%
    mean()
  
  return( n/N * (new_recall - old_recall) )
}
get_data <- function(data, level){
  sample_full <- data%>%
    rename_at(vars(ends_with(paste0("predictions_", level))), ~ "PRED")%>%
    rename_at(vars(ends_with(paste0("ground_truth_", level))), ~ "TRUTH")%>%
    mutate(Results = ground_truth_5 == predictions_5,
           Score = probabilities - probabilities_k2,
           Revised = F)%>%
    select(Results, Score, liasseNb, PRED, TRUTH, Revised)
  
  return(sample_full)
}
gain_accuracy <- function(data, step, cat){
  
  sample_class <- data%>%
    subset(PRED == cat)
  
  n <- nrow(sample_class)
  
  idx2Remove <- sample_class%>%
    select(liasseNb, Score)%>%
    arrange(desc(Score))%>%
    slice_tail(n = ifelse(step<n, step, n))%>%
    pull(liasseNb)
  
  old_accuracy <- data%>%
    pull(Results)%>%
    mean()
  
  new_accuracy <- data%>%
    mutate(Results = case_when(
      liasseNb %in% idx2Remove ~ T,
      TRUE ~ Results
    ))%>%
    pull(Results)%>%
    mean()
  
  return( list(Gain = new_accuracy - old_accuracy, Liasses = idx2Remove, Size = n) )
}
run_algo <- function(data, method, step, level, q){
  
  # Initialisation
  Nrevised = 0 
  iter = 0
  sample <- get_data(data, level)
  N <- nrow(data)
  Target2Revise <- floor(N*q)
  list2Revise <- c()
  
  while (Nrevised < Target2Revise) {
    start_time <- Sys.time()
    step <- ifelse(Nrevised + step > Target2Revise, Target2Revise - Nrevised, step)
    
    AllModalities <- sample%>% pull(TRUTH)%>%unique()%>%sort()
    
    if (method == "Accuracy") {
      # Compute all gain in accuracy for all classes
      results <- sapply(AllModalities, gain_accuracy, data = sample, step = step, simplify = FALSE)
    }else{
      # Compute all gain in recall for all classes
      results <- sapply(AllModalities, gain_recall, data = sample, step = step, simplify = FALSE)
    }
    
    # Retrieve the gain accuracies for all classes
    accuracies <- sapply(results,"[[",1, simplify = FALSE)
    # Extract the class that has the largest marginal gain in accuracy
    class2revise <- names(accuracies[order(unlist(accuracies),decreasing=TRUE)])[1]
    # Extract the indexes that are needed to remove to get this accuracy
    idx2Remove <- as.vector(sapply(results[class2revise],"[[",2))
    # Extract the size of the revised class
    n <- as.vector(sapply(results[class2revise],"[[",3))
    
    sample <- sample %>%
      mutate(Score = case_when(liasseNb %in% idx2Remove ~ 1,
                               T ~ Score
      ),
      Results = case_when(liasseNb %in% idx2Remove ~ T,
                          T ~ Results
      ),
      Revised = case_when(liasseNb %in% idx2Remove ~ T,
                          T ~ Revised
      )
      )
    
    
    Nrevised <- Nrevised + ifelse(step<n, step, n)
    iter <- iter + 1
    list2Revise <- c(list2Revise, idx2Remove)
    end_time <- Sys.time()
    cat("*** Iteration ", iter, " : ", ifelse(step<n, step, n), " revised in class : ", class2revise, "in ", end_time - start_time, "sec \n")
  }
  
  return(list2Revise)
}

list_New <- run_algo(df, "Accuracy", 100, 3, 0.1)

new_accuracy <- df%>%
  mutate(
    Results = ground_truth_5 == predictions_5,
    Results = case_when(
      liasseNb %in% list_New ~ T,
      TRUE ~ Results
    ))%>%
  pull(Results)%>%
  mean()

## Plot histograms ####
data <- df
level <- 2
modality<-"19"
get_thresholds <- function(data, listOld, list_New, level){
  temp <- data %>%
    rename_at(vars(ends_with(paste0("predictions_", level))), ~ "PRED")%>%
    filter(liasseNb %in% listOld)%>%
    group_by(PRED)%>%
    summarise(
      Nold = n()
    )
  
  thresholds <- data %>%
    rename_at(vars(ends_with(paste0("predictions_", level))), ~ "PRED")%>%
    filter(liasseNb %in% list_New)%>%
    group_by(PRED)%>%
    summarise(
      N = n()
    )%>%
    full_join(temp)
  return(thresholds)
}
Plot_histograms <- function(data, level, modality, listOld, list_New){
  
  seuils <- get_thresholds(data, listOld, list_New, level)%>%
    subset(PRED == modality)%>%
    replace_na(list(N = 1, Nold = 1))%>%
    select(N, Nold)%>%
    as.numeric()
  
  sample <- data %>%
    rename_at(vars(ends_with(paste0("ground_truth_", level))), ~ "TRUTH")%>%
    rename_at(vars(ends_with(paste0("predictions_", level))), ~ "PRED")%>%
    subset(PRED == modality)%>%
    mutate(Results = ground_truth_5 == predictions_5, # ne va pas marcher si level = 5
           Score = probabilities - probabilities_k2,
           Results = factor(Results, level = c(T,F)))%>%
    select(Results, Type, Score, PRED)
  
  
  Thresholds <- sample %>% 
    arrange(Score)%>%
    slice(seuils)%>%
    pull(Score) 
  
  ypos <- sample%>%
    count(round(Score, 2))%>%
    pull(n)%>%
    max()*7/10
  
  if (length(Thresholds)==0) {
    xpos <- c(NA,NA)
  }else if (Thresholds[1] < Thresholds[2]) {
    xpos <- Thresholds + c(-0.05, 0.05)
  }else{
    xpos <-Thresholds + c(0.05, -0.05)
  }
  
  plot<- ggplot(sample)+
    ggtitle(modality)+
    geom_histogram(aes(x=Score, fill=Results), binwidth=.01, alpha=.5, position="identity") +
    scale_fill_manual(values=c(Palette_col[2],Palette_col[1]))+
    geom_vline(xintercept = Thresholds, color = rgb(83, 83, 83, maxColorValue = 255))+
    {if (length(Thresholds)!=0)
      annotate("text", x = xpos, y = ypos, label = c("New", "Old"))
    }+
    theme_custom()+
    theme(      
      plot.title=element_text(size=10,face="plain"),
    )+
    guides(fill = "none")  
  
  
  return(plot)
}
Plot_histograms(df, 2, "19", listOld, list_New)
ListPlots <- sapply(sort(unique(df$ground_truth_3)), Plot_histograms, data = df, level = 3, listOld = listOld, list_New = list_New, simplify = FALSE)
plot_grid(plotlist = ListPlots[1:4], align = "h", ncol = 2, vjust = -0.8)
plot_grid(plotlist = ListPlots[5:8], align = "h", ncol = 2, vjust = -0.8)
plot_grid(plotlist = ListPlots[9:12], align = "h", ncol = 2, vjust = -0.8)
plot_grid(plotlist = ListPlots[13:16], align = "h", ncol = 2, vjust = -0.8)
plot_grid(plotlist = ListPlots[17:20], align = "h", ncol = 2, vjust = -0.8)
plot_grid(plotlist = ListPlots[29:32], align = "h", ncol = 2, vjust = -0.8)
plot_grid(plotlist = ListPlots[33:36], align = "h", ncol = 2, vjust = -0.8)



###Probleme avec la modalité 19 Je comprends pas le grpahique et les données



w<-df%>%
    mutate(Score = probabilities - probabilities_k2,
     Results = ground_truth_5 == predictions_5,
     Revisions_Old = case_when(Score >= quantile(Score, 0.1) ~ Results,
                               TRUE ~ TRUE),
     Revisions_New = case_when(liasseNb %in% list_New ~ TRUE,
                               TRUE ~ Results)
    )%>%
    select(liasseNb, ground_truth_5, predictions_5, Score, Results, Revisions_Old, Revisions_New)


df_new <-df%>%
  mutate(predictions_1 = case_when(liasseNb %in% list_New ~ ground_truth_1,
                                   TRUE ~ predictions_1),
         predictions_2 = case_when(liasseNb %in% list_New ~ ground_truth_2,
                                   TRUE ~ predictions_2),
         predictions_3 = case_when(liasseNb %in% list_New ~ ground_truth_3,
                                   TRUE ~ predictions_3),
         predictions_4 = case_when(liasseNb %in% list_New ~ ground_truth_4,
                                   TRUE ~ predictions_4),
         predictions_5 = case_when(liasseNb %in% list_New ~ ground_truth_5,
                                   TRUE ~ predictions_5),
  )


list_Old <- df %>% 
  mutate(Score = probabilities - probabilities_k2)%>%
  subset(Score < quantile(Score, 0.1))%>%
  pull(liasseNb)


df_old <- df%>%
  mutate(predictions_1 = case_when(liasseNb %in% list_Old ~ ground_truth_1,
                                   TRUE ~ predictions_1),
         predictions_2 = case_when(liasseNb %in% list_Old ~ ground_truth_2,
                                   TRUE ~ predictions_2),
         predictions_3 = case_when(liasseNb %in% list_Old ~ ground_truth_3,
                                   TRUE ~ predictions_3),
         predictions_4 = case_when(liasseNb %in% list_Old ~ ground_truth_4,
                                   TRUE ~ predictions_4),
         predictions_5 = case_when(liasseNb %in% list_Old ~ ground_truth_5,
                                   TRUE ~ predictions_5),
  )

get_dataF1 <- function(data, level){
  Factors  <- data %>% 
    rename_at(vars(starts_with(paste0("ground_truth_", level))), ~ "TRUTH")%>%
    pull(TRUTH)%>%
    unique()%>%
    sort()
  
  sample <- data %>% 
    select(starts_with(paste0("ground_truth_", level)), (ends_with(paste0("predictions_", level))), Type)%>%
    rename_at(vars(starts_with(paste0("ground_truth_", level))), ~ "TRUTH")%>%
    rename_at(vars(ends_with(paste0("predictions_", level))), ~ "PRED")%>%
    mutate(TRUTH = factor(TRUTH, levels = Factors),
           PRED = factor(PRED, levels = Factors))
  

  sample <- confusionMatrix(sample$TRUTH, sample$PRED)$byClass[,7]%>%
    as_tibble()%>%
    mutate(Class = Factors)%>%
    rename(F1 = value)
  
  return(sample)
}

w <- get_dataF1(df_new, 3)%>%
  mutate(F1_old = get_dataF1(df_old, 3)$F1,
         Diff = F1 - F1_old)%>%
  full_join(
    df_new%>%
      group_by(ground_truth_3)%>%
      summarise(
        N = n()
      )%>%
      rename(Class = ground_truth_3)
  )









