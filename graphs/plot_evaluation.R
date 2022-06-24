rm(list = ls())
setwd("~/codification-ape/graphs")
install.packages("tibble")
install.packages("caret")
library(aws.s3)
library(dplyr)
library(ggplot2)
library(caret)
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
PlotF1 <- function(data, level, title){
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
  
  plot <-ggplot(sample, aes(x=Class, y=F1, fill=Type))+
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
           Results = ground_truth == predictions)%>%
    subset(Score >= quantile(Score, q)
           #Revisions = case_when(Score >= quantile(Score, q) ~ Results,
            #                     TRUE ~ TRUE)
    )%>%
    pull(Results)%>%
    mean
  return(accuracy)
}

df_reprise <- tibble(Rate = character(), Level = character(), Type = character(), Accuracy = numeric())
for (type in c("GU", "Test")) {
  for (level in paste(1:5)) {
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
    theme_custom()
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
  sample <- subset(data, Type %in% type)%>%
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
    theme_custom()
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

data <- df%>%
  mutate(Results = ground_truth_5 == predictions_5,
         Score = probabilities - probabilities_k2)%>%
  select(probabilities, probabilities_k2, Results, Type, Score)

thresholds <- df%>%
  mutate(Results = ground_truth_5 == predictions_5,
         Score = probabilities - probabilities_k2)%>%
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

##### Graphique de reprise des données ##### 

acc_w_rep <- function(sample, q, level){
  N <- nrow(sample)
  model_sample <- sample %>% 
    subset(probabilities >= quantile(probabilities, q))
  n <- nrow(model_sample)
  accuracy <- (sum(model_sample[paste0("ground_truth_",level)] == 
                     model_sample[paste0("predictions_", level)], na.rm = T) + (N-n))/N
  return(accuracy)
}

acc_w_rep(df_gu, 0.1, "2")


df %>% 
  group_by(Type)%>%
  summarise(
    Threshold = quantile(probabilities, seq(0.05, 0.25, 0.05))
  )%>%
  mutate(Q = seq(0.05, 0.25, 0.05))


##### 

##### F1 Score pour le niveau 1 ##### 
PlotF1 <- function(data, level, title){
  data <- df
  level <- 5
  type <- "Test"
  
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
  
  samplee <- confusionMatrix(sample$TRUTH, sample$PRED)$byClass[,7]%>%
    as_tibble()%>%
    mutate(Class = Factors,
           N = prop$N)%>%
    rename(F1 = value)

  w <- samplee %>% 
    subset(N > quantile(N, 0.80))
data<-  samplee%>%
    subset((F1<0.75) & (N>50))
  
  plot <-ggplot(w, aes(x=N, y=F1 ))+
    ggtitle('')+
    geom_point()+
    scale_fill_manual(values=c(Palette_col))+
    theme_custom()
  return(plot)
}
PlotF1(df, 5, "F1-score au niveau 1")




data <- df
level <- 5
type <- "Test"

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

ggplot(sample, aes(x=Precision, y=Recall, size = N, alpha=N))+
  ggtitle('')+
  geom_point(color = Palette_col[1])+
  scale_x_continuous(limits = c(0, 1))+
  scale_y_continuous(limits = c(0, 1))+
  theme_custom()



violin_df <- samplee %>%
  arrange(desc(N)) %>%
  mutate(N_cumulative = cumsum(N)) %>%
  mutate(cumulative_pct = N_cumulative / sum(samplee$N)) %>%
  mutate(class = case_when(
    cumulative_pct < 0.4 ~ "Catégories très représentées",
    cumulative_pct < 0.7 ~ "Catégories assez représentées",
    cumulative_pct < 0.9 ~ "Catégories peu représentées",
    TRUE ~ "Catégories très peu représentées"
  )) %>%
  mutate(class = as.factor(class))

violin_df %>%
  ggplot(aes(x=class, y=F1)) + 
  geom_violin(draw_quantiles = )

violin_df %>%
  group_by(class) %>%
  summarise(N_tot = sum(N),
            count = n())



















